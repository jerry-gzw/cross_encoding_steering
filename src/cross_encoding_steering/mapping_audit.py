from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .config import ModelConfig
from .decomposition import contrast_directions, decompose_directions
from .io import write_tables
from .metrics import js_divergence
from .statistics import holm_adjust, paired_sign_flip_pvalue, paired_weighted_sign_flip_pvalue
from .steering import (
    cleanup_model,
    collect_choice_probs,
    collect_choice_probs_and_activations,
    collect_choice_probs_and_position_activations,
    decoder_layers,
    load_tokenizer_and_model,
    patched_choice_probs,
    resolve_layer_index,
)


def _merge_config_mappings(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Merge mapping-audit configs while replacing lists atomically."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config_mappings(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_mapping_audit_config(config_path: Path) -> dict[str, Any]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base_value = data.pop("base_config", None)
    if not base_value:
        return data
    base_path = Path(str(base_value)).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    if not base_path.exists():
        raise FileNotFoundError(
            f"Mapping-audit base_config does not exist: {base_path}"
        )
    return _merge_config_mappings(_load_mapping_audit_config(base_path.resolve()), data)


PAPER_MODES = [
    "raw_canonical_direction",
    "raw_position_norm_matched",
    "shared_canonical_direction",
    "residual_canonical_direction",
    "loco_shared_canonical_direction",
    "loco_residual_canonical_direction",
]
CONTROL_MODES = [
    "random_direction_control",
    "wrong_direction_control",
    "zero_direction_control",
]


@dataclass(frozen=True)
class MappingDefinition:
    name: str
    option_order: tuple[str, ...]
    option_texts: dict[str, str]
    header_lines: tuple[str, ...] = ()
    family: str = "semantic_words"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MappingDefinition":
        return cls(
            name=str(data["name"]),
            option_order=tuple(str(value) for value in data["option_order"]),
            option_texts={str(key): str(value) for key, value in data["option_texts"].items()},
            header_lines=tuple(str(value) for value in data.get("header_lines", [])),
            family=str(data.get("family", "semantic_words")),
        )


@dataclass(frozen=True)
class PromptField:
    column: str
    prefix: str
    optional: bool = False


@dataclass(frozen=True)
class RankedChoiceDatasetConfig:
    dataset_name: str
    items_path: Path
    pair_id_column: str
    contrast_column: str
    split_column: str
    label_column: str
    label_ranks: dict[str, float]
    canonical_mapping: str
    prompt_header: str
    prompt_fields: tuple[PromptField, ...]
    question: str
    answer_instruction: str
    extraction_filters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    train_split: str = "train"
    validation_split: str = "val"
    test_split: str = "test"
    max_train_pairs_per_contrast: int = 0
    max_validation_pairs_per_contrast: int = 0
    max_test_pairs_per_contrast: int = 0


@dataclass(frozen=True)
class MappingAuditModelConfig:
    model: ModelConfig
    locked_layer: int
    locked_alpha: float


@dataclass(frozen=True)
class MappingAuditStatisticsConfig:
    n_boot: int = 10_000
    n_permutations: int = 20_000
    confidence: float = 0.95


@dataclass(frozen=True)
class MappingAuditRuntimeConfig:
    batch_size: int = 2
    max_length: int = 512
    seed: int = 13
    continue_on_error: bool = True
    force_rerun: bool = False


@dataclass(frozen=True)
class FixedDirectionMappingAuditConfig:
    project_root: Path
    output_dir: Path
    dataset: RankedChoiceDatasetConfig
    mappings: tuple[MappingDefinition, ...]
    models: tuple[MappingAuditModelConfig, ...]
    subspace_dim: int
    extraction_position: str
    injection_position: str
    enabled_modes: tuple[str, ...]
    statistics: MappingAuditStatisticsConfig
    runtime: MappingAuditRuntimeConfig

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "FixedDirectionMappingAuditConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        dataset_data = dict(data["dataset"])
        dataset_data["items_path"] = resolve(str(dataset_data["items_path"]))
        dataset_data["label_ranks"] = {
            str(key): float(value) for key, value in dataset_data["label_ranks"].items()
        }
        dataset_data["prompt_fields"] = tuple(
            PromptField(
                column=str(item["column"]),
                prefix=str(item.get("prefix", "")),
                optional=bool(item.get("optional", False)),
            )
            for item in dataset_data.get("prompt_fields", [])
        )
        dataset_data["extraction_filters"] = {
            str(key): tuple(str(value) for value in (values if isinstance(values, list) else [values]))
            for key, values in dataset_data.get("extraction_filters", {}).items()
        }
        dataset = RankedChoiceDatasetConfig(**dataset_data)

        mappings = tuple(MappingDefinition.from_mapping(item) for item in data.get("mappings", []))
        if not mappings:
            raise ValueError("Fixed-direction mapping audit requires at least one mapping")
        mapping_names = [mapping.name for mapping in mappings]
        if len(mapping_names) != len(set(mapping_names)):
            raise ValueError("Mapping names must be unique")
        if dataset.canonical_mapping not in mapping_names:
            raise ValueError(f"Canonical mapping {dataset.canonical_mapping!r} is not configured")
        labels = set(dataset.label_ranks)
        for mapping in mappings:
            if set(mapping.option_order) != labels or set(mapping.option_texts) != labels:
                raise ValueError(
                    f"Mapping {mapping.name!r} must contain every configured semantic label exactly once"
                )

        overrides = model_source_overrides or {}
        models = []
        for item in data.get("models", []):
            model_data = dict(item)
            locked_layer = int(model_data.pop("locked_layer"))
            locked_alpha = float(model_data.pop("locked_alpha", 0.8))
            alias = str(model_data["alias"])
            if alias in overrides:
                model_data["source"] = overrides[alias]
            models.append(
                MappingAuditModelConfig(
                    model=ModelConfig.from_mapping(model_data),
                    locked_layer=locked_layer,
                    locked_alpha=locked_alpha,
                )
            )
        if not models:
            raise ValueError("Fixed-direction mapping audit requires at least one model")

        extraction_position = str(data.get("extraction_position", "pre_answer"))
        if extraction_position not in {"pre_answer", "scenario_end"}:
            raise ValueError("extraction_position must be pre_answer or scenario_end")
        injection_position = str(data.get("injection_position", "pre_answer"))
        if injection_position not in {"pre_answer", "scenario_end"}:
            raise ValueError("injection_position must be pre_answer or scenario_end")
        enabled_modes = tuple(
            str(value)
            for value in data.get("enabled_modes", [*PAPER_MODES, *CONTROL_MODES])
        )
        unknown_modes = sorted(set(enabled_modes) - set(PAPER_MODES) - set(CONTROL_MODES))
        if unknown_modes:
            raise ValueError(f"Unknown mapping-audit modes: {unknown_modes}")
        if not enabled_modes:
            raise ValueError("enabled_modes must contain at least one direction mode")
        output_dir = resolve(
            str(data.get("output_dir", "outputs/fixed_direction_mapping_audit"))
        )
        return cls(
            project_root=root,
            output_dir=output_dir,
            dataset=dataset,
            mappings=mappings,
            models=tuple(models),
            subspace_dim=int(data.get("subspace_dim", 1)),
            extraction_position=extraction_position,
            injection_position=injection_position,
            enabled_modes=enabled_modes,
            statistics=MappingAuditStatisticsConfig(**data.get("statistics", {})),
            runtime=MappingAuditRuntimeConfig(**data.get("runtime", {})),
        )


def mapping_choice_map(mapping: MappingDefinition) -> dict[str, int]:
    return {label: index for index, label in enumerate(mapping.option_order)}


def build_ranked_choice_prompt(
    row: pd.Series,
    dataset: RankedChoiceDatasetConfig,
    mapping: MappingDefinition,
) -> str:
    lines = [build_scenario_context(row, dataset)]
    lines.extend(mapping.header_lines)
    lines.append(dataset.question)
    for index, semantic_label in enumerate(mapping.option_order):
        letter = chr(ord("A") + index)
        lines.append(f"{letter}. {mapping.option_texts[semantic_label]}")
    lines.extend([dataset.answer_instruction, "Answer:"])
    return "\n".join(lines)


def build_scenario_context(
    row: pd.Series,
    dataset: RankedChoiceDatasetConfig,
) -> str:
    lines = [dataset.prompt_header]
    for field_spec in dataset.prompt_fields:
        value = row.get(field_spec.column, "")
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            if field_spec.optional:
                continue
            raise ValueError(f"Required prompt field {field_spec.column!r} is empty")
        lines.append(f"{field_spec.prefix}{text}")
    return "\n".join(lines)


def _sample_pairs(
    rows: pd.DataFrame,
    dataset: RankedChoiceDatasetConfig,
    *,
    seed: int,
) -> pd.DataFrame:
    limits = {
        dataset.train_split: int(dataset.max_train_pairs_per_contrast),
        dataset.validation_split: int(dataset.max_validation_pairs_per_contrast),
        dataset.test_split: int(dataset.max_test_pairs_per_contrast),
    }
    selected_ids: list[str] = []
    pair_table = rows[[dataset.pair_id_column, dataset.contrast_column, dataset.split_column]].drop_duplicates()
    for (contrast, split), group in pair_table.groupby(
        [dataset.contrast_column, dataset.split_column], sort=True
    ):
        limit = limits.get(str(split), 0)
        if limit > 0 and len(group) > limit:
            random_state = seed + sum(ord(char) for char in f"{contrast}:{split}")
            group = group.sample(n=limit, random_state=random_state)
        selected_ids.extend(group[dataset.pair_id_column].astype(str).tolist())
    return rows[rows[dataset.pair_id_column].astype(str).isin(selected_ids)].copy()


def _resplit_pairs(
    rows: pd.DataFrame,
    dataset: RankedChoiceDatasetConfig,
    *,
    seed: int,
) -> pd.DataFrame:
    split_sizes = {
        dataset.train_split: int(dataset.max_train_pairs_per_contrast),
        dataset.validation_split: int(dataset.max_validation_pairs_per_contrast),
        dataset.test_split: int(dataset.max_test_pairs_per_contrast),
    }
    if any(size <= 0 for size in split_sizes.values()):
        raise ValueError(
            "Fresh pair resplitting requires positive train, validation, and test limits"
        )
    requested = sum(split_sizes.values())
    assignments = []
    pair_table = rows[[dataset.pair_id_column, dataset.contrast_column]].drop_duplicates()
    for contrast, group in pair_table.groupby(dataset.contrast_column, sort=True):
        if len(group) < requested:
            raise ValueError(
                f"Contrast {contrast!r} has {len(group)} pairs, but fresh resplitting "
                f"requires {requested}"
            )
        digest = hashlib.sha256(f"{seed}:{contrast}".encode("utf-8")).digest()
        contrast_seed = int.from_bytes(digest[:4], "little")
        shuffled = group.sample(frac=1.0, random_state=contrast_seed).reset_index(drop=True)
        offset = 0
        for split, size in split_sizes.items():
            selected = shuffled.iloc[offset : offset + size].copy()
            selected[dataset.split_column] = split
            assignments.append(selected)
            offset += size
    assignment_table = pd.concat(assignments, ignore_index=True, sort=False)
    source = rows.drop(columns=[dataset.split_column])
    return source.merge(
        assignment_table,
        on=[dataset.pair_id_column, dataset.contrast_column],
        how="inner",
        validate="many_to_one",
    )


def load_base_items(
    dataset: RankedChoiceDatasetConfig,
    *,
    seed: int,
    resplit_pairs: bool = False,
) -> pd.DataFrame:
    if not dataset.items_path.exists():
        raise FileNotFoundError(f"Missing ranked endpoint table: {dataset.items_path}")
    rows = pd.read_csv(dataset.items_path)
    required = {
        dataset.pair_id_column,
        dataset.contrast_column,
        dataset.split_column,
        dataset.label_column,
        *(field.column for field in dataset.prompt_fields),
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Ranked endpoint table is missing columns: {missing}")
    for column, values in dataset.extraction_filters.items():
        if column not in rows.columns:
            raise ValueError(f"Extraction filter column {column!r} is missing")
        rows = rows[rows[column].astype(str).isin(values)].copy()
    rows[dataset.label_column] = rows[dataset.label_column].astype(str)
    rows["semantic_rank"] = rows[dataset.label_column].map(dataset.label_ranks)
    if rows["semantic_rank"].isna().any():
        unknown = sorted(rows.loc[rows["semantic_rank"].isna(), dataset.label_column].unique())
        raise ValueError(f"Unmapped semantic labels: {unknown}")
    rows = rows.drop_duplicates(
        [dataset.pair_id_column, dataset.label_column, dataset.split_column]
    ).reset_index(drop=True)
    rows = (
        _resplit_pairs(rows, dataset, seed=seed)
        if resplit_pairs
        else _sample_pairs(rows, dataset, seed=seed)
    )
    invalid = []
    for pair_id, group in rows.groupby(dataset.pair_id_column, sort=True):
        if len(group) != 2 or group["semantic_rank"].nunique() != 2:
            invalid.append(str(pair_id))
    if invalid:
        raise ValueError(f"Expected two ranked endpoints per pair; invalid examples: {invalid[:5]}")
    return rows.reset_index(drop=True)


def build_mapping_items(
    base_items: pd.DataFrame,
    dataset: RankedChoiceDatasetConfig,
    mapping: MappingDefinition,
    canonical_mapping: MappingDefinition,
) -> pd.DataFrame:
    semantic_choices = mapping_choice_map(mapping)
    canonical_choices = mapping_choice_map(canonical_mapping)
    rows = base_items.copy()
    rows["mapping_name"] = mapping.name
    rows["mapping_family"] = mapping.family
    rows["semantic_choice"] = rows[dataset.label_column].map(semantic_choices).astype(int)
    rows["canonical_letter_choice"] = rows[dataset.label_column].map(canonical_choices).astype(int)
    rows["label_moved_by_mapping"] = rows["semantic_choice"].ne(rows["canonical_letter_choice"])
    scenario_contexts = [build_scenario_context(row, dataset) for _, row in rows.iterrows()]
    rows["scenario_char_end"] = [len(value) for value in scenario_contexts]
    rows["prompt"] = [build_ranked_choice_prompt(row, dataset, mapping) for _, row in rows.iterrows()]
    return rows


def build_canonical_train_index(
    canonical_items: pd.DataFrame,
    dataset: RankedChoiceDatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = canonical_items[
        canonical_items[dataset.split_column].astype(str).eq(dataset.train_split)
    ].copy().reset_index(drop=True)
    items["item_index"] = np.arange(len(items), dtype=int)
    pair_rows = []
    for (contrast, pair_id), group in items.groupby(
        [dataset.contrast_column, dataset.pair_id_column], sort=True
    ):
        ordered = group.sort_values("semantic_rank")
        pair_rows.append(
            {
                "pair_id": str(pair_id),
                "pair_type": str(contrast),
                "split": dataset.train_split,
                "negative_item_index": int(ordered["item_index"].iloc[0]),
                "positive_item_index": int(ordered["item_index"].iloc[-1]),
            }
        )
    return items, pd.DataFrame(pair_rows)


def build_mapping_eval_items(
    mapping_items: pd.DataFrame,
    dataset: RankedChoiceDatasetConfig,
    mapping: MappingDefinition,
    canonical_mapping: MappingDefinition,
    *,
    split: str | None = None,
) -> pd.DataFrame:
    selected_split = str(split or dataset.test_split)
    items = mapping_items[
        mapping_items[dataset.split_column].astype(str).eq(selected_split)
    ].copy()
    mapping_choices = mapping_choice_map(mapping)
    canonical_choices = mapping_choice_map(canonical_mapping)
    rows = []
    for (contrast, pair_id), group in items.groupby(
        [dataset.contrast_column, dataset.pair_id_column], sort=True
    ):
        ordered = group.sort_values("semantic_rank")
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        for eval_direction, endpoint, target, sign in [
            ("lower_to_higher", low, high, 1.0),
            ("higher_to_lower", high, low, -1.0),
        ]:
            target_label = str(target[dataset.label_column])
            semantic_choice = int(mapping_choices[target_label])
            original_choice = int(canonical_choices[target_label])
            rows.append(
                {
                    "dataset": dataset.dataset_name,
                    "mapping_name": mapping.name,
                    "mapping_family": mapping.family,
                    "pair_id": str(pair_id),
                    "pair_type": str(contrast),
                    "eval_direction": eval_direction,
                    "direction_sign": sign,
                    "prompt": str(endpoint["prompt"]),
                    "scenario_char_end": int(endpoint["scenario_char_end"]),
                    "base_semantic_label": str(endpoint[dataset.label_column]),
                    "target_semantic_label": target_label,
                    "semantic_target_choice": semantic_choice,
                    "original_answer_letter_choice": original_choice,
                    "target_letter_moved": semantic_choice != original_choice,
                    "original_letter_semantic_label": mapping.option_order[original_choice],
                }
            )
    return pd.DataFrame(rows)


def build_canonical_direction_bank(
    model: Any,
    tokenizer: Any,
    canonical_items: pd.DataFrame,
    dataset: RankedChoiceDatasetConfig,
    *,
    layer_index: int,
    subspace_dim: int,
    batch_size: int,
    max_length: int,
    seed: int,
    extraction_position: str = "pre_answer",
    enabled_modes: tuple[str, ...] | None = None,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    resolved_layer = resolve_layer_index(layer_index, len(decoder_layers(model)))
    train_items, train_pairs = build_canonical_train_index(canonical_items, dataset)
    if extraction_position == "pre_answer":
        _, activations = collect_choice_probs_and_activations(
            model,
            tokenizer,
            train_items["prompt"].tolist(),
            layer_indices=[resolved_layer],
            batch_size=batch_size,
            max_length=max_length,
            choice_letters=[chr(ord("A") + index) for index in range(len(dataset.label_ranks))],
        )
        selected_activations = activations[resolved_layer]
    else:
        _, activations_by_position = collect_choice_probs_and_position_activations(
            model,
            tokenizer,
            train_items["prompt"].tolist(),
            position_character_offsets={
                "scenario_end": train_items["scenario_char_end"].astype(int).tolist()
            },
            layer_indices=[resolved_layer],
            batch_size=batch_size,
            max_length=max_length,
            choice_letters=[chr(ord("A") + index) for index in range(len(dataset.label_ranks))],
        )
        selected_activations = activations_by_position["scenario_end"][resolved_layer]
    raw_directions, raw_inventory = contrast_directions(
        train_pairs,
        selected_activations,
        train_split=dataset.train_split,
    )
    bank = decompose_directions(
        raw_directions,
        subspace_dims=[subspace_dim],
        mixing_grid=[],
        seed=seed,
        include_loco=True,
        loco_subspace_dim=1,
    )
    mode_map = {
        "raw_caa_direction": "raw_canonical_direction",
        "shared_only_direction": "shared_canonical_direction",
        "residual_only_direction": "residual_canonical_direction",
        "loco_shared_direction": "loco_shared_canonical_direction",
        "loco_residual_direction": "loco_residual_canonical_direction",
        "random_direction_control": "random_direction_control",
        "wrong_direction_control": "wrong_direction_control",
    }
    directions: dict[tuple[str, str], np.ndarray] = {}
    inventory_rows = []
    enabled = set(enabled_modes or [*PAPER_MODES, *CONTROL_MODES])
    for (pair_type, source_mode, _, _), vector in bank.vectors.items():
        if source_mode not in mode_map:
            continue
        mode = mode_map[source_mode]
        if mode not in enabled:
            continue
        directions[(pair_type, mode)] = np.asarray(vector, dtype=np.float32)
        source_row = bank.inventory[
            bank.inventory["pair_type"].astype(str).eq(pair_type)
            & bank.inventory["mode"].astype(str).eq(source_mode)
        ].iloc[0]
        inventory_rows.append(
            {
                "pair_type": pair_type,
                "mode": mode,
                "extraction_mapping": canonical_items["mapping_name"].iloc[0],
                "extraction_position": extraction_position,
                "layer_index": resolved_layer,
                "subspace_dim": subspace_dim,
                "direction_l2": float(np.linalg.norm(vector)),
                "shared_explained_variance": float(source_row["shared_explained_variance"]),
                "basis_protocol": source_row.get("basis_protocol", "in_sample"),
                "basis_source_contrasts": source_row.get("basis_source_contrasts", "__all__"),
                "n_basis_contrasts": source_row.get("n_basis_contrasts", np.nan),
                "self_included_in_basis": source_row.get("self_included_in_basis", True),
                "unnormalized_projection_l2": source_row.get("unnormalized_projection_l2", np.nan),
                "unnormalized_residual_l2": source_row.get("unnormalized_residual_l2", np.nan),
                "projection_ratio": source_row.get("projection_ratio", np.nan),
                "n_train_pairs": int(train_pairs[train_pairs["pair_type"].eq(pair_type)]["pair_id"].nunique()),
            }
        )
    hidden_dim = next(iter(raw_directions.values())).shape[0]
    for pair_type in raw_directions:
        if "zero_direction_control" not in enabled:
            continue
        vector = np.zeros(hidden_dim, dtype=np.float32)
        directions[(pair_type, "zero_direction_control")] = vector
        inventory_rows.append(
            {
                "pair_type": pair_type,
                "mode": "zero_direction_control",
                "extraction_mapping": canonical_items["mapping_name"].iloc[0],
                "extraction_position": extraction_position,
                "layer_index": resolved_layer,
                "subspace_dim": 0,
                "direction_l2": 0.0,
                "shared_explained_variance": np.nan,
                "n_train_pairs": int(train_pairs[train_pairs["pair_type"].eq(pair_type)]["pair_id"].nunique()),
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    return directions, inventory


def evaluate_fixed_directions(
    model: Any,
    tokenizer: Any,
    eval_items: pd.DataFrame,
    directions: dict[tuple[str, str], np.ndarray],
    direction_inventory: pd.DataFrame,
    *,
    layer_index: int,
    alpha: float,
    n_choices: int,
    batch_size: int,
    max_length: int,
    injection_position: str = "pre_answer",
) -> pd.DataFrame:
    choice_letters = [chr(ord("A") + index) for index in range(n_choices)]
    prompts = eval_items["prompt"].tolist()
    baseline = collect_choice_probs(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_length=max_length,
        choice_letters=choice_letters,
    )
    base_prediction = baseline.argmax(axis=1)
    extraction_mappings = direction_inventory.set_index(["pair_type", "mode"])[
        "extraction_mapping"
    ].to_dict()
    rows = []
    for (pair_type, mode), direction in sorted(directions.items()):
        pair_mask = eval_items["pair_type"].astype(str).eq(pair_type).to_numpy()
        for sign in [1.0, -1.0]:
            mask = pair_mask & eval_items["direction_sign"].eq(sign).to_numpy()
            if not mask.any():
                continue
            subset = eval_items.loc[mask].reset_index(drop=True)
            subset_base = baseline[mask]
            subset_base_pred = base_prediction[mask]
            if mode == "zero_direction_control":
                patched = subset_base.copy()
            else:
                offsets = (
                    subset["scenario_char_end"].astype(int).tolist()
                    if injection_position == "scenario_end"
                    else None
                )
                patched = patched_choice_probs(
                    model,
                    tokenizer,
                    subset["prompt"].tolist(),
                    layer_index=layer_index,
                    direction=np.asarray(direction, dtype=np.float32) * sign,
                    alpha=alpha,
                    batch_size=batch_size,
                    max_length=max_length,
                    choice_letters=choice_letters,
                    injection_character_offsets=offsets,
                )
            patched_prediction = patched.argmax(axis=1)
            semantic_choices = subset["semantic_target_choice"].to_numpy(dtype=int)
            original_choices = subset["original_answer_letter_choice"].to_numpy(dtype=int)
            for index, item in subset.iterrows():
                semantic_choice = int(semantic_choices[index])
                original_choice = int(original_choices[index])
                rows.append(
                    {
                        "dataset": item["dataset"],
                        "mapping_name": item["mapping_name"],
                        "mapping_family": item["mapping_family"],
                        "pair_id": item["pair_id"],
                        "pair_type": pair_type,
                        "eval_direction": item["eval_direction"],
                        "base_semantic_label": item["base_semantic_label"],
                        "target_semantic_label": item["target_semantic_label"],
                        "semantic_target_choice": semantic_choice,
                        "original_answer_letter_choice": original_choice,
                        "target_letter_moved": bool(item["target_letter_moved"]),
                        "original_letter_semantic_label": item["original_letter_semantic_label"],
                        "mode": mode,
                        "extraction_mapping": extraction_mappings[(pair_type, mode)],
                        "injection_position": injection_position,
                        "layer_index": int(layer_index),
                        "alpha": float(alpha),
                        "base_semantic_target_prob": float(subset_base[index, semantic_choice]),
                        "patched_semantic_target_prob": float(patched[index, semantic_choice]),
                        "delta_semantic_target_prob": float(
                            patched[index, semantic_choice] - subset_base[index, semantic_choice]
                        ),
                        "base_original_letter_prob": float(subset_base[index, original_choice]),
                        "patched_original_letter_prob": float(patched[index, original_choice]),
                        "delta_original_letter_prob": float(
                            patched[index, original_choice] - subset_base[index, original_choice]
                        ),
                        "base_semantic_target_correct": bool(subset_base_pred[index] == semantic_choice),
                        "patched_semantic_target_correct": bool(patched_prediction[index] == semantic_choice),
                        "base_original_letter_selected": bool(subset_base_pred[index] == original_choice),
                        "patched_original_letter_selected": bool(patched_prediction[index] == original_choice),
                        "prediction_changed": bool(patched_prediction[index] != subset_base_pred[index]),
                        "js_shift": float(js_divergence(subset_base[index], patched[index])),
                    }
                )
    return pd.DataFrame(rows)


def build_pair_effects(eval_rows: pd.DataFrame) -> pd.DataFrame:
    rows = eval_rows.copy()
    rows["semantic_accuracy_gain"] = (
        rows["patched_semantic_target_correct"].astype(float)
        - rows["base_semantic_target_correct"].astype(float)
    )
    rows["original_letter_accuracy_gain"] = (
        rows["patched_original_letter_selected"].astype(float)
        - rows["base_original_letter_selected"].astype(float)
    )
    rows["moved_semantic_prob_gain"] = rows["delta_semantic_target_prob"].where(
        rows["target_letter_moved"]
    )
    rows["moved_original_letter_prob_gain"] = rows["delta_original_letter_prob"].where(
        rows["target_letter_moved"]
    )
    rows["moved_semantic_accuracy_gain"] = rows["semantic_accuracy_gain"].where(
        rows["target_letter_moved"]
    )
    rows["moved_original_letter_accuracy_gain"] = rows["original_letter_accuracy_gain"].where(
        rows["target_letter_moved"]
    )
    grouped = (
        rows.groupby(["mapping_name", "mapping_family", "pair_id", "pair_type", "mode"], as_index=False)
        .agg(
            n_directions=("eval_direction", "size"),
            n_moved_targets=("target_letter_moved", "sum"),
            semantic_accuracy_gain=("semantic_accuracy_gain", "mean"),
            original_letter_accuracy_gain=("original_letter_accuracy_gain", "mean"),
            semantic_prob_gain=("delta_semantic_target_prob", "mean"),
            original_letter_prob_gain=("delta_original_letter_prob", "mean"),
            moved_semantic_accuracy_gain=("moved_semantic_accuracy_gain", "mean"),
            moved_original_letter_accuracy_gain=("moved_original_letter_accuracy_gain", "mean"),
            moved_semantic_prob_gain=("moved_semantic_prob_gain", "mean"),
            moved_original_letter_prob_gain=("moved_original_letter_prob_gain", "mean"),
            js_shift=("js_shift", "mean"),
            prediction_changed_rate=("prediction_changed", "mean"),
        )
    )
    grouped["semantic_minus_original_prob_gain_moved"] = (
        grouped["moved_semantic_prob_gain"] - grouped["moved_original_letter_prob_gain"]
    )
    grouped["semantic_minus_original_accuracy_gain_moved"] = (
        grouped["moved_semantic_accuracy_gain"] - grouped["moved_original_letter_accuracy_gain"]
    )
    return grouped


AUDIT_METRICS = [
    "semantic_accuracy_gain",
    "original_letter_accuracy_gain",
    "semantic_prob_gain",
    "original_letter_prob_gain",
    "moved_semantic_accuracy_gain",
    "moved_original_letter_accuracy_gain",
    "moved_semantic_prob_gain",
    "moved_original_letter_prob_gain",
    "semantic_minus_original_prob_gain_moved",
    "semantic_minus_original_accuracy_gain_moved",
    "js_shift",
]


def _equal_group_bootstrap(
    group: pd.DataFrame,
    metric: str,
    *,
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    parts = []
    for _, part in group.groupby("pair_type", sort=True):
        values = part[metric].dropna().astype(float).to_numpy()
        if len(values):
            parts.append(values)
    if not parts:
        return np.nan, np.nan, np.nan, 0
    observed = float(np.mean([values.mean() for values in parts]))
    boot = np.zeros(n_boot, dtype=float)
    for values in parts:
        indices = rng.integers(0, len(values), size=(n_boot, len(values)))
        boot += values[indices].mean(axis=1) / len(parts)
    tail = (1.0 - confidence) / 2.0
    return observed, float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail)), int(sum(len(v) for v in parts))


def build_mapping_mode_ci(
    pair_effects: pd.DataFrame,
    *,
    n_boot: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for (mapping_name, mapping_family, mode), group in pair_effects.groupby(
        ["mapping_name", "mapping_family", "mode"], sort=True
    ):
        record = {
            "mapping_name": mapping_name,
            "mapping_family": mapping_family,
            "mode": mode,
            "n_pairs": int(group["pair_id"].nunique()),
            "n_pair_types": int(group["pair_type"].nunique()),
            "n_pairs_with_moved_targets": int(group.loc[group["n_moved_targets"].gt(0), "pair_id"].nunique()),
        }
        for metric in AUDIT_METRICS:
            mean, low, high, n = _equal_group_bootstrap(
                group,
                metric,
                n_boot=n_boot,
                confidence=confidence,
                rng=rng,
            )
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
            record[f"{metric}_n"] = n
        records.append(record)
    return pd.DataFrame(records)


def build_mapping_signature_statistics(
    pair_effects: pd.DataFrame,
    *,
    n_boot: int,
    n_permutations: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Directly test S_map = semantic movement minus original-slot movement."""
    rows = pair_effects[
        pair_effects["mode"].eq("raw_canonical_direction")
        & pair_effects["n_moved_targets"].gt(0)
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["s_map"] = rows["semantic_minus_original_prob_gain_moved"].astype(float)
    rows = rows[np.isfinite(rows["s_map"])].copy()
    rng = np.random.default_rng(seed)
    records = []
    for (mapping_name, mapping_family), group in rows.groupby(
        ["mapping_name", "mapping_family"], sort=True
    ):
        mean, low, high, n_pairs = _equal_group_bootstrap(
            group,
            "s_map",
            n_boot=n_boot,
            confidence=confidence,
            rng=rng,
        )
        weights = group.groupby("pair_type", sort=True)["pair_id"].transform(
            lambda values: 1.0 / len(values)
        ).to_numpy(dtype=float)
        records.append(
            {
                "mapping_name": mapping_name,
                "mapping_family": mapping_family,
                "n_pairs": int(group["pair_id"].nunique()),
                "n_pair_types": int(group["pair_type"].nunique()),
                "s_map_mean": mean,
                "s_map_ci_low": low,
                "s_map_ci_high": high,
                "s_map_ci_excludes_zero": bool(low > 0 or high < 0),
                "s_map_p_two_sided": paired_weighted_sign_flip_pvalue(
                    group["s_map"].to_numpy(dtype=float),
                    weights,
                    n_permutations=n_permutations,
                    rng=rng,
                ),
                "inference_unit": "pair_id_cluster; equal_pair_type_weight",
                "n_pair_observations": n_pairs,
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["s_map_p_holm"] = holm_adjust(out["s_map_p_two_sided"])
    out["s_map_significant_holm"] = out["s_map_p_holm"].lt(0.05)
    return out


def build_global_mapping_signature_statistics(
    pair_effects: pd.DataFrame,
    *,
    n_boot: int,
    n_permutations: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Compute pair-clustered, equal-model S_map inference across audit runs.

    Repeated mappings of a pair are averaged before inference. Each
    model-by-contrast stratum receives equal weight, so large NormBank
    contrasts and models with more mappings cannot dominate the aggregate.
    """
    required = {
        "mode",
        "n_moved_targets",
        "semantic_minus_original_prob_gain_moved",
        "model_alias",
        "mapping_family",
        "pair_id",
        "pair_type",
    }
    if pair_effects.empty or not required.issubset(pair_effects.columns):
        return pd.DataFrame()
    rows = pair_effects[
        pair_effects["mode"].eq("raw_canonical_direction")
        & pair_effects["n_moved_targets"].gt(0)
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["s_map"] = rows["semantic_minus_original_prob_gain_moved"].astype(float)
    rows = rows[np.isfinite(rows["s_map"])].copy()
    if rows.empty:
        return pd.DataFrame()

    scopes: list[tuple[str, pd.DataFrame]] = [("all_remapped", rows)]
    for family, group in rows.groupby("mapping_family", sort=True):
        scopes.append((str(family), group.copy()))

    rng = np.random.default_rng(seed)
    records = []
    for scope, source in scopes:
        collapsed = (
            source.groupby(["model_alias", "pair_type", "pair_id"], as_index=False)
            .agg(s_map=("s_map", "mean"))
        )
        strata = [
            group["s_map"].to_numpy(dtype=float)
            for _, group in collapsed.groupby(["model_alias", "pair_type"], sort=True)
        ]
        strata = [values for values in strata if len(values)]
        if not strata:
            continue
        estimate = float(np.mean([values.mean() for values in strata]))
        boot = np.zeros(int(n_boot), dtype=float)
        for values in strata:
            indices = rng.integers(0, len(values), size=(int(n_boot), len(values)))
            boot += values[indices].mean(axis=1) / len(strata)
        tail = (1.0 - float(confidence)) / 2.0
        weighted_values = np.concatenate(strata)
        weights = np.concatenate(
            [
                np.full(len(values), 1.0 / (len(strata) * len(values)), dtype=float)
                for values in strata
            ]
        )
        records.append(
            {
                "scope": scope,
                "n_models": int(collapsed["model_alias"].nunique()),
                "n_model_contrast_strata": len(strata),
                "n_pair_clusters": int(len(collapsed)),
                "s_map_mean": estimate,
                "s_map_ci_low": float(np.quantile(boot, tail)),
                "s_map_ci_high": float(np.quantile(boot, 1.0 - tail)),
                "s_map_ci_excludes_zero": bool(
                    np.quantile(boot, tail) > 0 or np.quantile(boot, 1.0 - tail) < 0
                ),
                "s_map_p_two_sided": paired_weighted_sign_flip_pvalue(
                    weighted_values,
                    weights,
                    n_permutations=n_permutations,
                    rng=rng,
                ),
                "inference_unit": "pair_id_cluster; equal_model_contrast_weight",
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["s_map_p_holm"] = holm_adjust(out["s_map_p_two_sided"])
    out["s_map_significant_holm"] = out["s_map_p_holm"].lt(0.05)
    return out


def build_opaque_mapping_competence_items(
    mappings: tuple[MappingDefinition, ...],
) -> pd.DataFrame:
    """Create context-free checks that test codeword mapping, not norm judgment."""
    rows = []
    for mapping in mappings:
        if mapping.family != "opaque_codewords":
            continue
        for target_label in mapping.option_order:
            lines = list(mapping.header_lines)
            lines.append(
                f"Question: Which option means {target_label.replace('_', ' ')}?"
            )
            for index, label in enumerate(mapping.option_order):
                lines.append(f"{chr(ord('A') + index)}. {mapping.option_texts[label]}")
            lines.extend(["Answer with only the option letter.", "Answer:"])
            rows.append(
                {
                    "mapping_name": mapping.name,
                    "mapping_family": mapping.family,
                    "target_label": target_label,
                    "codeword": mapping.option_texts[target_label],
                    "target_choice": int(mapping.option_order.index(target_label)),
                    "prompt": "\n".join(lines),
                }
            )
    return pd.DataFrame(rows)


def evaluate_opaque_mapping_competence(
    model: Any,
    tokenizer: Any,
    items: pd.DataFrame,
    *,
    n_choices: int,
    batch_size: int,
    max_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if items.empty:
        return pd.DataFrame(), pd.DataFrame()
    choice_letters = [chr(ord("A") + index) for index in range(n_choices)]
    probs = collect_choice_probs(
        model,
        tokenizer,
        items["prompt"].tolist(),
        batch_size=batch_size,
        max_length=max_length,
        choice_letters=choice_letters,
    )
    rows = items.copy()
    targets = rows["target_choice"].to_numpy(dtype=int)
    rows["target_prob"] = probs[np.arange(len(rows)), targets]
    rows["prediction"] = probs.argmax(axis=1)
    rows["correct"] = rows["prediction"].eq(targets)
    audit_rows = []
    for mapping_name, group in items.groupby("mapping_name", sort=True):
        for _, item in group.iterrows():
            text = str(item["codeword"])
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            audit_rows.append(
                {
                    "mapping_name": mapping_name,
                    "target_label": item["target_label"],
                    "codeword": text,
                    "n_tokens": len(token_ids),
                    "token_ids": ",".join(str(value) for value in token_ids),
                    "tokens": " ".join(str(value) for value in tokens),
                }
            )
    return rows, pd.DataFrame(audit_rows)


MODE_COMPARISONS = [
    ("raw_canonical_direction", "random_direction_control"),
    ("raw_canonical_direction", "wrong_direction_control"),
    ("raw_canonical_direction", "zero_direction_control"),
    ("shared_canonical_direction", "random_direction_control"),
    ("residual_canonical_direction", "random_direction_control"),
    ("loco_shared_canonical_direction", "loco_residual_canonical_direction"),
    ("loco_shared_canonical_direction", "random_direction_control"),
    ("shared_canonical_direction", "loco_shared_canonical_direction"),
]


def build_mode_comparisons(
    pair_effects: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for mapping_name, mapping_rows in pair_effects.groupby("mapping_name", sort=True):
        for left, right in MODE_COMPARISONS:
            wide = mapping_rows.pivot_table(
                index=["pair_id", "pair_type"],
                columns="mode",
                values=["semantic_accuracy_gain", "semantic_prob_gain"],
                aggfunc="mean",
            )
            if left not in wide["semantic_accuracy_gain"] or right not in wide["semantic_accuracy_gain"]:
                continue
            accuracy_difference = (
                wide[("semantic_accuracy_gain", left)] - wide[("semantic_accuracy_gain", right)]
            ).dropna()
            probability_difference = (
                wide[("semantic_prob_gain", left)] - wide[("semantic_prob_gain", right)]
            ).dropna()
            records.append(
                {
                    "mapping_name": mapping_name,
                    "comparison": f"{left}_minus_{right}",
                    "left_mode": left,
                    "right_mode": right,
                    "n_pairs": int(len(accuracy_difference)),
                    "mean_semantic_accuracy_gain_difference": float(accuracy_difference.mean()),
                    "mean_semantic_prob_gain_difference": float(probability_difference.mean()),
                    "semantic_accuracy_p_two_sided": paired_sign_flip_pvalue(
                        accuracy_difference.to_numpy(),
                        n_permutations=n_permutations,
                        rng=rng,
                    ),
                    "semantic_prob_p_two_sided": paired_sign_flip_pvalue(
                        probability_difference.to_numpy(),
                        n_permutations=n_permutations,
                        rng=rng,
                    ),
                }
            )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["semantic_accuracy_p_holm"] = np.nan
    out["semantic_prob_p_holm"] = np.nan
    for _, indices in out.groupby("mapping_name", sort=True).groups.items():
        idx = list(indices)
        out.loc[idx, "semantic_accuracy_p_holm"] = holm_adjust(out.loc[idx, "semantic_accuracy_p_two_sided"])
        out.loc[idx, "semantic_prob_p_holm"] = holm_adjust(out.loc[idx, "semantic_prob_p_two_sided"])
    out["semantic_accuracy_significant_holm"] = out["semantic_accuracy_p_holm"].lt(0.05)
    out["semantic_prob_significant_holm"] = out["semantic_prob_p_holm"].lt(0.05)
    return out


def build_contrast_summary(eval_rows: pd.DataFrame) -> pd.DataFrame:
    rows = eval_rows.copy()
    rows["semantic_accuracy_gain"] = (
        rows["patched_semantic_target_correct"].astype(float)
        - rows["base_semantic_target_correct"].astype(float)
    )
    rows["original_letter_accuracy_gain"] = (
        rows["patched_original_letter_selected"].astype(float)
        - rows["base_original_letter_selected"].astype(float)
    )
    return (
        rows.groupby(["mapping_name", "mapping_family", "pair_type", "eval_direction", "mode"], as_index=False)
        .agg(
            n_items=("pair_id", "nunique"),
            semantic_accuracy_gain=("semantic_accuracy_gain", "mean"),
            original_letter_accuracy_gain=("original_letter_accuracy_gain", "mean"),
            semantic_prob_gain=("delta_semantic_target_prob", "mean"),
            original_letter_prob_gain=("delta_original_letter_prob", "mean"),
            mean_js_shift=("js_shift", "mean"),
        )
    )


def mapping_audit_run_is_complete(model_dir: str | Path) -> bool:
    root = Path(model_dir)
    return all(
        (root / filename).exists()
        for filename in [
            "mapping_mode_ci.csv",
            "mode_comparisons.csv",
            "canonical_direction_inventory.csv",
            "mapping_audit_run_complete.csv",
        ]
    )


def annotate_model_frame(frame: pd.DataFrame, *, model_alias: str, model_name: str) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["model_alias"] = model_alias
    annotated["model_name"] = model_name
    leading = ["model_alias", "model_name"]
    return annotated[leading + [column for column in annotated.columns if column not in leading]]


def run_single_model_mapping_audit(
    model: Any,
    tokenizer: Any,
    config: FixedDirectionMappingAuditConfig,
    model_config: MappingAuditModelConfig,
) -> dict[str, pd.DataFrame]:
    alias = model_config.model.alias
    model_dir = config.output_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    base_items = load_base_items(config.dataset, seed=config.runtime.seed)
    mapping_by_name = {mapping.name: mapping for mapping in config.mappings}
    canonical_mapping = mapping_by_name[config.dataset.canonical_mapping]
    canonical_items = build_mapping_items(
        base_items,
        config.dataset,
        canonical_mapping,
        canonical_mapping,
    )
    directions, direction_inventory = build_canonical_direction_bank(
        model,
        tokenizer,
        canonical_items,
        config.dataset,
        layer_index=model_config.locked_layer,
        subspace_dim=config.subspace_dim,
        batch_size=config.runtime.batch_size,
        max_length=config.runtime.max_length,
        seed=config.runtime.seed,
        extraction_position=config.extraction_position,
        enabled_modes=config.enabled_modes,
    )
    vector_payload = {
        f"{pair_type}__{mode}": vector for (pair_type, mode), vector in directions.items()
    }
    np.savez_compressed(model_dir / "canonical_direction_bank.npz", **vector_payload)

    all_mapping_items = []
    all_eval_rows = []
    for mapping in config.mappings:
        mapping_items = build_mapping_items(
            base_items,
            config.dataset,
            mapping,
            canonical_mapping,
        )
        eval_items = build_mapping_eval_items(
            mapping_items,
            config.dataset,
            mapping,
            canonical_mapping,
        )
        eval_rows = evaluate_fixed_directions(
            model,
            tokenizer,
            eval_items,
            directions,
            direction_inventory,
            layer_index=model_config.locked_layer,
            alpha=model_config.locked_alpha,
            n_choices=len(config.dataset.label_ranks),
            batch_size=config.runtime.batch_size,
            max_length=config.runtime.max_length,
            injection_position=config.injection_position,
        )
        all_mapping_items.append(mapping_items)
        all_eval_rows.append(eval_rows)
    mapping_items_all = pd.concat(all_mapping_items, ignore_index=True, sort=False)
    eval_rows_all = pd.concat(all_eval_rows, ignore_index=True, sort=False)
    pair_effects = build_pair_effects(eval_rows_all)
    pair_effects["extraction_position"] = config.extraction_position
    pair_effects["injection_position"] = config.injection_position
    mapping_mode_ci = build_mapping_mode_ci(
        pair_effects,
        n_boot=config.statistics.n_boot,
        confidence=config.statistics.confidence,
        seed=config.runtime.seed,
    )
    mode_comparisons = build_mode_comparisons(
        pair_effects,
        n_permutations=config.statistics.n_permutations,
        seed=config.runtime.seed + 1,
    )
    competence_items = build_opaque_mapping_competence_items(config.mappings)
    competence_rows, token_audit = evaluate_opaque_mapping_competence(
        model,
        tokenizer,
        competence_items,
        n_choices=len(config.dataset.label_ranks),
        batch_size=config.runtime.batch_size,
        max_length=config.runtime.max_length,
    )
    tables = {
        "base_items": base_items,
        "mapping_items": mapping_items_all,
        "canonical_direction_inventory": direction_inventory,
        "eval_rows": eval_rows_all,
        "pair_effects": pair_effects,
        "mapping_mode_ci": mapping_mode_ci,
        "mode_comparisons": mode_comparisons,
        "mapping_signature_statistics": build_mapping_signature_statistics(
            pair_effects,
            n_boot=config.statistics.n_boot,
            n_permutations=config.statistics.n_permutations,
            confidence=config.statistics.confidence,
            seed=config.runtime.seed + 2,
        ),
        "opaque_mapping_competence": competence_rows,
        "opaque_codeword_token_audit": token_audit,
        "contrast_summary": build_contrast_summary(eval_rows_all),
        "mapping_inventory": pd.DataFrame(
            [
                {
                    "mapping_name": mapping.name,
                    "mapping_family": mapping.family,
                    "option_order": ",".join(mapping.option_order),
                    "is_canonical": mapping.name == config.dataset.canonical_mapping,
                }
                for mapping in config.mappings
            ]
        ),
    }
    for name, frame in tables.items():
        # Some reusable endpoint tables retain provenance from the model that
        # originally materialized the prompts. Replace that provenance with
        # the model executing this audit instead of inserting duplicate names.
        tables[name] = annotate_model_frame(
            frame,
            model_alias=alias,
            model_name=model_config.model.name,
        )
    write_tables(tables, model_dir)
    completion = pd.DataFrame(
        [
            {
                "model_alias": alias,
                "model_name": model_config.model.name,
                "status": "complete",
                "locked_layer": model_config.locked_layer,
                "locked_alpha": model_config.locked_alpha,
                "subspace_dim": config.subspace_dim,
                "extraction_mapping": config.dataset.canonical_mapping,
                "extraction_position": config.extraction_position,
                "injection_position": config.injection_position,
                "enabled_modes": ",".join(config.enabled_modes),
                "n_eval_mappings": len(config.mappings),
            }
        ]
    )
    completion.to_csv(model_dir / "mapping_audit_run_complete.csv", index=False)
    return tables


def aggregate_mapping_audit(config: FixedDirectionMappingAuditConfig) -> dict[str, pd.DataFrame]:
    outputs: dict[str, list[pd.DataFrame]] = {}
    for model_config in config.models:
        model_dir = config.output_dir / model_config.model.alias
        for name in [
            "mapping_audit_run_complete",
            "mapping_mode_ci",
            "mapping_signature_statistics",
            "mode_comparisons",
            "pair_effects",
            "opaque_mapping_competence",
            "opaque_codeword_token_audit",
            "contrast_summary",
            "canonical_direction_inventory",
        ]:
            path = model_dir / f"{name}.csv"
            if path.exists():
                outputs.setdefault(name, []).append(pd.read_csv(path))
    aggregated = {
        name: pd.concat(parts, ignore_index=True, sort=False)
        for name, parts in outputs.items()
    }
    ci = aggregated.get("mapping_mode_ci", pd.DataFrame())
    focus = ci[
        ci.get("mode", pd.Series(dtype=str)).eq("raw_canonical_direction")
        & ci.get("n_pairs_with_moved_targets", pd.Series(dtype=int)).gt(0)
    ].copy()
    if focus.empty:
        global_decision = pd.DataFrame()
    else:
        focus["semantic_gain_positive"] = focus["moved_semantic_prob_gain_mean"].gt(0)
        focus["semantic_beats_original_letter"] = focus[
            "semantic_minus_original_prob_gain_moved_mean"
        ].gt(0)
        global_decision = (
            focus.groupby(["mapping_name", "mapping_family"], as_index=False)
            .agg(
                n_models=("model_alias", "nunique"),
                n_models_semantic_gain_positive=("semantic_gain_positive", "sum"),
                n_models_semantic_beats_original_letter=("semantic_beats_original_letter", "sum"),
                mean_semantic_prob_gain=("moved_semantic_prob_gain_mean", "mean"),
                mean_original_letter_prob_gain=("moved_original_letter_prob_gain_mean", "mean"),
                mean_semantic_minus_original=("semantic_minus_original_prob_gain_moved_mean", "mean"),
            )
        )
    aggregated["global_mapping_decision"] = global_decision
    pair_effects = aggregated.get("pair_effects", pd.DataFrame())
    aggregated["global_mapping_signature_statistics"] = build_global_mapping_signature_statistics(
        pair_effects,
        n_boot=config.statistics.n_boot,
        n_permutations=config.statistics.n_permutations,
        confidence=config.statistics.confidence,
        seed=config.runtime.seed + 3,
    )
    write_tables(aggregated, config.output_dir)
    return aggregated


def run_fixed_direction_mapping_audit(
    config: FixedDirectionMappingAuditConfig,
    *,
    model_aliases: Iterable[str] | None = None,
    model_loader: Callable[..., tuple[Any, Any]] = load_tokenizer_and_model,
    progress: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(
        model_aliases
        or [model_config.model.alias for model_config in config.models if model_config.model.enabled]
    )
    errors = []
    for model_config in config.models:
        public_model = model_config.model
        if not public_model.enabled or public_model.alias not in requested:
            continue
        model_dir = config.output_dir / public_model.alias
        if mapping_audit_run_is_complete(model_dir) and not config.runtime.force_rerun:
            progress(f"[{public_model.alias}] complete; skipping")
            continue
        tokenizer = None
        model = None
        try:
            progress(f"[{public_model.alias}] loading {public_model.load_source}")
            tokenizer, model = model_loader(
                public_model.load_source,
                device_map=public_model.device_map,
                torch_dtype=public_model.torch_dtype,
            )
            run_single_model_mapping_audit(model, tokenizer, config, model_config)
            (model_dir / "mapping_audit_run_error.csv").unlink(missing_ok=True)
            progress(f"[{public_model.alias}] complete")
        except Exception as exc:
            error = {
                "model_alias": public_model.alias,
                "model_name": public_model.name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            model_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([error]).to_csv(model_dir / "mapping_audit_run_error.csv", index=False)
            progress(f"[{public_model.alias}] failed: {type(exc).__name__}: {exc}")
            if not config.runtime.continue_on_error:
                raise
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            cleanup_model()
        aggregate_mapping_audit(config)
    if errors:
        pd.DataFrame(errors).to_csv(config.output_dir / "mapping_audit_errors.csv", index=False)
    else:
        (config.output_dir / "mapping_audit_errors.csv").unlink(missing_ok=True)
    return aggregate_mapping_audit(config)


def run_fixed_direction_mapping_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    config = FixedDirectionMappingAuditConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    return run_fixed_direction_mapping_audit(config, model_aliases=model_aliases)
