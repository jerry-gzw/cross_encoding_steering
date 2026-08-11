"""Matched three-way MNLI control for cross-interface steering effects."""
from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ModelConfig
from .io import write_tables
from .steering import (
    cleanup_model,
    collect_choice_probs_and_activations,
    completion_sequence_logprobs,
    decoder_layers,
    load_tokenizer_and_model,
    resolve_layer_index,
)


MNLI_LABELS = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True)
class MnliContrast:
    name: str
    source_label: str
    target_label: str


@dataclass(frozen=True)
class MnliInterface:
    name: str
    kind: str
    label_values: dict[str, str]
    option_order: tuple[str, ...] = ()
    key_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class MnliTemplate:
    name: str
    question: str
    answer_instruction: str


@dataclass(frozen=True)
class MnliModelRun:
    model: ModelConfig
    locked_layer: int
    locked_alpha: float


@dataclass(frozen=True)
class MnliControlConfig:
    project_root: Path
    input_path: Path
    output_dir: Path
    contrasts: tuple[MnliContrast, ...]
    interfaces: tuple[MnliInterface, ...]
    templates: tuple[MnliTemplate, ...]
    models: tuple[MnliModelRun, ...]
    reference_interface: str
    primary_score: str
    max_train_pairs_per_contrast: int
    max_test_pairs_per_contrast: int
    batch_size: int
    max_length: int
    seed: int
    random_seeds: tuple[int, ...]
    attribution_analysis: bool
    include_inverse_control: bool
    force_rerun: bool
    n_boot: int
    confidence: float
    bootstrap_chunk_size: int

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "MnliControlConfig":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        contrasts = tuple(MnliContrast(**item) for item in data.get("contrasts", []))
        if not contrasts:
            raise ValueError("MNLI control requires at least one label contrast")
        for contrast in contrasts:
            if contrast.source_label not in MNLI_LABELS or contrast.target_label not in MNLI_LABELS:
                raise ValueError(f"Unknown MNLI contrast labels in {contrast.name!r}")
            if contrast.source_label == contrast.target_label:
                raise ValueError(f"MNLI contrast {contrast.name!r} uses the same source and target label")

        interfaces = tuple(
            MnliInterface(
                name=str(item["name"]),
                kind=str(item["kind"]),
                label_values={str(key): str(value) for key, value in item["label_values"].items()},
                option_order=tuple(str(value) for value in item.get("option_order", [])),
                key_lines=tuple(str(value) for value in item.get("key_lines", [])),
            )
            for item in data.get("interfaces", [])
        )
        if not interfaces:
            raise ValueError("MNLI control requires at least one answer interface")
        for interface in interfaces:
            if interface.kind not in {"letter_mcq", "completion"}:
                raise ValueError(f"Unsupported MNLI interface kind: {interface.kind}")
            if set(interface.label_values) != set(MNLI_LABELS):
                raise ValueError(f"Interface {interface.name!r} must map all MNLI labels")
            if interface.kind == "letter_mcq" and set(interface.option_order) != set(MNLI_LABELS):
                raise ValueError(f"Letter interface {interface.name!r} requires a complete option_order")

        templates = tuple(MnliTemplate(**item) for item in data.get("templates", []))
        if not templates:
            raise ValueError("MNLI control requires at least one prompt template")
        overrides = model_source_overrides or {}
        models = []
        for item in data.get("models", []):
            model_data = dict(item)
            locked_layer = int(model_data.pop("locked_layer"))
            locked_alpha = float(model_data.pop("locked_alpha", 0.8))
            alias = str(model_data["alias"])
            if alias in overrides:
                model_data["source"] = overrides[alias]
            models.append(MnliModelRun(ModelConfig.from_mapping(model_data), locked_layer, locked_alpha))
        if not models:
            raise ValueError("MNLI control requires at least one model")

        reference = str(data.get("reference_interface", interfaces[0].name))
        if reference not in {item.name for item in interfaces}:
            raise ValueError("reference_interface must name a configured interface")
        primary_score = str(data.get("primary_score", "mean_logprob"))
        if primary_score not in {"sum_logprob", "mean_logprob"}:
            raise ValueError("primary_score must be sum_logprob or mean_logprob")
        runtime = dict(data.get("runtime", {}))
        statistics = dict(data.get("statistics", {}))
        random_seeds = tuple(int(value) for value in data.get("random_seeds", []))
        if len(set(random_seeds)) != len(random_seeds):
            raise ValueError("random_seeds must not contain duplicates")
        n_boot = int(statistics.get("n_boot", 10_000))
        confidence = float(statistics.get("confidence", 0.95))
        if n_boot < 100:
            raise ValueError("statistics.n_boot must be at least 100")
        if not 0.0 < confidence < 1.0:
            raise ValueError("statistics.confidence must be between zero and one")
        chunk_size = int(statistics.get("bootstrap_chunk_size", 500))
        if chunk_size < 1:
            raise ValueError("statistics.bootstrap_chunk_size must be positive")
        return cls(
            project_root=root,
            input_path=resolve(str(data.get("input_path", "datasets/multinli_1.0_train.jsonl"))),
            output_dir=resolve(str(data.get("output_dir", "outputs/mnli_non_norm_control"))),
            contrasts=contrasts,
            interfaces=interfaces,
            templates=templates,
            models=tuple(models),
            reference_interface=reference,
            primary_score=primary_score,
            max_train_pairs_per_contrast=int(data.get("max_train_pairs_per_contrast", 2048)),
            max_test_pairs_per_contrast=int(data.get("max_test_pairs_per_contrast", 512)),
            batch_size=int(runtime.get("batch_size", 2)),
            max_length=int(runtime.get("max_length", 512)),
            seed=int(runtime.get("seed", 13)),
            random_seeds=random_seeds,
            attribution_analysis=bool(data.get("attribution_analysis", False)),
            include_inverse_control=bool(data.get("include_inverse_control", True)),
            force_rerun=bool(runtime.get("force_rerun", False)),
            n_boot=n_boot,
            confidence=confidence,
            bootstrap_chunk_size=chunk_size,
        )


def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_mnli_items(config: MnliControlConfig) -> pd.DataFrame:
    if not config.input_path.exists():
        raise FileNotFoundError(f"MNLI input file does not exist: {config.input_path}")
    rows = pd.read_json(config.input_path, lines=True)
    required = {"pairID", "promptID", "genre", "gold_label", "sentence1", "sentence2"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"MNLI input is missing columns: {missing}")
    rows = rows[rows["gold_label"].astype(str).isin(MNLI_LABELS)].copy()
    rows["premise"] = rows["sentence1"].fillna("").astype(str).str.strip()
    rows["hypothesis"] = rows["sentence2"].fillna("").astype(str).str.strip()
    rows["label"] = rows["gold_label"].astype(str)
    rows["group_id"] = rows["premise"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    )
    rows = rows.drop_duplicates(["group_id", "hypothesis", "label"]).reset_index(drop=True)
    rows["item_id"] = [
        f"{group_id}:{pair_id}:{index}"
        for index, (group_id, pair_id) in enumerate(zip(rows["group_id"], rows["pairID"].astype(str)))
    ]
    rows["text_length"] = rows["premise"].str.split().str.len() + rows["hypothesis"].str.split().str.len()
    # Split at the normalized-premise group level. MultiNLI normally provides one
    # entailment, neutral, and contradiction hypothesis for the same premise;
    # splitting individual pair IDs would leak that premise across train/test.
    buckets = rows["group_id"].map(lambda value: _stable_hash(value, config.seed) % 100)
    rows["split"] = np.where(buckets.lt(80), "train", np.where(buckets.lt(90), "validation", "test"))
    rows["tie_break"] = rows["item_id"].map(lambda value: _stable_hash(value, config.seed + 1))
    return rows[["group_id", "item_id", "genre", "premise", "hypothesis", "label", "text_length", "split", "tie_break"]].reset_index(drop=True)


def _matched_label_pairs(
    items: pd.DataFrame,
    contrast: MnliContrast,
    *,
    split: str,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    selected = items[items["split"].eq(split)].copy()

    def side(label: str, prefix: str) -> pd.DataFrame:
        frame = (
            selected[selected["label"].eq(label)]
            .sort_values(["group_id", "text_length", "tie_break", "item_id"])
            .drop_duplicates("group_id")
            .copy()
        )
        return frame.rename(
            columns={column: f"{prefix}_{column}" for column in frame.columns if column != "group_id"}
        )

    source = side(contrast.source_label, "source")
    target = side(contrast.target_label, "target")
    pairs = source.merge(target, on="group_id", how="inner", validate="one_to_one")
    if not pairs["source_premise"].eq(pairs["target_premise"]).all():
        raise ValueError("Same-prompt MNLI pairs unexpectedly contain different premises")
    pairs["pair_type"] = contrast.name
    pairs["split"] = split
    pairs["length_difference"] = (pairs["source_text_length"] - pairs["target_text_length"]).abs()
    pairs["pair_hash"] = [
        _stable_hash(f"{contrast.name}:{left}:{right}", seed)
        for left, right in zip(pairs["source_item_id"], pairs["target_item_id"])
    ]
    # Round-robin over genres before truncation so one easy genre cannot
    # dominate the matched control merely because its examples are shorter.
    pairs = pairs.sort_values(["source_genre", "length_difference", "pair_hash"]).copy()
    pairs["genre_rank"] = pairs.groupby("source_genre").cumcount()
    pairs = pairs.sort_values(["genre_rank", "pair_hash"]).head(max_pairs).copy()
    pairs["pair_id"] = [
        f"mnli:{contrast.name}:{left}:{right}"
        for left, right in zip(pairs["source_item_id"], pairs["target_item_id"])
    ]
    return pairs.reset_index(drop=True)


def build_mnli_pairs(items: pd.DataFrame, config: MnliControlConfig) -> pd.DataFrame:
    frames = []
    for contrast in config.contrasts:
        frames.append(
            _matched_label_pairs(
                items,
                contrast,
                split="train",
                max_pairs=config.max_train_pairs_per_contrast,
                seed=config.seed,
            )
        )
        frames.append(
            _matched_label_pairs(
                items,
                contrast,
                split="test",
                max_pairs=config.max_test_pairs_per_contrast,
                seed=config.seed + 7,
            )
        )
    pairs = pd.concat(frames, ignore_index=True, sort=False)
    expected = {(contrast.name, split) for contrast in config.contrasts for split in ("train", "test")}
    observed = set(map(tuple, pairs[["pair_type", "split"]].drop_duplicates().to_numpy()))
    if observed != expected:
        raise ValueError(f"MNLI pairing did not produce every contrast/split: missing {sorted(expected - observed)}")
    return pairs


def _interface_prefix(row: pd.Series, template: MnliTemplate, interface: MnliInterface) -> str:
    lines = [f"Premise: {row['premise']}", f"Hypothesis: {row['hypothesis']}"]
    lines.extend(interface.key_lines)
    lines.append(template.question)
    if interface.kind == "letter_mcq":
        for index, label in enumerate(interface.option_order):
            lines.append(f"{chr(ord('A') + index)}. {interface.label_values[label]}")
    else:
        values = ", ".join(interface.label_values[label] for label in MNLI_LABELS)
        lines.append(f"Valid completions: {values}.")
    lines.extend([template.answer_instruction, "Answer:"])
    return "\n".join(lines)


def _candidate_values(interface: MnliInterface) -> dict[str, str]:
    if interface.kind == "completion":
        return interface.label_values
    return {label: chr(ord("A") + interface.option_order.index(label)) for label in MNLI_LABELS}


def _canonical_interface(config: MnliControlConfig) -> MnliInterface:
    return next(interface for interface in config.interfaces if interface.name == config.reference_interface)


def _endpoint_table(pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for pair in pairs[pairs["split"].eq(split)].itertuples():
        for endpoint, prefix, base_label, target_label, sign in [
            ("source", "source", pair.source_label, pair.target_label, 1.0),
            ("target", "target", pair.target_label, pair.source_label, -1.0),
        ]:
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "group_id": pair.group_id,
                    "pair_type": pair.pair_type,
                    "endpoint": endpoint,
                    "direction_sign": sign,
                    "base_label": base_label,
                    "target_label": target_label,
                    "item_id": getattr(pair, f"{prefix}_item_id"),
                    "premise": getattr(pair, f"{prefix}_premise"),
                    "hypothesis": getattr(pair, f"{prefix}_hypothesis"),
                    "genre": getattr(pair, f"{prefix}_genre"),
                    "text_length": getattr(pair, f"{prefix}_text_length"),
                }
            )
    return pd.DataFrame(rows)


def _extract_directions(
    model: Any,
    tokenizer: Any,
    pairs: pd.DataFrame,
    config: MnliControlConfig,
    model_cfg: MnliModelRun,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    canonical = _canonical_interface(config)
    template = config.templates[0]
    endpoints = _endpoint_table(pairs, "train")
    unique = endpoints.drop_duplicates("item_id").reset_index(drop=True)
    prompts = [_interface_prefix(row, template, canonical) for _, row in unique.iterrows()]
    resolved = resolve_layer_index(model_cfg.locked_layer, len(decoder_layers(model)))
    _, activations = collect_choice_probs_and_activations(
        model,
        tokenizer,
        prompts,
        layer_indices=[resolved],
        batch_size=config.batch_size,
        max_length=config.max_length,
        choice_letters=["A", "B", "C"],
    )
    item_index = {item_id: index for index, item_id in enumerate(unique["item_id"].astype(str))}
    hidden = activations[resolved]
    legacy_rng = np.random.default_rng(config.seed + resolved)
    directions: dict[tuple[str, str], np.ndarray] = {}
    inventory = []
    for contrast in config.contrasts:
        group = pairs[(pairs["split"].eq("train")) & (pairs["pair_type"].eq(contrast.name))]
        source_indices = [item_index[str(value)] for value in group["source_item_id"]]
        target_indices = [item_index[str(value)] for value in group["target_item_id"]]
        raw = (hidden[target_indices] - hidden[source_indices]).mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        modes = {
            "raw_mnli_direction": raw,
        }
        random_seed_by_mode: dict[str, int | None] = {
            "raw_mnli_direction": None,
        }
        if config.include_inverse_control:
            modes["inverse_direction_control"] = -raw
            random_seed_by_mode["inverse_direction_control"] = None
        if config.random_seeds:
            for random_seed in config.random_seeds:
                seed = _stable_hash(contrast.name, random_seed + resolved) % (2**32)
                rng = np.random.default_rng(seed)
                random = rng.normal(size=raw.shape).astype(np.float32)
                random = random / max(float(np.linalg.norm(random)), 1e-8) * norm
                mode = f"random_direction_control_seed_{random_seed}"
                modes[mode] = random.astype(np.float32)
                random_seed_by_mode[mode] = random_seed
        else:
            random = legacy_rng.normal(size=raw.shape).astype(np.float32)
            random = random / max(float(np.linalg.norm(random)), 1e-8) * norm
            modes["random_direction_control"] = random.astype(np.float32)
            random_seed_by_mode["random_direction_control"] = config.seed
        for mode, vector in modes.items():
            directions[(contrast.name, mode)] = vector
            inventory.append(
                {
                    "pair_type": contrast.name,
                    "source_label": contrast.source_label,
                    "target_label": contrast.target_label,
                    "mode": mode,
                    "random_seed": random_seed_by_mode[mode],
                    "extraction_interface": canonical.name,
                    "extraction_template": template.name,
                    "layer_index": resolved,
                    "direction_l2": float(np.linalg.norm(vector)),
                    "n_train_pairs": len(group),
                    "pair_protocol": "same_premise_different_hypothesis_group_split",
                }
            )
    return directions, pd.DataFrame(inventory)


def _evaluate(
    model: Any,
    tokenizer: Any,
    pairs: pd.DataFrame,
    directions: dict[tuple[str, str], np.ndarray],
    config: MnliControlConfig,
    model_cfg: MnliModelRun,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoints = _endpoint_table(pairs, "test")
    rows, tokens, baseline_rows = [], [], []
    canonical_candidates = _candidate_values(_canonical_interface(config))
    for interface in config.interfaces:
        candidates = _candidate_values(interface)
        identifier_to_label = {value: label for label, value in candidates.items()}
        for template in config.templates:
            prefixes = [_interface_prefix(row, template, interface) for _, row in endpoints.iterrows()]
            completions = [[" " + candidates[label] for label in MNLI_LABELS] for _ in prefixes]
            base_sum, base_mean, counts = completion_sequence_logprobs(
                model, tokenizer, prefixes, completions,
                batch_size=config.batch_size, max_length=config.max_length,
            )
            base_primary = base_mean if config.primary_score == "mean_logprob" else base_sum
            predicted = np.argmax(base_primary, axis=1)
            for index, item in endpoints.iterrows():
                gold_index = MNLI_LABELS.index(item.base_label)
                other = np.delete(base_primary[index], gold_index)
                baseline_rows.append(
                    {
                        "group_id": item.group_id,
                        "item_id": item.item_id,
                        "pair_id": item.pair_id,
                        "pair_type": item.pair_type,
                        "interface": interface.name,
                        "interface_kind": interface.kind,
                        "template": template.name,
                        "gold_label": item.base_label,
                        "predicted_label": MNLI_LABELS[int(predicted[index])],
                        "correct": bool(int(predicted[index]) == gold_index),
                        "gold_vs_runner_up_margin": float(
                            base_primary[index, gold_index] - np.max(other)
                        ),
                        "score_normalization": config.primary_score,
                    }
                )
            for label_index, label in enumerate(MNLI_LABELS):
                for count in np.unique(counts[:, label_index]):
                    tokens.append(
                        {
                            "interface": interface.name,
                            "template": template.name,
                            "label": label,
                            "completion": candidates[label],
                            "n_tokens": int(count),
                        }
                    )
            for (pair_type, mode), direction in directions.items():
                pair_mask = endpoints["pair_type"].astype(str).eq(pair_type).to_numpy()
                for sign in (1.0, -1.0):
                    mask = pair_mask & endpoints["direction_sign"].eq(sign).to_numpy()
                    if not mask.any():
                        continue
                    indices = np.flatnonzero(mask)
                    subset = endpoints.loc[mask].reset_index(drop=True)
                    patched_sum, patched_mean, _ = completion_sequence_logprobs(
                        model,
                        tokenizer,
                        [prefixes[index] for index in indices],
                        [completions[index] for index in indices],
                        batch_size=config.batch_size,
                        max_length=config.max_length,
                        layer_index=model_cfg.locked_layer,
                        direction=np.asarray(direction) * sign,
                        alpha=model_cfg.locked_alpha,
                    )
                    base_scores = base_mean[mask] if config.primary_score == "mean_logprob" else base_sum[mask]
                    patched_scores = patched_mean if config.primary_score == "mean_logprob" else patched_sum
                    for index, item in subset.iterrows():
                        target = MNLI_LABELS.index(item.target_label)
                        source = MNLI_LABELS.index(item.base_label)
                        base_margin = base_scores[index, target] - base_scores[index, source]
                        patched_margin = patched_scores[index, target] - patched_scores[index, source]
                        current_label_effect = float(patched_margin - base_margin)
                        extraction_target_id = canonical_candidates[item.target_label]
                        extraction_source_id = canonical_candidates[item.base_label]
                        id_target_label = identifier_to_label.get(extraction_target_id)
                        id_source_label = identifier_to_label.get(extraction_source_id)
                        if id_target_label is not None and id_source_label is not None:
                            id_target = MNLI_LABELS.index(id_target_label)
                            id_source = MNLI_LABELS.index(id_source_label)
                            base_id_margin = base_scores[index, id_target] - base_scores[index, id_source]
                            patched_id_margin = patched_scores[index, id_target] - patched_scores[index, id_source]
                            extraction_id_effect = float(patched_id_margin - base_id_margin)
                            id_advantage = extraction_id_effect - current_label_effect
                        else:
                            base_id_margin = patched_id_margin = np.nan
                            extraction_id_effect = id_advantage = np.nan
                        rows.append(
                            {
                                "pair_id": item.pair_id,
                                "group_id": item.group_id,
                                "pair_type": item.pair_type,
                                "endpoint": item.endpoint,
                                "direction_sign": sign,
                                "mode": mode,
                                "interface": interface.name,
                                "interface_kind": interface.kind,
                                "template": template.name,
                                "base_label": item.base_label,
                                "target_label": item.target_label,
                                "base_target_margin": float(base_margin),
                                "patched_target_margin": float(patched_margin),
                                "target_margin_gain": current_label_effect,
                                "current_label_effect": current_label_effect,
                                "extraction_target_id": extraction_target_id,
                                "extraction_source_id": extraction_source_id,
                                "base_extraction_id_margin": float(base_id_margin),
                                "patched_extraction_id_margin": float(patched_id_margin),
                                "extraction_id_effect": extraction_id_effect,
                                "id_advantage": id_advantage,
                                "score_normalization": config.primary_score,
                            }
                        )
    baseline = pd.DataFrame(baseline_rows).drop_duplicates(
        ["item_id", "interface", "template"]
    )
    return pd.DataFrame(rows), pd.DataFrame(tokens).drop_duplicates(), baseline


def _summaries(rows: pd.DataFrame, config: MnliControlConfig) -> dict[str, pd.DataFrame]:
    if rows.empty:
        return {
            "mnli_pair_effects": pd.DataFrame(),
            "mnli_interface_summary": pd.DataFrame(),
            "mnli_interface_retention": pd.DataFrame(),
        }
    group_prefix = ["model_alias", "model_name"] if "model_alias" in rows else []
    metric_columns = [
        column
        for column in (
            "target_margin_gain",
            "current_label_effect",
            "extraction_id_effect",
            "id_advantage",
        )
        if column in rows.columns
    ]
    pair_identity = ["group_id", "pair_id"] if "group_id" in rows.columns else ["pair_id"]
    pair = (
        rows.groupby(
            [
                *group_prefix,
                *pair_identity,
                "pair_type",
                "mode",
                "interface",
                "interface_kind",
                "template",
            ],
            as_index=False,
        )
        .agg(**{column: (column, "mean") for column in metric_columns})
    )
    summary = (
        pair.groupby([*group_prefix, "mode", "interface", "interface_kind", "template", "pair_type"], as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            **({"n_groups": ("group_id", "nunique")} if "group_id" in pair else {}),
            **{column: (column, "mean") for column in metric_columns},
        )
    )
    reference = pair[pair["interface"].eq(config.reference_interface)].rename(
        columns={"target_margin_gain": "reference_gain"}
    )
    merge_keys = [*group_prefix, "pair_id", "pair_type", "mode", "template"]
    retention = pair.merge(reference[[*merge_keys, "reference_gain"]], on=merge_keys, how="left")
    eligible = retention["reference_gain"].abs().ge(0.01)
    retention["retention_ratio"] = np.where(
        eligible,
        retention["target_margin_gain"] / retention["reference_gain"],
        np.nan,
    )
    retention["sign_consistent"] = np.where(
        eligible,
        np.sign(retention["target_margin_gain"]) == np.sign(retention["reference_gain"]),
        np.nan,
    )
    return {
        "mnli_pair_effects": pair,
        "mnli_interface_summary": summary,
        "mnli_interface_retention": retention,
    }


def _summarize_baseline_accuracy(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    per_item = (
        rows.groupby(
            ["model_alias", "model_name", "group_id", "item_id", "interface", "interface_kind"],
            as_index=False,
        )
        .agg(
            template_accuracy=("correct", "mean"),
            n_templates=("template", "nunique"),
        )
    )
    return (
        per_item.groupby(
            ["model_alias", "model_name", "interface", "interface_kind"],
            as_index=False,
        )
        .agg(
            baseline_accuracy=("template_accuracy", "mean"),
            n_items=("item_id", "nunique"),
            n_groups=("group_id", "nunique"),
        )
    )


def _validate_attribution_outputs(
    rows: pd.DataFrame,
    baseline: pd.DataFrame,
    config: MnliControlConfig,
) -> pd.DataFrame:
    if not config.attribution_analysis:
        return pd.DataFrame()
    expected_models = {item.model.alias for item in config.models}
    expected_interfaces = {item.name for item in config.interfaces}
    expected_randoms = {
        f"random_direction_control_seed_{seed}" for seed in config.random_seeds
    }
    observed_models = set(rows.get("model_alias", pd.Series(dtype=str)).astype(str))
    missing_models = sorted(expected_models - observed_models)
    if missing_models:
        raise RuntimeError(f"MNLI attribution is missing model outputs: {missing_models}")
    if baseline.empty or "model_alias" not in baseline:
        raise RuntimeError("MNLI attribution is missing baseline-competence rows")
    records = []
    for model_alias in sorted(expected_models):
        model_rows = rows[rows["model_alias"].astype(str).eq(model_alias)]
        model_baseline = baseline[baseline["model_alias"].astype(str).eq(model_alias)]
        interfaces = set(model_rows["interface"].astype(str))
        random_modes = {
            mode
            for mode in model_rows["mode"].astype(str).unique()
            if mode.startswith("random_direction_control")
        }
        required_metrics = model_rows[
            ["current_label_effect", "extraction_id_effect", "id_advantage"]
        ]
        if required_metrics.isna().any().any():
            raise RuntimeError(f"MNLI attribution contains missing metrics for {model_alias}")
        identity_error = (
            model_rows["id_advantage"]
            - (model_rows["extraction_id_effect"] - model_rows["current_label_effect"])
        ).abs().max()
        record = {
            "model_alias": model_alias,
            "n_eval_rows": len(model_rows),
            "n_interfaces": len(interfaces),
            "n_modes": int(model_rows["mode"].nunique()),
            "n_random_directions": len(random_modes),
            "n_contrasts": int(model_rows["pair_type"].nunique()),
            "n_templates": int(model_rows["template"].nunique()),
            "n_baseline_rows": len(model_baseline),
            "max_id_advantage_identity_error": float(identity_error),
            "complete": bool(
                interfaces == expected_interfaces
                and random_modes == expected_randoms
                and set(model_baseline["interface"].astype(str)) == expected_interfaces
                and identity_error < 1e-6
            ),
        }
        records.append(record)
    inventory = pd.DataFrame(records)
    if not inventory["complete"].all():
        failed = inventory.loc[~inventory["complete"], "model_alias"].tolist()
        raise RuntimeError(f"MNLI attribution integrity check failed for: {failed}")
    return inventory


def _equal_strata_bootstrap(
    values_by_stratum: list[np.ndarray],
    *,
    n_boot: int,
    confidence: float,
    chunk_size: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values_by_stratum = [np.asarray(values, dtype=float) for values in values_by_stratum if len(values)]
    if not values_by_stratum:
        return np.nan, np.nan, np.nan
    estimate = float(np.mean([values.mean() for values in values_by_stratum]))
    boot = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        chunk = np.zeros(stop - start, dtype=float)
        for values in values_by_stratum:
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            chunk += values[indices].mean(axis=1) / len(values_by_stratum)
        boot[start:stop] = chunk
    tail = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail))


def _ci_record(
    frame: pd.DataFrame,
    *,
    strata_columns: list[str],
    config: MnliControlConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    strata = [
        group["effect"].astype(float).to_numpy()
        for _, group in frame.groupby(strata_columns, sort=True)
    ]
    estimate, low, high = _equal_strata_bootstrap(
        strata,
        n_boot=config.n_boot,
        confidence=config.confidence,
        chunk_size=config.bootstrap_chunk_size,
        rng=rng,
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "n_pairs": int(frame["pair_id"].nunique()),
        "n_strata": len(strata),
        "confidence": config.confidence,
        "estimand": "equal_strata_pair_bootstrap",
    }


def build_mnli_paired_statistics(
    pair_effects: pd.DataFrame,
    config: MnliControlConfig,
) -> dict[str, pd.DataFrame]:
    required = {"model_alias", "pair_id", "pair_type", "mode", "interface", "template", "target_margin_gain"}
    missing = sorted(required - set(pair_effects.columns))
    if missing:
        raise ValueError(f"MNLI pair effects are missing columns: {missing}")
    # Prompt templates are repeated measurements of the same independent pair.
    pair = (
        pair_effects.groupby(
            ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface"],
            as_index=False,
        )
        .agg(effect=("target_margin_gain", "mean"), n_templates=("template", "nunique"))
    )
    rng = np.random.default_rng(config.seed + 101)

    mode_rows = []
    for (model_alias, model_name, mode, interface), group in pair.groupby(
        ["model_alias", "model_name", "mode", "interface"], sort=True
    ):
        record = {
            "model_alias": model_alias,
            "model_name": model_name,
            "mode": mode,
            "interface": interface,
            "comparison": f"{mode}_vs_zero",
        }
        record.update(_ci_record(group, strata_columns=["pair_type"], config=config, rng=rng))
        mode_rows.append(record)

    def paired_mode(left_mode: str, right_mode: str, comparison: str) -> pd.DataFrame:
        keys = ["model_alias", "model_name", "pair_id", "pair_type", "interface"]
        left = pair[pair["mode"].eq(left_mode)][keys + ["effect"]].rename(columns={"effect": "left"})
        right = pair[pair["mode"].eq(right_mode)][keys + ["effect"]].rename(columns={"effect": "right"})
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
        merged["effect"] = merged["left"] - merged["right"]
        merged["comparison"] = comparison
        return merged

    comparisons = [
        paired_mode("raw_mnli_direction", "random_direction_control", "raw_minus_random"),
        paired_mode("raw_mnli_direction", "inverse_direction_control", "raw_minus_inverse"),
    ]
    raw = pair[pair["mode"].eq("raw_mnli_direction")].copy()
    interface_specs = [
        ("letter_canonical", "letter_reversed", "canonical_minus_reversed"),
        ("direct_label_completion", "letter_canonical", "direct_minus_canonical"),
        ("opaque_completion", "letter_canonical", "opaque_minus_canonical"),
    ]
    for left_interface, right_interface, comparison in interface_specs:
        keys = ["model_alias", "model_name", "pair_id", "pair_type"]
        left = raw[raw["interface"].eq(left_interface)][keys + ["effect"]].rename(columns={"effect": "left"})
        right = raw[raw["interface"].eq(right_interface)][keys + ["effect"]].rename(columns={"effect": "right"})
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
        merged["effect"] = merged["left"] - merged["right"]
        merged["comparison"] = comparison
        merged["interface"] = f"{left_interface}_vs_{right_interface}"
        comparisons.append(merged)
    comparison_pairs = pd.concat(comparisons, ignore_index=True, sort=False)

    comparison_rows = []
    for (model_alias, model_name, comparison, interface), group in comparison_pairs.groupby(
        ["model_alias", "model_name", "comparison", "interface"], sort=True
    ):
        record = {
            "model_alias": model_alias,
            "model_name": model_name,
            "comparison": comparison,
            "interface": interface,
        }
        record.update(_ci_record(group, strata_columns=["pair_type"], config=config, rng=rng))
        comparison_rows.append(record)

    global_rows = []
    global_inputs = []
    raw_global = pair[pair["mode"].eq("raw_mnli_direction")].copy()
    for interface, group in raw_global.groupby("interface", sort=True):
        keep = group.copy()
        keep["comparison"] = "raw_vs_zero"
        keep["comparison_interface"] = interface
        global_inputs.append(keep)
    for (comparison, interface), group in comparison_pairs.groupby(["comparison", "interface"], sort=True):
        keep = group.copy()
        keep["comparison_interface"] = str(interface)
        global_inputs.append(keep)
    for group in global_inputs:
        record = {
            "comparison": str(group["comparison"].iloc[0]),
            "interface": str(group["comparison_interface"].iloc[0]),
        }
        record.update(
            _ci_record(group, strata_columns=["model_alias", "pair_type"], config=config, rng=rng)
        )
        record["n_models"] = int(group["model_alias"].nunique())
        record["n_contrasts"] = int(group["pair_type"].nunique())
        global_rows.append(record)

    leave_one_out = []
    for held_out in sorted(pair["model_alias"].astype(str).unique()):
        for group in global_inputs:
            selected = group[group["model_alias"].astype(str).ne(held_out)]
            record = {
                "held_out_model": held_out,
                "comparison": str(group["comparison"].iloc[0]),
                "interface": str(group["comparison_interface"].iloc[0]),
                "estimate": float(
                    selected.groupby(["model_alias", "pair_type"])["effect"].mean().mean()
                ),
                "n_models": int(selected["model_alias"].nunique()),
                "n_contrasts": int(selected["pair_type"].nunique()),
            }
            leave_one_out.append(record)
    return {
        "mnli_mode_interface_bootstrap_ci": pd.DataFrame(mode_rows),
        "mnli_paired_comparison_bootstrap_ci": pd.DataFrame(comparison_rows),
        "mnli_global_bootstrap_ci": pd.DataFrame(global_rows),
        "mnli_leave_one_model_out": pd.DataFrame(leave_one_out),
        "mnli_statistics_pair_rows": pair,
        "mnli_statistics_comparison_pairs": comparison_pairs,
    }


def _cluster_bootstrap_record(
    frame: pd.DataFrame,
    *,
    value_column: str,
    stratum_columns: list[str],
    config: MnliControlConfig,
    seed_key: str,
) -> dict[str, Any]:
    columns = [*stratum_columns, "group_id", value_column]
    cells = frame[columns].dropna(subset=[value_column]).copy()
    if cells.empty:
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_excludes_zero": False,
            "n_groups": 0,
            "n_strata": 0,
            "confidence": config.confidence,
            "estimand": "equal_strata_premise_cluster_bootstrap",
        }
    if not stratum_columns:
        cells["_stratum"] = "all"
        stratum_columns = ["_stratum"]
    cells = (
        cells.groupby([*stratum_columns, "group_id"], as_index=False)[value_column]
        .mean()
    )
    clusters = sorted(cells["group_id"].astype(str).unique())
    strata = list(cells[stratum_columns].drop_duplicates().itertuples(index=False, name=None))
    cluster_index = {value: index for index, value in enumerate(clusters)}
    stratum_index = {value: index for index, value in enumerate(strata)}
    matrix = np.full((len(strata), len(clusters)), np.nan, dtype=np.float64)
    for _, row in cells.iterrows():
        stratum = tuple(row[column] for column in stratum_columns)
        matrix[stratum_index[stratum], cluster_index[str(row["group_id"])]] = float(
            row[value_column]
        )
    estimate = float(np.nanmean(np.nanmean(matrix, axis=1)))
    rng_seed = _stable_hash(seed_key, config.seed + 303) % (2**32)
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(config.n_boot, dtype=np.float64)
    chunk_size = min(config.bootstrap_chunk_size, 250)
    for start in range(0, config.n_boot, chunk_size):
        stop = min(start + chunk_size, config.n_boot)
        indices = rng.integers(
            0,
            len(clusters),
            size=(stop - start, len(clusters)),
        )
        sampled = matrix[:, indices]
        with np.errstate(invalid="ignore"):
            stratum_means = np.nanmean(sampled, axis=2)
            boot[start:stop] = np.nanmean(stratum_means, axis=0)
    tail = (1.0 - config.confidence) / 2.0
    low = float(np.nanquantile(boot, tail))
    high = float(np.nanquantile(boot, 1.0 - tail))
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "n_groups": len(clusters),
        "n_strata": len(strata),
        "confidence": config.confidence,
        "estimand": "equal_strata_premise_cluster_bootstrap",
    }


def build_mnli_attribution_statistics(
    pair_effects: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    config: MnliControlConfig,
) -> dict[str, pd.DataFrame]:
    metrics = ["current_label_effect", "extraction_id_effect", "id_advantage"]
    required = {
        "model_alias",
        "model_name",
        "group_id",
        "pair_id",
        "pair_type",
        "mode",
        "interface",
        "template",
        *metrics,
    }
    missing = sorted(required - set(pair_effects.columns))
    if missing:
        raise ValueError(f"MNLI attribution rows are missing columns: {missing}")
    keys = ["model_alias", "model_name", "group_id", "pair_id", "pair_type", "interface"]
    pair = (
        pair_effects.groupby([*keys, "mode"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in metrics},
            n_templates=("template", "nunique"),
        )
    )
    raw = pair[pair["mode"].eq("raw_mnli_direction")].copy()
    random = pair[pair["mode"].astype(str).str.startswith("random_direction_control")].copy()
    if raw.empty or random.empty:
        raise ValueError("MNLI attribution requires canonical CAA and random-direction rows")
    random_mean = (
        random.groupby(keys, as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in metrics},
            n_random_directions=("mode", "nunique"),
        )
        .rename(columns={metric: f"random_{metric}" for metric in metrics})
    )
    adjusted = raw.merge(random_mean, on=keys, how="inner", validate="one_to_one")
    for metric in metrics:
        adjusted[f"raw_{metric}"] = adjusted[metric]
        adjusted[metric] = adjusted[metric] - adjusted[f"random_{metric}"]
    if adjusted["n_random_directions"].nunique() != 1:
        raise ValueError("MNLI attribution has inconsistent random-control counts")

    ci_rows: list[dict[str, Any]] = []
    for (interface, metric), group in (
        adjusted.melt(
            id_vars=keys,
            value_vars=metrics,
            var_name="metric",
            value_name="effect",
        )
        .groupby(["interface", "metric"], sort=True)
    ):
        record = {"scope": "pooled", "model_alias": "all", "interface": interface, "metric": metric}
        record.update(
            _cluster_bootstrap_record(
                group,
                value_column="effect",
                stratum_columns=["model_alias", "pair_type"],
                config=config,
                seed_key=f"pooled:{interface}:{metric}",
            )
        )
        ci_rows.append(record)
    for (model_alias, interface, metric), group in (
        adjusted.melt(
            id_vars=keys,
            value_vars=metrics,
            var_name="metric",
            value_name="effect",
        )
        .groupby(["model_alias", "interface", "metric"], sort=True)
    ):
        record = {
            "scope": "model",
            "model_alias": model_alias,
            "interface": interface,
            "metric": metric,
        }
        record.update(
            _cluster_bootstrap_record(
                group,
                value_column="effect",
                stratum_columns=["pair_type"],
                config=config,
                seed_key=f"model:{model_alias}:{interface}:{metric}",
            )
        )
        ci_rows.append(record)
    ci = pd.DataFrame(ci_rows)

    pooled = ci[ci["scope"].eq("pooled")].copy()
    decision = pooled.pivot(index="interface", columns="metric", values="estimate").reset_index()
    id_ci = pooled[pooled["metric"].eq("id_advantage")][
        ["interface", "ci_low", "ci_high", "ci_excludes_zero"]
    ].rename(
        columns={
            "ci_low": "id_advantage_ci_low",
            "ci_high": "id_advantage_ci_high",
            "ci_excludes_zero": "id_advantage_ci_excludes_zero",
        }
    )
    decision = decision.merge(id_ci, on="interface", how="left", validate="one_to_one")
    decision["profile"] = np.where(
        decision["id_advantage_ci_low"].gt(0),
        "extraction_id_dominant",
        np.where(
            decision["id_advantage_ci_high"].lt(0),
            "current_label_dominant",
            "mixed_or_unresolved",
        ),
    )

    leave_one_out = []
    for held_out in sorted(adjusted["model_alias"].astype(str).unique()):
        selected = adjusted[adjusted["model_alias"].astype(str).ne(held_out)]
        for interface, group in selected.groupby("interface", sort=True):
            stratum_means = group.groupby(["model_alias", "pair_type"])[metrics].mean()
            record = {
                "held_out_model": held_out,
                "interface": interface,
                "n_models": int(group["model_alias"].nunique()),
                "n_contrasts": int(group["pair_type"].nunique()),
            }
            record.update({metric: float(stratum_means[metric].mean()) for metric in metrics})
            leave_one_out.append(record)

    baseline_ci_rows = []
    if not baseline_rows.empty:
        baseline_items = (
            baseline_rows.groupby(
                ["model_alias", "model_name", "group_id", "item_id", "interface"],
                as_index=False,
            )
            .agg(accuracy=("correct", "mean"), n_templates=("template", "nunique"))
        )
        for (model_alias, model_name, interface), group in baseline_items.groupby(
            ["model_alias", "model_name", "interface"], sort=True
        ):
            record = {
                "model_alias": model_alias,
                "model_name": model_name,
                "interface": interface,
                "n_items": int(group["item_id"].nunique()),
            }
            record.update(
                _cluster_bootstrap_record(
                    group,
                    value_column="accuracy",
                    stratum_columns=[],
                    config=config,
                    seed_key=f"baseline:{model_alias}:{interface}",
                )
            )
            record["chance_accuracy"] = 1.0 / len(MNLI_LABELS)
            record["ci_above_chance"] = bool(record["ci_low"] > record["chance_accuracy"])
            baseline_ci_rows.append(record)

    return {
        "mnli_attribution_random_adjusted_pairs": adjusted,
        "mnli_attribution_group_cluster_ci": ci,
        "mnli_attribution_decision": decision,
        "mnli_attribution_leave_one_model_out": pd.DataFrame(leave_one_out),
        "mnli_baseline_accuracy_group_cluster_ci": pd.DataFrame(baseline_ci_rows),
    }


def run_mnli_statistics_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = MnliControlConfig.from_json(config_path, project_root=project_root)
    pair_path = config.output_dir / "mnli_pair_effects.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"MNLI pair effects do not exist: {pair_path}")
    pair_effects = pd.read_csv(pair_path)
    if config.attribution_analysis:
        baseline_path = config.output_dir / "mnli_baseline_rows.csv"
        if not baseline_path.exists():
            raise FileNotFoundError(f"MNLI baseline rows do not exist: {baseline_path}")
        tables = build_mnli_attribution_statistics(
            pair_effects,
            pd.read_csv(baseline_path),
            config,
        )
        independent_unit = "normalized_premise_group"
        estimand = "equal_model_equal_contrast_premise_cluster_bootstrap"
    else:
        tables = build_mnli_paired_statistics(pair_effects, config)
        independent_unit = "matched_same_premise_pair"
        estimand = "equal_model_equal_contrast_pair_bootstrap"
    tables["mnli_statistics_config"] = pd.DataFrame(
        [
            {
                "n_boot": config.n_boot,
                "confidence": config.confidence,
                "bootstrap_chunk_size": config.bootstrap_chunk_size,
                "seed": config.seed,
                "independent_unit": independent_unit,
                "stratification": estimand,
                "random_seeds": ",".join(map(str, config.random_seeds)),
            }
        ]
    )
    write_tables(tables, config.output_dir / "paired_statistics")
    return tables


def run_mnli_control_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    config = MnliControlConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    if phase not in {"run", "aggregate", "all"}:
        raise ValueError("phase must be run, aggregate, or all")
    items = load_mnli_items(config)
    pairs = build_mnli_pairs(items, config)
    requested = set(model_aliases or [item.model.alias for item in config.models])
    errors = []
    if phase in {"run", "all"}:
        for model_cfg in config.models:
            if model_cfg.model.alias not in requested:
                continue
            model_dir = config.output_dir / model_cfg.model.alias
            if (model_dir / "mnli_run_complete.csv").exists() and not config.force_rerun:
                continue
            tokenizer = model = None
            try:
                tokenizer, model = load_tokenizer_and_model(
                    model_cfg.model.load_source,
                    device_map=model_cfg.model.device_map,
                    torch_dtype=model_cfg.model.torch_dtype,
                )
                directions, inventory = _extract_directions(model, tokenizer, pairs, config, model_cfg)
                rows, tokenization, baseline = _evaluate(
                    model,
                    tokenizer,
                    pairs,
                    directions,
                    config,
                    model_cfg,
                )
                for frame in (rows, tokenization, inventory, baseline):
                    frame.insert(0, "model_alias", model_cfg.model.alias)
                    frame.insert(1, "model_name", model_cfg.model.name)
                write_tables(
                    {
                        "mnli_eval_rows": rows,
                        "mnli_baseline_rows": baseline,
                        "mnli_tokenization": tokenization,
                        "mnli_direction_inventory": inventory,
                        "mnli_run_complete": pd.DataFrame(
                            [
                                {
                                    "model_alias": model_cfg.model.alias,
                                    "status": "complete",
                                    "locked_layer": model_cfg.locked_layer,
                                    "locked_alpha": model_cfg.locked_alpha,
                                    "n_interfaces": len(config.interfaces),
                                    "n_random_directions": len(config.random_seeds) or 1,
                                }
                            ]
                        ),
                    },
                    model_dir,
                )
            except Exception as exc:
                error = {
                    "model_alias": model_cfg.model.alias,
                    "model_name": model_cfg.model.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                write_tables({"mnli_error": pd.DataFrame([error])}, model_dir)
            finally:
                del model, tokenizer
                cleanup_model()

    if phase == "run":
        if errors:
            failed = ", ".join(error["model_alias"] for error in errors)
            raise RuntimeError(f"MNLI model run failed for: {failed}")
        complete = []
        for alias in sorted(requested):
            marker = config.output_dir / alias / "mnli_run_complete.csv"
            if marker.exists():
                complete.append(pd.read_csv(marker))
        return {
            "mnli_run_complete": (
                pd.concat(complete, ignore_index=True, sort=False)
                if complete
                else pd.DataFrame()
            )
        }

    row_files = sorted(config.output_dir.glob("*/mnli_eval_rows.csv"))
    baseline_files = sorted(config.output_dir.glob("*/mnli_baseline_rows.csv"))
    inventory_files = sorted(config.output_dir.glob("*/mnli_direction_inventory.csv"))
    token_files = sorted(config.output_dir.glob("*/mnli_tokenization.csv"))
    rows = pd.concat([pd.read_csv(path) for path in row_files], ignore_index=True, sort=False) if row_files else pd.DataFrame()
    baseline = (
        pd.concat([pd.read_csv(path) for path in baseline_files], ignore_index=True, sort=False)
        if baseline_files
        else pd.DataFrame()
    )
    attribution_integrity = _validate_attribution_outputs(rows, baseline, config)
    tables = _summaries(rows, config)
    tables["mnli_baseline_summary"] = _summarize_baseline_accuracy(baseline)
    tables["mnli_attribution_integrity"] = attribution_integrity
    pair_inventory = (
        pairs.groupby(["split", "pair_type"], as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_length_difference=("length_difference", "mean"),
            median_length_difference=("length_difference", "median"),
            n_genres=("source_genre", "nunique"),
        )
    )
    tables.update(
        {
            "mnli_eval_rows": rows,
            "mnli_baseline_rows": baseline,
            "mnli_direction_inventory": pd.concat([pd.read_csv(path) for path in inventory_files], ignore_index=True, sort=False) if inventory_files else pd.DataFrame(),
            "mnli_tokenization": pd.concat([pd.read_csv(path) for path in token_files], ignore_index=True, sort=False) if token_files else pd.DataFrame(),
            "mnli_pair_inventory": pair_inventory,
            "mnli_pair_examples": pairs.head(30),
            "mnli_errors": pd.DataFrame(errors),
            "mnli_control_config": pd.DataFrame(
                [
                    {
                        "protocol_version": "v2_same_premise",
                        "split_unit": "normalized_premise_hash",
                        "pair_protocol": "same_premise_different_hypothesis",
                        "reference_interface": config.reference_interface,
                        "primary_score": config.primary_score,
                        "attribution_analysis": config.attribution_analysis,
                        "random_seeds": ",".join(map(str, config.random_seeds)),
                    }
                ]
            ),
        }
    )
    write_tables(tables, config.output_dir)
    if errors:
        failed = ", ".join(error["model_alias"] for error in errors)
        raise RuntimeError(f"MNLI model run failed for: {failed}")
    return tables
