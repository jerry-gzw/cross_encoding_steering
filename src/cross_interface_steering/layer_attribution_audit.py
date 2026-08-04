from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .decomposition import contrast_directions
from .io import write_tables
from .layer_attribution_statistics import (
    build_layer_group_cluster_statistics,
)
from .mapping_audit import (
    FixedDirectionMappingAuditConfig,
    MappingAuditModelConfig,
    _load_mapping_audit_config,
    annotate_model_frame,
    build_canonical_train_index,
    build_mapping_eval_items,
    build_mapping_items,
    build_mapping_mode_ci,
    build_pair_effects,
    evaluate_fixed_directions,
    load_base_items,
)
from .multimodel import resolve_fractional_layers
from .readout_baseline_audit import (
    _oriented_gradient_subset,
    _rescale,
)
from .readout_geometry_audit import (
    collect_local_identifier_gradients,
    decompose_against_subspace,
    sample_gradient_prompts,
    select_readout_rank,
)
from .steering import (
    cleanup_model,
    collect_choice_probs_and_activations,
    decoder_layers,
    load_tokenizer_and_model,
)


LAYER_ATTRIBUTION_MODES = (
    "raw_canonical_direction",
    "mean_local_gradient_norm_matched",
    "readout_projection_norm_matched",
    "readout_orthogonal_norm_matched",
)


@dataclass(frozen=True)
class LayerAttributionSettings:
    layer_fractions: tuple[float, ...] = (0.5, 0.625, 0.75, 0.875)
    rank_candidates: tuple[int, ...] = (2, 4, 8, 16)
    validation_explained_energy_threshold: float = 0.9
    max_train_prompts_per_contrast: int = 64
    max_validation_prompts_per_contrast: int = 32
    gradient_batch_size: int = 1
    normalize_gradients: bool = True
    enabled_modes: tuple[str, ...] = LAYER_ATTRIBUTION_MODES


@dataclass(frozen=True)
class LayerAttributionAuditConfig:
    mapping_audit: FixedDirectionMappingAuditConfig
    settings: LayerAttributionSettings

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "LayerAttributionAuditConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        audit = FixedDirectionMappingAuditConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        raw = dict(data.get("layer_attribution", {}))
        layer_fractions = tuple(
            float(value)
            for value in raw.pop(
                "layer_fractions", (0.5, 0.625, 0.75, 0.875)
            )
        )
        rank_candidates = tuple(
            int(value)
            for value in raw.pop("rank_candidates", (2, 4, 8, 16))
        )
        enabled_modes = tuple(
            str(value)
            for value in raw.pop("enabled_modes", LAYER_ATTRIBUTION_MODES)
        )
        if not layer_fractions or any(
            not 0.0 <= value <= 1.0 for value in layer_fractions
        ):
            raise ValueError(
                "layer_attribution.layer_fractions must contain values in [0, 1]"
            )
        if not rank_candidates or min(rank_candidates) <= 0:
            raise ValueError(
                "layer_attribution.rank_candidates must contain positive integers"
            )
        unknown_modes = sorted(
            set(enabled_modes) - set(LAYER_ATTRIBUTION_MODES)
        )
        if unknown_modes:
            raise ValueError(
                f"Unknown layer-attribution modes: {unknown_modes}"
            )
        settings = LayerAttributionSettings(
            layer_fractions=layer_fractions,
            rank_candidates=rank_candidates,
            enabled_modes=enabled_modes,
            **raw,
        )
        if settings.max_train_prompts_per_contrast <= 0:
            raise ValueError(
                "max_train_prompts_per_contrast must be positive"
            )
        if settings.max_validation_prompts_per_contrast <= 0:
            raise ValueError(
                "max_validation_prompts_per_contrast must be positive"
            )
        if settings.gradient_batch_size <= 0:
            raise ValueError("gradient_batch_size must be positive")
        return cls(mapping_audit=audit, settings=settings)

    @property
    def output_dir(self) -> Path:
        return self.mapping_audit.output_dir


def _fingerprint(vector: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(vector, dtype=np.float32).tobytes()
    ).hexdigest()[:16]


def _mean_oriented_gradient(
    gradients: np.ndarray,
    inventory: pd.DataFrame,
    *,
    source_identifier: str,
    target_identifier: str,
    normalize_rows: bool,
) -> tuple[np.ndarray, int]:
    selected = _oriented_gradient_subset(
        gradients,
        inventory,
        source_identifier=source_identifier,
        target_identifier=target_identifier,
    )
    if normalize_rows:
        norms = np.linalg.norm(selected, axis=1, keepdims=True)
        selected = selected[norms[:, 0] > 0]
        norms = norms[norms[:, 0] > 0]
        selected = selected / norms
    if not len(selected):
        raise ValueError("No nonzero oriented gradients remain")
    return selected.mean(axis=0), int(len(selected))


def build_layer_direction_bank(
    *,
    raw_directions: dict[str, np.ndarray],
    readout_basis: np.ndarray,
    train_gradients: np.ndarray,
    train_gradient_inventory: pd.DataFrame,
    canonical_labels: tuple[str, ...],
    normalize_gradient_rows: bool,
    enabled_modes: tuple[str, ...],
    extraction_mapping: str,
    layer_index: int,
    requested_layer_fraction: float,
    resolved_layer_fraction: float,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    identifiers = tuple(
        chr(ord("A") + index) for index in range(len(canonical_labels))
    )
    identifier_by_label = dict(zip(canonical_labels, identifiers))
    directions: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    saved_arrays: dict[str, np.ndarray] = {
        "readout_basis": np.asarray(readout_basis, dtype=np.float32)
    }
    for pair_type, raw in sorted(raw_directions.items()):
        labels = str(pair_type).split("_vs_", maxsplit=1)
        if len(labels) != 2 or any(
            label not in identifier_by_label for label in labels
        ):
            raise ValueError(f"Cannot parse configured contrast {pair_type!r}")
        source_label, target_label = labels
        source_identifier = identifier_by_label[source_label]
        target_identifier = identifier_by_label[target_label]
        mean_gradient, n_gradient_vectors = _mean_oriented_gradient(
            train_gradients,
            train_gradient_inventory,
            source_identifier=source_identifier,
            target_identifier=target_identifier,
            normalize_rows=normalize_gradient_rows,
        )
        decomposition = decompose_against_subspace(raw, readout_basis)
        raw_norm = float(decomposition["raw_l2"])
        candidates = {
            "raw_canonical_direction": np.asarray(
                decomposition["raw"], dtype=np.float32
            ),
            "mean_local_gradient_norm_matched": _rescale(
                mean_gradient, raw_norm, label="mean local gradient"
            ),
            "readout_projection_norm_matched": np.asarray(
                decomposition["projection_norm_matched"], dtype=np.float32
            ),
            "readout_orthogonal_norm_matched": np.asarray(
                decomposition["residual_norm_matched"], dtype=np.float32
            ),
        }
        for mode, vector in candidates.items():
            if mode not in enabled_modes:
                continue
            directions[(pair_type, mode)] = vector
            saved_arrays[f"{pair_type}__{mode}"] = vector
            rows.append(
                {
                    "pair_type": pair_type,
                    "mode": mode,
                    "source_label": source_label,
                    "target_label": target_label,
                    "source_identifier": source_identifier,
                    "target_identifier": target_identifier,
                    "extraction_mapping": extraction_mapping,
                    "layer_index": int(layer_index),
                    "requested_layer_fraction": float(
                        requested_layer_fraction
                    ),
                    "resolved_layer_fraction": float(
                        resolved_layer_fraction
                    ),
                    "direction_l2": float(np.linalg.norm(vector)),
                    "raw_caa_l2": raw_norm,
                    "readout_projection_l2": float(
                        decomposition["projection_l2"]
                    ),
                    "readout_orthogonal_l2": float(
                        decomposition["residual_l2"]
                    ),
                    "readout_projection_energy_fraction": float(
                        decomposition["projection_energy_fraction"]
                    ),
                    "readout_orthogonal_energy_fraction": float(
                        decomposition["residual_energy_fraction"]
                    ),
                    "n_gradient_vectors": n_gradient_vectors,
                    "readout_subspace_rank": int(readout_basis.shape[1]),
                    "direction_sha256": _fingerprint(vector),
                }
            )
    if not directions:
        raise ValueError("No layer-attribution directions were built")
    return directions, pd.DataFrame(rows), saved_arrays


def _layer_dir(model_dir: Path, layer_index: int) -> Path:
    return model_dir / f"layer_{int(layer_index):02d}"


def layer_run_is_complete(layer_dir: str | Path) -> bool:
    root = Path(layer_dir)
    return all(
        (root / filename).exists()
        for filename in [
            "layer_attribution_layer_complete.csv",
            "layer_attribution_direction_inventory.csv",
            "layer_attribution_pair_effects.csv",
            "layer_attribution_mapping_mode_ci.csv",
        ]
    )


def _write_layer_tables(
    *,
    layer_dir: Path,
    model_config: MappingAuditModelConfig,
    layer_row: pd.Series,
    tables: dict[str, pd.DataFrame],
    arrays: dict[str, np.ndarray],
) -> None:
    alias = model_config.model.alias
    annotated = {
        name: annotate_model_frame(
            frame,
            model_alias=alias,
            model_name=model_config.model.name,
        )
        for name, frame in tables.items()
    }
    write_tables(annotated, layer_dir)
    np.savez_compressed(
        layer_dir / "layer_attribution_arrays.npz", **arrays
    )
    pd.DataFrame(
        [
            {
                "model_alias": alias,
                "model_name": model_config.model.name,
                "status": "complete",
                "layer_index": int(layer_row["resolved_layer_index"]),
                "requested_layer_fraction": float(
                    layer_row["requested_layer_fraction"]
                ),
                "resolved_layer_fraction": float(
                    layer_row["resolved_layer_fraction"]
                ),
                "n_decoder_layers": int(layer_row["n_decoder_layers"]),
                "alpha": float(model_config.locked_alpha),
            }
        ]
    ).to_csv(
        layer_dir / "layer_attribution_layer_complete.csv", index=False
    )


def _collect_layer_tables(model_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "layer_attribution_layer_complete",
        "layer_attribution_rank_selection",
        "layer_attribution_direction_inventory",
        "layer_attribution_mapping_mode_ci",
        "layer_attribution_pair_effects",
        "layer_attribution_model_layer_summary",
    ]
    collected: dict[str, list[pd.DataFrame]] = {}
    for layer_dir in sorted(model_dir.glob("layer_*")):
        if not layer_run_is_complete(layer_dir):
            continue
        for name in names:
            path = layer_dir / f"{name}.csv"
            if path.exists():
                collected.setdefault(name, []).append(pd.read_csv(path))
    return {
        name: pd.concat(parts, ignore_index=True, sort=False)
        for name, parts in collected.items()
    }


def _summarize_layer_effects(pair_effects: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "semantic_prob_gain",
        "original_letter_prob_gain",
        "semantic_minus_original_prob_gain_moved",
        "js_shift",
    ]
    return (
        pair_effects.groupby(
            [
                "layer_index",
                "requested_layer_fraction",
                "resolved_layer_fraction",
                "mapping_name",
                "mapping_family",
                "mode",
            ],
            as_index=False,
        )[metrics]
        .mean()
        .rename(columns={metric: f"mean_{metric}" for metric in metrics})
    )


def run_single_model_layer_attribution(
    model: Any,
    tokenizer: Any,
    config: LayerAttributionAuditConfig,
    model_config: MappingAuditModelConfig,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    audit = config.mapping_audit
    alias = model_config.model.alias
    model_dir = audit.output_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    resolved_layers, layer_map = resolve_fractional_layers(
        model, config.settings.layer_fractions
    )
    pending_rows = [
        row
        for _, row in layer_map.iterrows()
        if not layer_run_is_complete(
            _layer_dir(model_dir, int(row["resolved_layer_index"]))
        )
        or audit.runtime.force_rerun
    ]
    if not pending_rows:
        progress(f"[{alias}] all requested layers already complete")
        return _collect_layer_tables(model_dir)

    base_items = load_base_items(audit.dataset, seed=audit.runtime.seed)
    canonical_mapping = next(
        mapping
        for mapping in audit.mappings
        if mapping.name == audit.dataset.canonical_mapping
    )
    canonical_items = build_mapping_items(
        base_items, audit.dataset, canonical_mapping, canonical_mapping
    )
    train_items, train_pairs = build_canonical_train_index(
        canonical_items, audit.dataset
    )
    pending_layers = [
        int(row["resolved_layer_index"]) for row in pending_rows
    ]
    identifiers = tuple(
        chr(ord("A") + index)
        for index in range(len(audit.dataset.label_ranks))
    )
    progress(
        f"[{alias}] collecting CAA train activations for layers "
        f"{pending_layers}"
    )
    _, activations = collect_choice_probs_and_activations(
        model,
        tokenizer,
        train_items["prompt"].tolist(),
        layer_indices=pending_layers,
        batch_size=audit.runtime.batch_size,
        max_length=audit.runtime.max_length,
        choice_letters=list(identifiers),
    )
    raw_by_layer = {}
    for layer_index in pending_layers:
        raw_by_layer[layer_index], _ = contrast_directions(
            train_pairs,
            activations[layer_index],
            train_split=audit.dataset.train_split,
        )
    del activations

    train_gradient_items = sample_gradient_prompts(
        canonical_items,
        split_column=audit.dataset.split_column,
        train_split=audit.dataset.train_split,
        contrast_column=audit.dataset.contrast_column,
        max_per_contrast=config.settings.max_train_prompts_per_contrast,
        seed=audit.runtime.seed,
    )
    validation_gradient_items = sample_gradient_prompts(
        canonical_items,
        split_column=audit.dataset.split_column,
        train_split=audit.dataset.validation_split,
        contrast_column=audit.dataset.contrast_column,
        max_per_contrast=(
            config.settings.max_validation_prompts_per_contrast
        ),
        seed=audit.runtime.seed + 1,
    )

    for layer_row in pending_rows:
        layer_index = int(layer_row["resolved_layer_index"])
        layer_dir = _layer_dir(model_dir, layer_index)
        layer_dir.mkdir(parents=True, exist_ok=True)
        progress(f"[{alias}] layer {layer_index}: collecting gradients")
        try:
            train_gradients, train_inventory = (
                collect_local_identifier_gradients(
                    model,
                    tokenizer,
                    train_gradient_items["prompt"].tolist(),
                    layer_index=layer_index,
                    identifiers=identifiers,
                    batch_size=config.settings.gradient_batch_size,
                    max_length=audit.runtime.max_length,
                )
            )
            validation_gradients, validation_inventory = (
                collect_local_identifier_gradients(
                    model,
                    tokenizer,
                    validation_gradient_items["prompt"].tolist(),
                    layer_index=layer_index,
                    identifiers=identifiers,
                    batch_size=config.settings.gradient_batch_size,
                    max_length=audit.runtime.max_length,
                )
            )
            basis, _, rank_selection = select_readout_rank(
                train_gradients,
                validation_gradients,
                rank_candidates=config.settings.rank_candidates,
                explained_energy_threshold=(
                    config.settings.validation_explained_energy_threshold
                ),
                normalize_gradients=config.settings.normalize_gradients,
            )
            directions, direction_inventory, arrays = (
                build_layer_direction_bank(
                    raw_directions=raw_by_layer[layer_index],
                    readout_basis=basis,
                    train_gradients=train_gradients,
                    train_gradient_inventory=train_inventory,
                    canonical_labels=tuple(canonical_mapping.option_order),
                    normalize_gradient_rows=(
                        config.settings.normalize_gradients
                    ),
                    enabled_modes=config.settings.enabled_modes,
                    extraction_mapping=audit.dataset.canonical_mapping,
                    layer_index=layer_index,
                    requested_layer_fraction=float(
                        layer_row["requested_layer_fraction"]
                    ),
                    resolved_layer_fraction=float(
                        layer_row["resolved_layer_fraction"]
                    ),
                )
            )
            rank_selection["layer_index"] = layer_index
            rank_selection["requested_layer_fraction"] = float(
                layer_row["requested_layer_fraction"]
            )
            rank_selection["resolved_layer_fraction"] = float(
                layer_row["resolved_layer_fraction"]
            )
            arrays["train_local_gradients"] = train_gradients
            arrays["validation_local_gradients"] = validation_gradients

            all_eval_rows = []
            all_pair_effects = []
            all_ci = []
            progress(
                f"[{alias}] layer {layer_index}: evaluating "
                f"{len(audit.mappings)} mappings"
            )
            for mapping in audit.mappings:
                mapping_items = build_mapping_items(
                    base_items, audit.dataset, mapping, canonical_mapping
                )
                eval_items = build_mapping_eval_items(
                    mapping_items,
                    audit.dataset,
                    mapping,
                    canonical_mapping,
                )
                eval_rows = evaluate_fixed_directions(
                    model,
                    tokenizer,
                    eval_items,
                    directions,
                    direction_inventory,
                    layer_index=layer_index,
                    alpha=model_config.locked_alpha,
                    n_choices=len(audit.dataset.label_ranks),
                    batch_size=audit.runtime.batch_size,
                    max_length=audit.runtime.max_length,
                    injection_position="pre_answer",
                )
                eval_rows["requested_layer_fraction"] = float(
                    layer_row["requested_layer_fraction"]
                )
                eval_rows["resolved_layer_fraction"] = float(
                    layer_row["resolved_layer_fraction"]
                )
                pair_effects = build_pair_effects(eval_rows)
                pair_effects["layer_index"] = layer_index
                pair_effects["requested_layer_fraction"] = float(
                    layer_row["requested_layer_fraction"]
                )
                pair_effects["resolved_layer_fraction"] = float(
                    layer_row["resolved_layer_fraction"]
                )
                ci = build_mapping_mode_ci(
                    pair_effects,
                    n_boot=audit.statistics.n_boot,
                    confidence=audit.statistics.confidence,
                    seed=audit.runtime.seed,
                )
                ci["layer_index"] = layer_index
                ci["requested_layer_fraction"] = float(
                    layer_row["requested_layer_fraction"]
                )
                ci["resolved_layer_fraction"] = float(
                    layer_row["resolved_layer_fraction"]
                )
                all_eval_rows.append(eval_rows)
                all_pair_effects.append(pair_effects)
                all_ci.append(ci)

            pair_effects = pd.concat(
                all_pair_effects, ignore_index=True, sort=False
            )
            tables = {
                "layer_attribution_rank_selection": rank_selection,
                "layer_attribution_direction_inventory": direction_inventory,
                "layer_attribution_eval_rows": pd.concat(
                    all_eval_rows, ignore_index=True, sort=False
                ),
                "layer_attribution_pair_effects": pair_effects,
                "layer_attribution_mapping_mode_ci": pd.concat(
                    all_ci, ignore_index=True, sort=False
                ),
                "layer_attribution_model_layer_summary": (
                    _summarize_layer_effects(pair_effects)
                ),
            }
            _write_layer_tables(
                layer_dir=layer_dir,
                model_config=model_config,
                layer_row=layer_row,
                tables=tables,
                arrays=arrays,
            )
            (layer_dir / "layer_attribution_layer_error.csv").unlink(
                missing_ok=True
            )
            progress(f"[{alias}] layer {layer_index}: complete")
        except Exception as exc:
            error = {
                "model_alias": alias,
                "model_name": model_config.model.name,
                "layer_index": layer_index,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            pd.DataFrame([error]).to_csv(
                layer_dir / "layer_attribution_layer_error.csv",
                index=False,
            )
            progress(
                f"[{alias}] layer {layer_index} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if not audit.runtime.continue_on_error:
                raise

    model_tables = _collect_layer_tables(model_dir)
    write_tables(model_tables, model_dir)
    completed_layers = set(
        model_tables.get(
            "layer_attribution_layer_complete", pd.DataFrame()
        ).get("layer_index", pd.Series(dtype=int))
        .astype(int)
        .tolist()
    )
    if completed_layers == set(resolved_layers):
        pd.DataFrame(
            [
                {
                    "model_alias": alias,
                    "model_name": model_config.model.name,
                    "status": "complete",
                    "resolved_layers": ",".join(map(str, resolved_layers)),
                    "layer_fractions": ",".join(
                        str(value)
                        for value in config.settings.layer_fractions
                    ),
                    "alpha": float(model_config.locked_alpha),
                }
            ]
        ).to_csv(
            model_dir / "layer_attribution_run_complete.csv", index=False
        )
    return model_tables


def model_run_is_complete(
    model_dir: Path, expected_layers: Iterable[int]
) -> bool:
    marker = model_dir / "layer_attribution_run_complete.csv"
    return marker.exists() and all(
        layer_run_is_complete(_layer_dir(model_dir, layer))
        for layer in expected_layers
    )


def _collect_model_tables(
    config: LayerAttributionAuditConfig,
) -> dict[str, pd.DataFrame]:
    names = [
        "layer_attribution_run_complete",
        "layer_attribution_layer_complete",
        "layer_attribution_rank_selection",
        "layer_attribution_direction_inventory",
        "layer_attribution_mapping_mode_ci",
        "layer_attribution_pair_effects",
        "layer_attribution_model_layer_summary",
    ]
    collected: dict[str, list[pd.DataFrame]] = {}
    for model_config in config.mapping_audit.models:
        model_dir = config.output_dir / model_config.model.alias
        for name in names:
            path = model_dir / f"{name}.csv"
            if path.exists():
                collected.setdefault(name, []).append(pd.read_csv(path))
    return {
        name: pd.concat(parts, ignore_index=True, sort=False)
        for name, parts in collected.items()
    }


def aggregate_layer_attribution(
    config: LayerAttributionAuditConfig,
) -> dict[str, pd.DataFrame]:
    aggregated = _collect_model_tables(config)
    effects = aggregated.get(
        "layer_attribution_pair_effects", pd.DataFrame()
    )
    if effects.empty:
        aggregated["layer_attribution_global_summary"] = pd.DataFrame()
        aggregated["layer_attribution_alternative_profile"] = pd.DataFrame()
        aggregated["layer_attribution_retention_summary"] = pd.DataFrame()
        write_tables(aggregated, config.output_dir)
        return aggregated

    metrics = [
        "semantic_prob_gain",
        "original_letter_prob_gain",
        "semantic_minus_original_prob_gain_moved",
        "js_shift",
    ]
    strata = (
        effects.groupby(
            [
                "model_alias",
                "pair_type",
                "layer_index",
                "requested_layer_fraction",
                "resolved_layer_fraction",
                "mapping_name",
                "mapping_family",
                "mode",
            ],
            as_index=False,
        )[metrics]
        .mean()
    )
    global_summary = (
        strata.groupby(
            [
                "requested_layer_fraction",
                "mapping_name",
                "mapping_family",
                "mode",
            ],
            as_index=False,
        )
        .agg(
            n_models=("model_alias", "nunique"),
            n_contrasts=("pair_type", "nunique"),
            min_layer_index=("layer_index", "min"),
            max_layer_index=("layer_index", "max"),
            mean_resolved_layer_fraction=(
                "resolved_layer_fraction", "mean"
            ),
            mean_current_label_gain=("semantic_prob_gain", "mean"),
            mean_extraction_identifier_gain=(
                "original_letter_prob_gain", "mean"
            ),
            mean_current_minus_identifier=(
                "semantic_minus_original_prob_gain_moved", "mean"
            ),
            mean_js_shift=("js_shift", "mean"),
        )
    )
    canonical = config.mapping_audit.dataset.canonical_mapping
    alternative = strata.loc[
        strata["mapping_name"].astype(str).ne(canonical)
    ]
    alternative_profile = (
        alternative.groupby(
            [
                "model_alias",
                "layer_index",
                "requested_layer_fraction",
                "resolved_layer_fraction",
                "mode",
            ],
            as_index=False,
        )
        .agg(
            n_contrasts=("pair_type", "nunique"),
            n_mappings=("mapping_name", "nunique"),
            mean_current_label_gain=("semantic_prob_gain", "mean"),
            mean_extraction_identifier_gain=(
                "original_letter_prob_gain", "mean"
            ),
            mean_current_minus_identifier=(
                "semantic_minus_original_prob_gain_moved", "mean"
            ),
            mean_js_shift=("js_shift", "mean"),
        )
    )
    aggregate_profile = (
        alternative_profile.groupby(
            ["requested_layer_fraction", "mode"],
            as_index=False,
        )
        .agg(
            n_models=("model_alias", "nunique"),
            min_layer_index=("layer_index", "min"),
            max_layer_index=("layer_index", "max"),
            mean_resolved_layer_fraction=(
                "resolved_layer_fraction", "mean"
            ),
            mean_current_label_gain=("mean_current_label_gain", "mean"),
            mean_extraction_identifier_gain=(
                "mean_extraction_identifier_gain", "mean"
            ),
            mean_current_minus_identifier=(
                "mean_current_minus_identifier", "mean"
            ),
            mean_js_shift=("mean_js_shift", "mean"),
        )
    )
    wide = aggregate_profile.pivot_table(
        index=["requested_layer_fraction"],
        columns="mode",
        values="mean_extraction_identifier_gain",
    ).reset_index()
    raw_column = "raw_canonical_direction"
    if raw_column in wide:
        for mode in [
            "mean_local_gradient_norm_matched",
            "readout_projection_norm_matched",
            "readout_orthogonal_norm_matched",
        ]:
            if mode in wide:
                wide[f"{mode}_over_raw"] = np.divide(
                    wide[mode],
                    wide[raw_column],
                    out=np.full(len(wide), np.nan, dtype=float),
                    where=np.abs(wide[raw_column].to_numpy(dtype=float))
                    > 1e-12,
                )
    aggregated["layer_attribution_global_summary"] = global_summary
    aggregated[
        "layer_attribution_alternative_profile_by_model"
    ] = alternative_profile
    aggregated["layer_attribution_alternative_profile"] = (
        aggregate_profile
    )
    aggregated["layer_attribution_retention_summary"] = wide
    pairs_path = config.mapping_audit.dataset.items_path.parent / "pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(
            pairs_path,
            usecols=["pair_id", "pair_type", "group_key", "split"],
        )
        group_ci, depth_contrasts = build_layer_group_cluster_statistics(
            effects,
            pairs,
            canonical_mapping=canonical,
            n_boot=config.mapping_audit.statistics.n_boot,
            confidence=config.mapping_audit.statistics.confidence,
            seed=config.mapping_audit.runtime.seed,
        )
        aggregated["layer_attribution_group_cluster_bootstrap_ci"] = (
            group_ci
        )
        aggregated["layer_attribution_depth_contrast_group_cluster_ci"] = (
            depth_contrasts
        )
    write_tables(aggregated, config.output_dir)
    return aggregated


def run_layer_attribution_audit(
    config: LayerAttributionAuditConfig,
    *,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
    model_loader: Callable[..., tuple[Any, Any]] = load_tokenizer_and_model,
    progress: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    if phase not in {"run", "aggregate", "all"}:
        raise ValueError("phase must be run, aggregate, or all")
    audit = config.mapping_audit
    audit.output_dir.mkdir(parents=True, exist_ok=True)
    if phase == "aggregate":
        return aggregate_layer_attribution(config)
    requested = set(
        model_aliases
        or [
            item.model.alias
            for item in audit.models
            if item.model.enabled
        ]
    )
    known = {item.model.alias for item in audit.models}
    unknown = sorted(requested - known)
    if unknown:
        raise KeyError(f"Unknown model aliases: {unknown}")
    errors = []
    for model_config in audit.models:
        public_model = model_config.model
        if not public_model.enabled or public_model.alias not in requested:
            continue
        tokenizer = None
        model = None
        model_dir = audit.output_dir / public_model.alias
        try:
            progress(f"[{public_model.alias}] loading {public_model.load_source}")
            tokenizer, model = model_loader(
                public_model.load_source,
                device_map=public_model.device_map,
                torch_dtype=public_model.torch_dtype,
            )
            expected_layers, _ = resolve_fractional_layers(
                model, config.settings.layer_fractions
            )
            if (
                model_run_is_complete(model_dir, expected_layers)
                and not audit.runtime.force_rerun
            ):
                progress(f"[{public_model.alias}] complete; skipping")
                continue
            run_single_model_layer_attribution(
                model,
                tokenizer,
                config,
                model_config,
                progress=progress,
            )
            (model_dir / "layer_attribution_run_error.csv").unlink(
                missing_ok=True
            )
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
            pd.DataFrame([error]).to_csv(
                model_dir / "layer_attribution_run_error.csv",
                index=False,
            )
            progress(
                f"[{public_model.alias}] failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if not audit.runtime.continue_on_error:
                raise
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            cleanup_model()
    if errors:
        pd.DataFrame(errors).to_csv(
            audit.output_dir / "layer_attribution_errors.csv", index=False
        )
    else:
        (audit.output_dir / "layer_attribution_errors.csv").unlink(
            missing_ok=True
        )
    if phase == "run":
        return _collect_model_tables(config)
    return aggregate_layer_attribution(config)


def run_layer_attribution_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    config = LayerAttributionAuditConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    return run_layer_attribution_audit(
        config,
        model_aliases=model_aliases,
        phase=phase,
    )
