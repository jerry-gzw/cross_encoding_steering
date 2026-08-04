from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .io import write_tables
from .mapping_audit import (
    FixedDirectionMappingAuditConfig,
    MappingAuditModelConfig,
    _load_mapping_audit_config,
    annotate_model_frame,
    build_mapping_eval_items,
    build_mapping_items,
    build_mapping_mode_ci,
    build_pair_effects,
    evaluate_fixed_directions,
    load_base_items,
)
from .readout_geometry_audit import _strict_identifier_token_ids
from .steering import cleanup_model, load_tokenizer_and_model


BASELINE_MODES = (
    "mean_local_gradient_norm_matched",
    "rank1_local_gradient_norm_matched",
    "unembedding_difference_norm_matched",
)
REFERENCE_MODES = (
    "raw_canonical_direction",
    "readout_projection_norm_matched",
)


@dataclass(frozen=True)
class ReadoutBaselineSettings:
    source_readout_dir: Path
    normalize_gradient_rows: bool = True
    enabled_modes: tuple[str, ...] = BASELINE_MODES
    reference_modes: tuple[str, ...] = REFERENCE_MODES


@dataclass(frozen=True)
class ReadoutBaselineAuditConfig:
    mapping_audit: FixedDirectionMappingAuditConfig
    settings: ReadoutBaselineSettings

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "ReadoutBaselineAuditConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        audit = FixedDirectionMappingAuditConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        settings_data = dict(data.get("readout_baselines", {}))
        source_value = settings_data.pop(
            "source_readout_dir",
            "outputs/group_disjoint_normbank/"
            "readout_geometry_audit_v2_full",
        )
        source_path = Path(str(source_value)).expanduser()
        if not source_path.is_absolute():
            source_path = audit.project_root / source_path
        enabled_modes = tuple(
            str(value)
            for value in settings_data.pop("enabled_modes", BASELINE_MODES)
        )
        reference_modes = tuple(
            str(value)
            for value in settings_data.pop("reference_modes", REFERENCE_MODES)
        )
        unknown = sorted(set(enabled_modes) - set(BASELINE_MODES))
        if unknown:
            raise ValueError(f"Unknown readout baseline modes: {unknown}")
        if not enabled_modes:
            raise ValueError("readout_baselines.enabled_modes must not be empty")
        settings = ReadoutBaselineSettings(
            source_readout_dir=source_path.resolve(),
            enabled_modes=enabled_modes,
            reference_modes=reference_modes,
            **settings_data,
        )
        return cls(mapping_audit=audit, settings=settings)

    @property
    def output_dir(self) -> Path:
        return self.mapping_audit.output_dir


def _fingerprint(vector: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(vector, dtype=np.float32).tobytes()
    ).hexdigest()[:16]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    keep = norms[:, 0] > 0
    if not keep.any():
        raise ValueError("Gradient subset contains no nonzero vectors")
    return values[keep] / norms[keep]


def _rescale(vector: np.ndarray, target_norm: float, *, label: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm == 0:
        raise ValueError(f"{label} direction has zero norm")
    return (values * float(target_norm) / norm).astype(np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    denominator = float(
        np.linalg.norm(left_values) * np.linalg.norm(right_values)
    )
    if denominator == 0:
        return np.nan
    return float(np.dot(left_values, right_values) / denominator)


def _load_cached_readout_arrays(
    source_model_dir: Path,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    np.ndarray,
    pd.DataFrame,
]:
    arrays_path = source_model_dir / "readout_geometry_arrays.npz"
    inventory_path = source_model_dir / "readout_gradient_inventory.csv"
    if not arrays_path.exists() or not inventory_path.exists():
        raise FileNotFoundError(
            "The simple readout baseline requires completed readout-geometry "
            f"caches under {source_model_dir}"
        )
    cached: dict[tuple[str, str], np.ndarray] = {}
    with np.load(arrays_path) as arrays:
        if "train_local_gradients" not in arrays:
            raise KeyError(f"{arrays_path} has no train_local_gradients array")
        train_gradients = np.asarray(
            arrays["train_local_gradients"], dtype=np.float32
        )
        for key in arrays.files:
            if "__" not in key:
                continue
            pair_type, mode = key.split("__", maxsplit=1)
            cached[(pair_type, mode)] = np.asarray(
                arrays[key], dtype=np.float32
            )
    inventory = pd.read_csv(inventory_path)
    train_inventory = inventory.loc[
        inventory["gradient_split"].astype(str).eq("train")
    ].reset_index(drop=True)
    if len(train_inventory) != len(train_gradients):
        raise ValueError(
            "Cached train gradients and train gradient inventory have "
            f"different lengths: {len(train_gradients)} vs "
            f"{len(train_inventory)}"
        )
    return cached, train_gradients, train_inventory


def _oriented_gradient_subset(
    gradients: np.ndarray,
    inventory: pd.DataFrame,
    *,
    source_identifier: str,
    target_identifier: str,
) -> np.ndarray:
    if source_identifier == target_identifier:
        raise ValueError("Source and target identifiers must differ")
    ordered = tuple(sorted((source_identifier, target_identifier)))
    mask = (
        inventory["left_identifier"].astype(str).eq(ordered[0])
        & inventory["right_identifier"].astype(str).eq(ordered[1])
    ).to_numpy()
    selected = np.asarray(gradients[mask], dtype=np.float64)
    if not len(selected):
        raise ValueError(
            f"No gradients found for {source_identifier}->{target_identifier}"
        )
    # Cached gradients are d(logit_left-logit_right)/dh.
    if (source_identifier, target_identifier) == ordered:
        selected = -selected
    return selected


def build_simple_readout_baseline_bank(
    *,
    cached_directions: dict[tuple[str, str], np.ndarray],
    train_gradients: np.ndarray,
    train_gradient_inventory: pd.DataFrame,
    canonical_labels: tuple[str, ...],
    identifier_token_vectors: np.ndarray,
    normalize_gradient_rows: bool,
    enabled_modes: tuple[str, ...],
    extraction_mapping: str,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
]:
    identifiers = tuple(
        chr(ord("A") + index) for index in range(len(canonical_labels))
    )
    identifier_by_label = dict(zip(canonical_labels, identifiers))
    token_vectors = np.asarray(identifier_token_vectors, dtype=np.float64)
    if token_vectors.shape[0] != len(identifiers):
        raise ValueError(
            "identifier_token_vectors must have one row per answer identifier"
        )

    directions: dict[tuple[str, str], np.ndarray] = {}
    inventory_rows = []
    cosine_rows = []
    pair_types = sorted(
        pair_type
        for pair_type, mode in cached_directions
        if mode == "raw_canonical_direction"
    )
    for pair_type in pair_types:
        labels = str(pair_type).split("_vs_", maxsplit=1)
        if len(labels) != 2 or any(
            label not in identifier_by_label for label in labels
        ):
            raise ValueError(f"Cannot parse configured contrast {pair_type!r}")
        source_label, target_label = labels
        source_identifier = identifier_by_label[source_label]
        target_identifier = identifier_by_label[target_label]
        selected_gradients = _oriented_gradient_subset(
            train_gradients,
            train_gradient_inventory,
            source_identifier=source_identifier,
            target_identifier=target_identifier,
        )
        gradient_matrix = (
            _unit_rows(selected_gradients)
            if normalize_gradient_rows
            else selected_gradients
        )
        mean_gradient = gradient_matrix.mean(axis=0)
        _, _, vh = np.linalg.svd(gradient_matrix, full_matrices=False)
        rank1_gradient = vh[0]
        if np.dot(rank1_gradient, mean_gradient) < 0:
            rank1_gradient = -rank1_gradient

        source_index = identifiers.index(source_identifier)
        target_index = identifiers.index(target_identifier)
        unembedding_difference = (
            token_vectors[target_index] - token_vectors[source_index]
        )
        raw = np.asarray(
            cached_directions[(pair_type, "raw_canonical_direction")],
            dtype=np.float32,
        )
        raw_norm = float(np.linalg.norm(raw))
        candidates = {
            "mean_local_gradient_norm_matched": _rescale(
                mean_gradient, raw_norm, label="mean local gradient"
            ),
            "rank1_local_gradient_norm_matched": _rescale(
                rank1_gradient, raw_norm, label="rank-1 local gradient"
            ),
            "unembedding_difference_norm_matched": _rescale(
                unembedding_difference,
                raw_norm,
                label="unembedding difference",
            ),
        }
        reference_vectors = {
            mode: np.asarray(vector, dtype=np.float32)
            for (cached_pair_type, mode), vector in cached_directions.items()
            if cached_pair_type == pair_type and mode in REFERENCE_MODES
        }
        for mode, vector in candidates.items():
            if mode not in enabled_modes:
                continue
            directions[(pair_type, mode)] = vector
            inventory_rows.append(
                {
                    "pair_type": pair_type,
                    "mode": mode,
                    "source_label": source_label,
                    "target_label": target_label,
                    "source_identifier": source_identifier,
                    "target_identifier": target_identifier,
                    "extraction_mapping": extraction_mapping,
                    "direction_l2": float(np.linalg.norm(vector)),
                    "raw_caa_l2": raw_norm,
                    "n_gradient_vectors": int(len(gradient_matrix)),
                    "normalize_gradient_rows": bool(normalize_gradient_rows),
                    "direction_sha256": _fingerprint(vector),
                }
            )
            for reference_mode, reference in reference_vectors.items():
                cosine_rows.append(
                    {
                        "pair_type": pair_type,
                        "baseline_mode": mode,
                        "reference_mode": reference_mode,
                        "cosine": _cosine(vector, reference),
                    }
                )
    if not directions:
        raise ValueError("No simple readout baseline directions were built")
    return (
        directions,
        pd.DataFrame(inventory_rows),
        pd.DataFrame(cosine_rows),
    )


def _identifier_token_vectors(
    model: Any,
    tokenizer: Any,
    identifiers: tuple[str, ...],
) -> np.ndarray:
    token_ids = _strict_identifier_token_ids(tokenizer, identifiers)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model does not expose an output-embedding weight")
    return (
        output_embeddings.weight[token_ids]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def _summarize_effects(pair_effects: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "semantic_prob_gain",
        "original_letter_prob_gain",
        "semantic_minus_original_prob_gain_moved",
        "js_shift",
    ]
    return (
        pair_effects.groupby(
            ["mapping_name", "mapping_family", "mode"], as_index=False
        )[metrics]
        .mean()
        .rename(columns={metric: f"mean_{metric}" for metric in metrics})
    )


def run_single_model_readout_baselines(
    model: Any,
    tokenizer: Any,
    config: ReadoutBaselineAuditConfig,
    model_config: MappingAuditModelConfig,
) -> dict[str, pd.DataFrame]:
    audit = config.mapping_audit
    alias = model_config.model.alias
    model_dir = audit.output_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    source_model_dir = config.settings.source_readout_dir / alias
    cached, train_gradients, train_inventory = _load_cached_readout_arrays(
        source_model_dir
    )
    canonical_mapping = next(
        mapping
        for mapping in audit.mappings
        if mapping.name == audit.dataset.canonical_mapping
    )
    canonical_labels = tuple(canonical_mapping.option_order)
    identifiers = tuple(
        chr(ord("A") + index) for index in range(len(canonical_labels))
    )
    directions, direction_inventory, cosine_table = (
        build_simple_readout_baseline_bank(
            cached_directions=cached,
            train_gradients=train_gradients,
            train_gradient_inventory=train_inventory,
            canonical_labels=canonical_labels,
            identifier_token_vectors=_identifier_token_vectors(
                model, tokenizer, identifiers
            ),
            normalize_gradient_rows=config.settings.normalize_gradient_rows,
            enabled_modes=config.settings.enabled_modes,
            extraction_mapping=audit.dataset.canonical_mapping,
        )
    )
    direction_inventory["layer_index"] = model_config.locked_layer

    base_items = load_base_items(audit.dataset, seed=audit.runtime.seed)
    all_eval_rows = []
    all_pair_effects = []
    all_ci = []
    for mapping in audit.mappings:
        mapping_items = build_mapping_items(
            base_items, audit.dataset, mapping, canonical_mapping
        )
        eval_items = build_mapping_eval_items(
            mapping_items, audit.dataset, mapping, canonical_mapping
        )
        eval_rows = evaluate_fixed_directions(
            model,
            tokenizer,
            eval_items,
            directions,
            direction_inventory,
            layer_index=model_config.locked_layer,
            alpha=model_config.locked_alpha,
            n_choices=len(audit.dataset.label_ranks),
            batch_size=audit.runtime.batch_size,
            max_length=audit.runtime.max_length,
            injection_position="pre_answer",
        )
        pair_effects = build_pair_effects(eval_rows)
        all_eval_rows.append(eval_rows)
        all_pair_effects.append(pair_effects)
        all_ci.append(
            build_mapping_mode_ci(
                pair_effects,
                n_boot=audit.statistics.n_boot,
                confidence=audit.statistics.confidence,
                seed=audit.runtime.seed,
            )
        )

    pair_effects = pd.concat(all_pair_effects, ignore_index=True, sort=False)
    tables = {
        "readout_baseline_direction_inventory": direction_inventory,
        "readout_baseline_cosines": cosine_table,
        "readout_baseline_eval_rows": pd.concat(
            all_eval_rows, ignore_index=True, sort=False
        ),
        "readout_baseline_pair_effects": pair_effects,
        "readout_baseline_mapping_mode_ci": pd.concat(
            all_ci, ignore_index=True, sort=False
        ),
        "readout_baseline_model_summary": _summarize_effects(pair_effects),
    }
    tables = {
        name: annotate_model_frame(
            frame,
            model_alias=alias,
            model_name=model_config.model.name,
        )
        for name, frame in tables.items()
    }
    write_tables(tables, model_dir)
    pd.DataFrame(
        [
            {
                "model_alias": alias,
                "model_name": model_config.model.name,
                "status": "complete",
                "locked_layer": model_config.locked_layer,
                "locked_alpha": model_config.locked_alpha,
                "source_readout_dir": str(source_model_dir),
                "n_baseline_modes": len(config.settings.enabled_modes),
                "n_eval_mappings": len(audit.mappings),
            }
        ]
    ).to_csv(model_dir / "readout_baseline_run_complete.csv", index=False)
    return tables


def readout_baseline_run_is_complete(model_dir: str | Path) -> bool:
    root = Path(model_dir)
    return all(
        (root / filename).exists()
        for filename in [
            "readout_baseline_run_complete.csv",
            "readout_baseline_direction_inventory.csv",
            "readout_baseline_pair_effects.csv",
            "readout_baseline_mapping_mode_ci.csv",
        ]
    )


def _collect_model_tables(
    config: ReadoutBaselineAuditConfig,
) -> dict[str, pd.DataFrame]:
    names = [
        "readout_baseline_run_complete",
        "readout_baseline_direction_inventory",
        "readout_baseline_cosines",
        "readout_baseline_model_summary",
        "readout_baseline_mapping_mode_ci",
        "readout_baseline_pair_effects",
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


def _load_reference_effects(
    config: ReadoutBaselineAuditConfig,
) -> pd.DataFrame:
    parts = []
    for model_config in config.mapping_audit.models:
        path = (
            config.settings.source_readout_dir
            / model_config.model.alias
            / "readout_pair_effects.csv"
        )
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        parts.append(
            frame.loc[
                frame["mode"].astype(str).isin(
                    config.settings.reference_modes
                )
                & frame.get(
                    "evaluation_scope",
                    pd.Series("full_test", index=frame.index),
                )
                .astype(str)
                .eq("full_test")
            ].copy()
        )
    return (
        pd.concat(parts, ignore_index=True, sort=False)
        if parts
        else pd.DataFrame()
    )


def aggregate_readout_baselines(
    config: ReadoutBaselineAuditConfig,
) -> dict[str, pd.DataFrame]:
    aggregated = _collect_model_tables(config)
    baseline_effects = aggregated.get(
        "readout_baseline_pair_effects", pd.DataFrame()
    )
    reference_effects = _load_reference_effects(config)
    combined_effects = pd.concat(
        [reference_effects, baseline_effects],
        ignore_index=True,
        sort=False,
    )
    aggregated["readout_baseline_combined_pair_effects"] = combined_effects
    if combined_effects.empty:
        aggregated["readout_baseline_global_summary"] = pd.DataFrame()
    else:
        metrics = [
            "semantic_prob_gain",
            "original_letter_prob_gain",
            "semantic_minus_original_prob_gain_moved",
            "js_shift",
        ]
        strata = (
            combined_effects.groupby(
                [
                    "model_alias",
                    "pair_type",
                    "mapping_name",
                    "mapping_family",
                    "mode",
                ],
                as_index=False,
            )[metrics]
            .mean()
        )
        aggregated["readout_baseline_global_summary"] = (
            strata.groupby(
                ["mapping_name", "mapping_family", "mode"], as_index=False
            )
            .agg(
                n_models=("model_alias", "nunique"),
                n_contrasts=("pair_type", "nunique"),
                mean_current_label_gain=("semantic_prob_gain", "mean"),
                mean_extraction_identifier_gain=(
                    "original_letter_prob_gain",
                    "mean",
                ),
                mean_current_minus_identifier=(
                    "semantic_minus_original_prob_gain_moved",
                    "mean",
                ),
                mean_js_shift=("js_shift", "mean"),
            )
        )
    write_tables(aggregated, config.output_dir)
    return aggregated


def run_readout_baseline_audit(
    config: ReadoutBaselineAuditConfig,
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
        return aggregate_readout_baselines(config)
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
        model_dir = audit.output_dir / public_model.alias
        if (
            readout_baseline_run_is_complete(model_dir)
            and not audit.runtime.force_rerun
        ):
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
            run_single_model_readout_baselines(
                model, tokenizer, config, model_config
            )
            (model_dir / "readout_baseline_run_error.csv").unlink(
                missing_ok=True
            )
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
            pd.DataFrame([error]).to_csv(
                model_dir / "readout_baseline_run_error.csv", index=False
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
            audit.output_dir / "readout_baseline_errors.csv", index=False
        )
    else:
        (audit.output_dir / "readout_baseline_errors.csv").unlink(
            missing_ok=True
        )
    if phase == "run":
        return _collect_model_tables(config)
    return aggregate_readout_baselines(config)


def run_readout_baseline_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    config = ReadoutBaselineAuditConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    return run_readout_baseline_audit(
        config,
        model_aliases=model_aliases,
        phase=phase,
    )
