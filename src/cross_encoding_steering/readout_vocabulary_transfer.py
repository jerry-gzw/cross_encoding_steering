"""Cross-vocabulary transfer for a frozen A/B/C local-readout basis.

The source readout audit estimates its basis from A/B/C identifier-logit
gradients. This module reuses the saved raw/projection/residual directions and
evaluates them under A/B/C, X/Y/Z, and 1/2/3 identifiers without re-estimating
the basis, rank, layer, position, or dose.
"""
from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .central_group_cluster_inference import _cluster_bootstrap_balanced_means
from .interface_factorial_audit import (
    InterfaceFactorialConfig,
    _evaluate_condition,
    _evaluate_key_competence,
    _factorial_conditions,
    _summarize_competence,
    _summarize_factorial,
    _cross_interface_view,
)
from .io import write_tables
from .mapping_audit import load_base_items
from .cross_interface_audit import _build_eval_items
from .steering import cleanup_model, load_tokenizer_and_model


CORE_MODES = (
    "raw_canonical_direction",
    "readout_projection_natural",
    "readout_orthogonal_natural",
)
MODE_LABELS = {
    "raw_canonical_direction": "raw",
    "readout_projection_natural": "projection",
    "readout_orthogonal_natural": "residual",
}


@dataclass(frozen=True)
class ReadoutVocabularyTransferConfig:
    factorial: InterfaceFactorialConfig
    source_readout_dir: Path
    source_basis_identifier_set: str
    enabled_modes: tuple[str, ...]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "ReadoutVocabularyTransferConfig":
        config_path = Path(path).expanduser().resolve()
        factorial = InterfaceFactorialConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        # InterfaceFactorialConfig already resolves the recursive base config.
        from .mapping_audit import _load_mapping_audit_config

        data = _load_mapping_audit_config(config_path)
        spec = dict(data.get("readout_vocabulary_transfer", {}))
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else config_path.parent.parent.resolve()
        )
        source = Path(
            spec.get(
                "source_readout_dir",
                "outputs/group_disjoint_normbank/"
                "readout_geometry_audit_v2_full",
            )
        ).expanduser()
        if not source.is_absolute():
            source = root / source
        enabled_modes = tuple(str(value) for value in spec.get("enabled_modes", CORE_MODES))
        if set(enabled_modes) != set(CORE_MODES):
            raise ValueError(
                "readout_vocabulary_transfer.enabled_modes must contain exactly "
                f"{list(CORE_MODES)}"
            )
        identifier_names = {item.name for item in factorial.identifier_sets}
        required_identifiers = {"letters_abc", "letters_xyz", "numbers_123"}
        missing_identifiers = sorted(required_identifiers - identifier_names)
        if missing_identifiers:
            raise ValueError(
                "Cross-vocabulary transfer requires identifier sets "
                f"{sorted(required_identifiers)}; missing {missing_identifiers}"
            )
        if len(factorial.row_orders) != 1 or factorial.row_orders[0].name != "order_123":
            raise ValueError(
                "Cross-vocabulary transfer must hold row position fixed at order_123"
            )
        if not source.exists():
            raise FileNotFoundError(f"Missing source readout directory: {source}")
        return cls(
            factorial=factorial,
            source_readout_dir=source.resolve(),
            source_basis_identifier_set=str(
                spec.get("source_basis_identifier_set", "letters_abc")
            ),
            enabled_modes=enabled_modes,
        )

    @property
    def output_dir(self) -> Path:
        return self.factorial.audit.output_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_directions(
    config: ReadoutVocabularyTransferConfig,
    model_config: Any,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame, str]:
    alias = model_config.model.alias
    model_dir = config.source_readout_dir / alias
    arrays_path = model_dir / "readout_geometry_arrays.npz"
    completion_path = model_dir / "readout_geometry_run_complete.csv"
    inventory_path = model_dir / "readout_direction_inventory.csv"
    missing = [
        str(path)
        for path in (arrays_path, completion_path, inventory_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing source readout outputs: " + ", ".join(missing))

    completion = pd.read_csv(completion_path)
    if set(completion["locked_layer"].astype(int)) != {int(model_config.locked_layer)}:
        raise ValueError(
            f"Source layer mismatch for {alias}: source="
            f"{sorted(completion['locked_layer'].astype(int).unique())}, "
            f"configured={model_config.locked_layer}"
        )
    if not np.allclose(
        completion["locked_alpha"].astype(float).to_numpy(),
        float(model_config.locked_alpha),
    ):
        raise ValueError(
            f"Source alpha mismatch for {alias}: configured={model_config.locked_alpha}"
        )

    directions: dict[tuple[str, str], np.ndarray] = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if "readout_basis" not in arrays.files:
            raise ValueError(f"Source arrays lack readout_basis: {arrays_path}")
        for key in arrays.files:
            if "__" not in key:
                continue
            pair_type, mode = key.split("__", 1)
            if mode in config.enabled_modes:
                directions[(pair_type, mode)] = np.asarray(arrays[key], dtype=np.float32)
    observed_modes = {mode for _, mode in directions}
    if observed_modes != set(config.enabled_modes):
        raise ValueError(
            f"Source direction modes for {alias} are {sorted(observed_modes)}, "
            f"expected {sorted(config.enabled_modes)}"
        )
    # Pair types are defined by ordered label pairs, not by individual labels.
    observed_contrasts = {pair_type for pair_type, _ in directions}
    if len(observed_contrasts) < 2 or not observed_contrasts:
        raise ValueError(f"Source direction bank for {alias} has too few contrasts")

    inventory = pd.read_csv(inventory_path)
    inventory = inventory[inventory["mode"].isin(config.enabled_modes)].copy()
    inventory["source_basis_identifier_set"] = config.source_basis_identifier_set
    inventory["source_arrays_path"] = str(arrays_path)
    source_hash = _sha256(arrays_path)
    inventory["source_arrays_sha256"] = source_hash
    inventory["transfer_protocol"] = "frozen_abc_basis_cross_vocabulary"
    return directions, inventory, source_hash


def _model_run_complete(
    model_dir: Path,
    *,
    source_hash: str,
    enabled_modes: tuple[str, ...],
) -> bool:
    path = model_dir / "readout_vocabulary_transfer_run_complete.csv"
    if not path.exists():
        return False
    completion = pd.read_csv(path)
    return (
        set(completion.get("source_arrays_sha256", pd.Series(dtype=str)).astype(str))
        == {source_hash}
        and set(completion.get("enabled_modes", pd.Series(dtype=str)).astype(str))
        == {",".join(enabled_modes)}
        and set(completion.get("status", pd.Series(dtype=str)).astype(str))
        == {"complete"}
    )


def _add_model_columns(frame: pd.DataFrame, model_config: Any) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "model_alias", model_config.model.alias)
    output.insert(1, "model_name", model_config.model.name)
    return output


def run_single_model_readout_vocabulary_transfer(
    model: Any,
    tokenizer: Any,
    config: ReadoutVocabularyTransferConfig,
    model_config: Any,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    factorial = config.factorial
    audit = factorial.audit
    alias = model_config.model.alias
    model_dir = config.output_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    directions, inventory, source_hash = _load_source_directions(config, model_config)

    base_items = load_base_items(audit.dataset, seed=audit.runtime.seed)
    eval_items = _build_eval_items(base_items, _cross_interface_view(factorial))
    conditions = _factorial_conditions(factorial)
    result_frames: list[pd.DataFrame] = []
    for condition_index, condition in enumerate(conditions, start=1):
        progress(f"[{alias}] vocabulary condition {condition_index}/{len(conditions)}: {condition.name}")
        for template in factorial.templates:
            condition_dir = model_dir / "conditions" / condition.name / template.name
            complete_path = condition_dir / "condition_complete.csv"
            baseline_path = condition_dir / "baseline_rows.csv"
            steering_path = condition_dir / "steering_rows.csv"
            if (
                complete_path.exists()
                and baseline_path.exists()
                and steering_path.exists()
                and not audit.runtime.force_rerun
            ):
                cached = pd.read_csv(steering_path)
                completion = pd.read_csv(complete_path)
                cached_modes = set(cached["mode"].dropna().astype(str).unique())
                cached_hash = set(completion["source_arrays_sha256"].astype(str))
                if cached_modes != set(config.enabled_modes) or cached_hash != {source_hash}:
                    raise RuntimeError(
                        f"Incompatible cache at {condition_dir}; use a fresh output directory"
                    )
                result_frames.extend([pd.read_csv(baseline_path), cached])
                continue

            rows = _evaluate_condition(
                model,
                tokenizer,
                base_items,
                eval_items,
                directions,
                factorial,
                model_config,
                template,
                condition,
            )
            baseline = rows[rows["record_type"].eq("baseline")].dropna(axis=1, how="all")
            steering = rows[rows["record_type"].eq("steering")].dropna(axis=1, how="all")
            write_tables(
                {
                    "baseline_rows": baseline,
                    "steering_rows": steering,
                    "condition_complete": pd.DataFrame(
                        [
                            {
                                "condition": condition.name,
                                "template": template.name,
                                "status": "complete",
                                "source_arrays_sha256": source_hash,
                                "source_basis_identifier_set": config.source_basis_identifier_set,
                                "enabled_modes": ",".join(config.enabled_modes),
                            }
                        ]
                    ),
                },
                condition_dir,
            )
            result_frames.extend([baseline, steering])

    combined = pd.concat(result_frames, ignore_index=True, sort=False)
    baseline = combined[combined["record_type"].eq("baseline")].dropna(axis=1, how="all")
    steering = combined[combined["record_type"].eq("steering")].dropna(axis=1, how="all")
    key_rows = _evaluate_key_competence(model, tokenizer, factorial)
    tables = {
        "baseline_competence_rows": _add_model_columns(baseline, model_config),
        "readout_vocabulary_transfer_rows": _add_model_columns(steering, model_config),
        "mapping_key_competence_rows": _add_model_columns(key_rows, model_config),
        "readout_vocabulary_transfer_direction_inventory": _add_model_columns(
            inventory.drop(columns=[column for column in ("model_alias", "model_name") if column in inventory]),
            model_config,
        ),
    }
    write_tables(tables, model_dir)
    completion = pd.DataFrame(
        [
            {
                "model_alias": alias,
                "model_name": model_config.model.name,
                "status": "complete",
                "n_conditions": len(conditions),
                "n_identifier_sets": len(factorial.identifier_sets),
                "n_semantic_mappings": len(factorial.mapping_names),
                "locked_layer": model_config.locked_layer,
                "locked_alpha": model_config.locked_alpha,
                "source_arrays_sha256": source_hash,
                "source_basis_identifier_set": config.source_basis_identifier_set,
                "enabled_modes": ",".join(config.enabled_modes),
            }
        ]
    )
    completion.to_csv(
        model_dir / "readout_vocabulary_transfer_run_complete.csv", index=False
    )
    tables["readout_vocabulary_transfer_run_complete"] = completion
    return tables


def _concat_model_tables(
    config: ReadoutVocabularyTransferConfig,
    filename: str,
) -> pd.DataFrame:
    frames = []
    for model_config in config.factorial.audit.models:
        path = config.output_dir / model_config.model.alias / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _competence_tables(
    config: ReadoutVocabularyTransferConfig,
    baseline: pd.DataFrame,
    key_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = list(config.factorial.audit.dataset.label_ranks)
    baseline_summary = _summarize_competence(baseline, labels)
    key_summary = (
        key_rows.groupby(
            ["model_alias", "model_name", "semantic_mapping", "identifier_set"],
            as_index=False,
        )
        .agg(
            n_items=("correct", "size"),
            accuracy=("correct", "mean"),
            min_target_margin=("target_margin", "min"),
        )
    )
    key_summary["threshold"] = config.factorial.competence_threshold
    key_summary["competence_passed"] = (
        key_summary["accuracy"].ge(config.factorial.competence_threshold)
        & key_summary["min_target_margin"].gt(0.0)
    )
    n_vocabularies = len(config.factorial.identifier_sets)
    common = (
        key_summary.groupby(["model_alias", "model_name", "semantic_mapping"], as_index=False)
        .agg(
            n_vocabularies=("identifier_set", "nunique"),
            n_vocabularies_passed=("competence_passed", "sum"),
        )
    )
    common["common_competence_passed"] = (
        common["n_vocabularies"].eq(n_vocabularies)
        & common["n_vocabularies_passed"].eq(n_vocabularies)
    )
    return baseline_summary, key_summary, common


def _wide_transfer_units(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_alias",
        "model_name",
        "pair_type",
        "pair_id",
        "group_key",
        "semantic_mapping",
        "identifier_set",
    ]
    metrics = [
        "semantic_margin_gain",
        "extraction_identifier_margin_gain",
        "extraction_row_margin_gain",
    ]
    collapsed = frame.groupby([*keys, "mode"], as_index=False)[metrics].mean()
    wide: pd.DataFrame | None = None
    for mode, short in MODE_LABELS.items():
        part = collapsed[collapsed["mode"].eq(mode)][[*keys, *metrics]].copy()
        part = part.rename(columns={metric: f"{short}_{metric}" for metric in metrics})
        wide = part if wide is None else wide.merge(part, on=keys, how="inner", validate="one_to_one")
    if wide is None or wide.empty:
        return pd.DataFrame()
    for short in MODE_LABELS.values():
        wide[f"{short}_id_advantage"] = (
            wide[f"{short}_extraction_identifier_margin_gain"]
            - wide[f"{short}_semantic_margin_gain"]
        )
    wide["projection_minus_residual_id"] = (
        wide["projection_extraction_identifier_margin_gain"]
        - wide["residual_extraction_identifier_margin_gain"]
    )
    return wide


def _scope_frames(
    pair_effects: pd.DataFrame,
    key_summary: pd.DataFrame,
    common: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    gate = key_summary[
        ["model_alias", "semantic_mapping", "identifier_set", "competence_passed"]
    ]
    annotated = pair_effects.merge(
        gate,
        on=["model_alias", "semantic_mapping", "identifier_set"],
        how="left",
        validate="many_to_one",
    )
    annotated = annotated.merge(
        common[["model_alias", "semantic_mapping", "common_competence_passed"]],
        on=["model_alias", "semantic_mapping"],
        how="left",
        validate="many_to_one",
    )
    if annotated[["competence_passed", "common_competence_passed"]].isna().any().any():
        raise ValueError("Some transfer cells lack vocabulary-competence records")
    return {
        "all_cells": annotated,
        "vocabulary_competence": annotated[annotated["competence_passed"].astype(bool)].copy(),
        "common_competence": annotated[
            annotated["common_competence_passed"].astype(bool)
        ].copy(),
    }


def build_readout_vocabulary_transfer_statistics(
    config: ReadoutVocabularyTransferConfig,
    pair_effects: pd.DataFrame,
    key_summary: pd.DataFrame,
    common: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    canonical = config.factorial.audit.dataset.canonical_mapping
    n_boot = int(config.factorial.audit.statistics.n_boot)
    confidence = float(config.factorial.audit.statistics.confidence)
    rng = np.random.default_rng(config.factorial.audit.runtime.seed + 911)
    component_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    vocabulary_contrast_rows: list[dict[str, Any]] = []
    by_model_rows: list[dict[str, Any]] = []
    unit_frames: list[pd.DataFrame] = []

    for scope_name, scoped in _scope_frames(pair_effects, key_summary, common).items():
        non_source = scoped[~scoped["semantic_mapping"].eq(canonical)].copy()
        all_units = _wide_transfer_units(non_source)
        if not all_units.empty:
            index = [
                "model_alias",
                "model_name",
                "pair_type",
                "pair_id",
                "group_key",
                "semantic_mapping",
            ]
            compared_metrics = [
                f"{component}_{metric}"
                for component in ("raw", "projection", "residual")
                for metric in (
                    "extraction_identifier_margin_gain",
                    "id_advantage",
                )
            ]
            vocabulary_wide = all_units.pivot_table(
                index=index,
                columns="identifier_set",
                values=compared_metrics,
                aggfunc="mean",
            ).reset_index()
            vocabulary_wide.columns = [
                "__".join(str(value) for value in column if str(value))
                if isinstance(column, tuple)
                else str(column)
                for column in vocabulary_wide.columns
            ]
            for alternative in ("letters_xyz", "numbers_123"):
                required = []
                values = []
                for metric in compared_metrics:
                    reference_column = f"{metric}__letters_abc"
                    alternative_column = f"{metric}__{alternative}"
                    required.extend([reference_column, alternative_column])
                    difference_column = f"{metric}__{alternative}_minus_letters_abc"
                    vocabulary_wide[difference_column] = (
                        vocabulary_wide[alternative_column]
                        - vocabulary_wide[reference_column]
                    )
                    values.append(difference_column)
                paired = vocabulary_wide.dropna(subset=required).copy()
                if paired.empty:
                    continue
                result = _cluster_bootstrap_balanced_means(
                    paired,
                    strata=("model_alias", "pair_type", "semantic_mapping"),
                    values=tuple(values),
                    n_boot=n_boot,
                    confidence=confidence,
                    rng=rng,
                )
                for metric in compared_metrics:
                    difference_column = f"{metric}__{alternative}_minus_letters_abc"
                    low, high = result["interval"][difference_column]
                    component, metric_name = metric.split("_", 1)
                    vocabulary_contrast_rows.append(
                        {
                            "scope": scope_name,
                            "alternative_identifier_set": alternative,
                            "reference_identifier_set": "letters_abc",
                            "component": component,
                            "metric": metric_name,
                            "comparison": "alternative_minus_reference",
                            "estimate": result["point"][difference_column],
                            "ci_low": low,
                            "ci_high": high,
                            "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                            "n_groups": result["n_groups"],
                            "n_unique_pairs": result["n_pairs"],
                            "n_model_contrast_mapping_strata": result["n_strata"],
                        }
                    )
        for identifier_set, vocabulary_rows in non_source.groupby("identifier_set", sort=True):
            units = _wide_transfer_units(vocabulary_rows)
            if units.empty:
                continue
            units["scope"] = scope_name
            unit_frames.append(units)
            values = (
                "raw_extraction_identifier_margin_gain",
                "projection_extraction_identifier_margin_gain",
                "residual_extraction_identifier_margin_gain",
                "raw_semantic_margin_gain",
                "projection_semantic_margin_gain",
                "residual_semantic_margin_gain",
                "raw_id_advantage",
                "projection_id_advantage",
                "residual_id_advantage",
                "projection_minus_residual_id",
            )
            result = _cluster_bootstrap_balanced_means(
                units,
                strata=("model_alias", "pair_type", "semantic_mapping"),
                values=values,
                n_boot=n_boot,
                confidence=confidence,
                rng=rng,
            )
            metric_map = {
                "extraction_identifier_effect": "extraction_identifier_margin_gain",
                "current_label_effect": "semantic_margin_gain",
                "id_advantage": "id_advantage",
            }
            for component in ("raw", "projection", "residual"):
                for metric_label, suffix in metric_map.items():
                    value = f"{component}_{suffix}"
                    low, high = result["interval"][value]
                    component_rows.append(
                        {
                            "scope": scope_name,
                            "identifier_set": identifier_set,
                            "component": component,
                            "metric": metric_label,
                            "estimate": result["point"][value],
                            "ci_low": low,
                            "ci_high": high,
                            "n_groups": result["n_groups"],
                            "n_unique_pairs": result["n_pairs"],
                            "n_model_contrast_mapping_strata": result["n_strata"],
                        }
                    )
            for component in ("projection", "residual"):
                numerator = f"{component}_extraction_identifier_margin_gain"
                denominator = "raw_extraction_identifier_margin_gain"
                denominator_point = result["point"][denominator]
                point = (
                    result["point"][numerator] / denominator_point
                    if abs(denominator_point) > 1e-12
                    else np.nan
                )
                boot = np.divide(
                    result["boot"][numerator],
                    result["boot"][denominator],
                    out=np.full(n_boot, np.nan),
                    where=np.abs(result["boot"][denominator]) > 1e-12,
                )
                finite = boot[np.isfinite(boot)]
                tail = (1.0 - confidence) / 2.0
                retention_rows.append(
                    {
                        "scope": scope_name,
                        "identifier_set": identifier_set,
                        "comparison": f"{component}_retention_ratio",
                        "statistic": "ratio_of_balanced_means",
                        "estimate": point,
                        "ci_low": (
                            float(np.quantile(finite, tail)) if len(finite) else np.nan
                        ),
                        "ci_high": (
                            float(np.quantile(finite, 1.0 - tail))
                            if len(finite)
                            else np.nan
                        ),
                        "n_groups": result["n_groups"],
                        "n_unique_pairs": result["n_pairs"],
                        "n_model_contrast_mapping_strata": result["n_strata"],
                    }
                )
            difference = "projection_minus_residual_id"
            low, high = result["interval"][difference]
            retention_rows.append(
                {
                    "scope": scope_name,
                    "identifier_set": identifier_set,
                    "comparison": "projection_minus_residual_id",
                    "statistic": "paired_difference",
                    "estimate": result["point"][difference],
                    "ci_low": low,
                    "ci_high": high,
                    "n_groups": result["n_groups"],
                    "n_unique_pairs": result["n_pairs"],
                    "n_model_contrast_mapping_strata": result["n_strata"],
                }
            )

            for model_alias, model_rows in units.groupby("model_alias", sort=True):
                model_name = str(model_rows["model_name"].iloc[0])
                for component in ("raw", "projection", "residual"):
                    by_model_rows.append(
                        {
                            "scope": scope_name,
                            "identifier_set": identifier_set,
                            "model_alias": model_alias,
                            "model_name": model_name,
                            "component": component,
                            "mean_extraction_identifier_effect": float(
                                model_rows[f"{component}_extraction_identifier_margin_gain"].mean()
                            ),
                            "mean_current_label_effect": float(
                                model_rows[f"{component}_semantic_margin_gain"].mean()
                            ),
                            "mean_id_advantage": float(
                                model_rows[f"{component}_id_advantage"].mean()
                            ),
                            "n_pairs": int(model_rows["pair_id"].nunique()),
                            "n_mappings": int(model_rows["semantic_mapping"].nunique()),
                        }
                    )

    component = pd.DataFrame(component_rows)
    retention = pd.DataFrame(retention_rows)
    if not component.empty:
        component["confidence"] = confidence
        component["n_boot"] = n_boot
        component["inference_unit"] = (
            "setting_behavior_group; jointly resampled across fixed "
            "model-contrast-mapping strata"
        )
    if not retention.empty:
        retention["confidence"] = confidence
        retention["n_boot"] = n_boot
        retention["ci_excludes_zero"] = (
            retention["ci_low"].gt(0.0) | retention["ci_high"].lt(0.0)
        )
        retention["inference_unit"] = (
            "setting_behavior_group; jointly resampled across fixed "
            "model-contrast-mapping strata"
        )
    return {
        "readout_vocabulary_transfer_units": (
            pd.concat(unit_frames, ignore_index=True, sort=False)
            if unit_frames
            else pd.DataFrame()
        ),
        "readout_vocabulary_transfer_component_ci": component,
        "readout_vocabulary_transfer_retention_ci": retention,
        "readout_vocabulary_transfer_vocabulary_contrast_ci": pd.DataFrame(
            vocabulary_contrast_rows
        ),
        "readout_vocabulary_transfer_by_model": pd.DataFrame(by_model_rows),
    }


def aggregate_readout_vocabulary_transfer(
    config: ReadoutVocabularyTransferConfig,
) -> dict[str, pd.DataFrame]:
    baseline = _concat_model_tables(config, "baseline_competence_rows.csv")
    steering = _concat_model_tables(config, "readout_vocabulary_transfer_rows.csv")
    key_rows = _concat_model_tables(config, "mapping_key_competence_rows.csv")
    inventory = _concat_model_tables(
        config, "readout_vocabulary_transfer_direction_inventory.csv"
    )
    completion = _concat_model_tables(
        config, "readout_vocabulary_transfer_run_complete.csv"
    )
    if steering.empty or key_rows.empty:
        raise FileNotFoundError(
            "No completed cross-vocabulary readout-transfer model outputs were found"
        )
    observed_modes = set(steering["mode"].dropna().astype(str).unique())
    if observed_modes != set(config.enabled_modes):
        raise RuntimeError(
            f"Aggregated transfer modes are {sorted(observed_modes)}, expected "
            f"{sorted(config.enabled_modes)}"
        )
    summarized = _summarize_factorial(steering)
    pair_effects = summarized["factorial_pair_effects"]
    base_items = load_base_items(
        config.factorial.audit.dataset,
        seed=config.factorial.audit.runtime.seed,
    )
    pair_groups = base_items[["pair_id", "group_key"]].drop_duplicates()
    if pair_groups["pair_id"].duplicated().any():
        raise ValueError("A pair_id maps to multiple setting-behavior groups")
    pair_effects = pair_effects.merge(
        pair_groups,
        on="pair_id",
        how="left",
        validate="many_to_one",
    )
    if pair_effects["group_key"].isna().any():
        raise ValueError("Some transfer pairs lack setting-behavior group metadata")
    summarized["factorial_pair_effects"] = pair_effects
    baseline_summary, key_summary, common = _competence_tables(
        config, baseline, key_rows
    )
    statistics = build_readout_vocabulary_transfer_statistics(
        config, pair_effects, key_summary, common
    )
    manifest = pd.DataFrame(
        [
            {
                "status": "complete",
                "source_basis_identifier_set": config.source_basis_identifier_set,
                "n_models": int(steering["model_alias"].nunique()),
                "n_identifier_sets": int(steering["identifier_set"].nunique()),
                "n_semantic_mappings": int(steering["semantic_mapping"].nunique()),
                "n_modes": int(steering["mode"].nunique()),
                "n_pair_effect_rows": len(pair_effects),
                "primary_scope": "common_competence",
                "source_readout_dir": str(config.source_readout_dir),
                "output_dir": str(config.output_dir),
            }
        ]
    )
    tables = {
        "readout_vocabulary_transfer_manifest": manifest,
        "readout_vocabulary_transfer_run_complete": completion,
        "readout_vocabulary_transfer_rows": steering,
        "readout_vocabulary_transfer_direction_inventory": inventory,
        "readout_vocabulary_transfer_baseline_summary": baseline_summary,
        "readout_vocabulary_transfer_key_competence": key_summary,
        "readout_vocabulary_transfer_common_competence": common,
        **summarized,
        **statistics,
    }
    write_tables(tables, config.output_dir)
    return tables


def run_readout_vocabulary_transfer(
    config: ReadoutVocabularyTransferConfig,
    *,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
    model_loader: Callable[..., tuple[Any, Any]] = load_tokenizer_and_model,
    progress: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    if phase not in {"run", "aggregate", "all"}:
        raise ValueError("phase must be run, aggregate, or all")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if phase == "aggregate":
        return aggregate_readout_vocabulary_transfer(config)
    requested = set(
        model_aliases
        or [item.model.alias for item in config.factorial.audit.models if item.model.enabled]
    )
    errors: list[dict[str, Any]] = []
    for model_config in config.factorial.audit.models:
        public_model = model_config.model
        if not public_model.enabled or public_model.alias not in requested:
            continue
        model_dir = config.output_dir / public_model.alias
        source_error: Exception | None = None
        try:
            _, _, source_hash = _load_source_directions(config, model_config)
            if (
                _model_run_complete(
                    model_dir,
                    source_hash=source_hash,
                    enabled_modes=config.enabled_modes,
                )
                and not config.factorial.audit.runtime.force_rerun
            ):
                progress(f"[{public_model.alias}] complete; skipping")
                continue
        except Exception as exc:
            source_error = exc
        tokenizer = model = None
        try:
            if source_error is not None:
                raise source_error
            progress(f"[{public_model.alias}] loading {public_model.load_source}")
            tokenizer, model = model_loader(
                public_model.load_source,
                device_map=public_model.device_map,
                torch_dtype=public_model.torch_dtype,
            )
            run_single_model_readout_vocabulary_transfer(
                model,
                tokenizer,
                config,
                model_config,
                progress=progress,
            )
            (model_dir / "readout_vocabulary_transfer_error.csv").unlink(missing_ok=True)
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
                model_dir / "readout_vocabulary_transfer_error.csv", index=False
            )
            progress(f"[{public_model.alias}] failed: {type(exc).__name__}: {exc}")
            if not config.factorial.audit.runtime.continue_on_error:
                raise
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            cleanup_model()
    if errors:
        pd.DataFrame(errors).to_csv(
            config.output_dir / "readout_vocabulary_transfer_errors.csv", index=False
        )
    elif (config.output_dir / "readout_vocabulary_transfer_errors.csv").exists():
        (config.output_dir / "readout_vocabulary_transfer_errors.csv").unlink()
    if phase == "run":
        return {
            "readout_vocabulary_transfer_run_complete": _concat_model_tables(
                config, "readout_vocabulary_transfer_run_complete.csv"
            )
        }
    return aggregate_readout_vocabulary_transfer(config)


def run_readout_vocabulary_transfer_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    config = ReadoutVocabularyTransferConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    return run_readout_vocabulary_transfer(
        config,
        model_aliases=model_aliases,
        phase=phase,
    )
