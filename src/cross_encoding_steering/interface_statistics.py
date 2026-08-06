"""Paired statistical synthesis for interface and context controls.

This module is intentionally CPU-only. Prompt templates are repeated
measurements of a matched pair and are averaged before inference. Global
estimates then give equal weight to each model-by-contrast stratum.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io import write_tables


@dataclass(frozen=True)
class InterfaceStatisticsConfig:
    project_root: Path
    cross_interface_pair_effects: Path
    nuisance_pair_effects: Path
    context_pair_rows: Path
    output_dir: Path
    n_boot: int
    confidence: float
    bootstrap_chunk_size: int
    seed: int

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> "InterfaceStatisticsConfig":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        statistics = dict(data.get("statistics", {}))
        n_boot = int(statistics.get("n_boot", 10_000))
        confidence = float(statistics.get("confidence", 0.95))
        chunk_size = int(statistics.get("bootstrap_chunk_size", 500))
        if n_boot < 100:
            raise ValueError("statistics.n_boot must be at least 100")
        if not 0.0 < confidence < 1.0:
            raise ValueError("statistics.confidence must be between zero and one")
        if chunk_size < 1:
            raise ValueError("statistics.bootstrap_chunk_size must be positive")
        return cls(
            project_root=root,
            cross_interface_pair_effects=resolve(str(data.get(
                "cross_interface_pair_effects",
                "outputs/cross_interface_steering/normbank/cross_interface_pair_effects.csv",
            ))),
            nuisance_pair_effects=resolve(str(data["nuisance_pair_effects"])),
            context_pair_rows=resolve(str(data["context_pair_rows"])),
            output_dir=resolve(str(data.get(
                "output_dir",
                "outputs/interface_statistical_synthesis",
            ))),
            n_boot=n_boot,
            confidence=confidence,
            bootstrap_chunk_size=chunk_size,
            seed=int(statistics.get("seed", 13)),
        )


NUISANCE_MODE_COMPARISONS = (
    ("raw_pre_answer", "random_direction_control", "raw_minus_random"),
    ("raw_pre_answer", "wrong_direction_control", "raw_minus_wrong"),
    ("raw_scenario_end", "random_direction_control", "scenario_end_minus_random"),
    ("raw_pre_answer", "raw_scenario_end", "pre_answer_minus_scenario_end"),
    ("mapping_balanced_direction", "random_direction_control", "mapping_balanced_minus_random"),
    ("mapping_sensitive_residual", "random_direction_control", "mapping_sensitive_minus_random"),
    ("label_only_direction", "random_direction_control", "label_only_minus_random"),
    ("slot_layout_direction", "random_direction_control", "slot_layout_minus_random"),
    ("raw_pre_answer", "mapping_balanced_direction", "raw_minus_mapping_balanced"),
    ("raw_pre_answer", "mapping_sensitive_residual", "raw_minus_mapping_sensitive"),
    ("mapping_sensitive_residual", "mapping_balanced_direction", "mapping_sensitive_minus_balanced"),
)

CONTEXT_MODE_COMPARISONS = NUISANCE_MODE_COMPARISONS

INTERFACE_COMPARISONS = (
    ("letter_canonical", "letter_reversed", "canonical_minus_reversed"),
    ("direct_label_completion", "letter_canonical", "direct_minus_canonical"),
    ("opaque_completion", "letter_canonical", "opaque_minus_canonical"),
    ("ordinal_completion", "letter_canonical", "ordinal_minus_canonical"),
)

CONTEXT_METRICS = (
    "context_interaction",
    "discrimination_gap_change",
    "patched_rank_correct",
    "rank_sign_preserved",
)

CALIBRATION_COMPARISONS = (
    "raw_minus_random",
    "mapping_balanced_minus_random",
    "label_only_minus_random",
    "slot_layout_minus_random",
)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _average_templates(frame: pd.DataFrame, *, metrics: Iterable[str]) -> pd.DataFrame:
    keys = ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface"]
    aggregations: dict[str, tuple[str, str]] = {
        metric: (metric, "mean") for metric in metrics
    }
    aggregations["n_templates"] = ("template", "nunique")
    return frame.groupby(keys, as_index=False).agg(**aggregations)


def _combine_pair_effect_sources(
    cross_interface: pd.DataFrame,
    nuisance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface", "template"]
    required = set(keys) | {"target_margin_gain"}
    _require_columns(cross_interface, required, source="cross_interface_pair_effects")
    _require_columns(nuisance, required, source="nuisance_pair_effects")
    frames = []
    inventory = []
    for priority, (name, frame) in enumerate((
        ("cross_interface", cross_interface),
        ("nuisance", nuisance),
    )):
        selected = frame[keys + ["target_margin_gain"]].copy()
        selected["source_priority"] = priority
        selected["source_name"] = name
        frames.append(selected)
        inventory.append({
            "analysis": "input_source",
            "metric": "target_margin_gain",
            "comparison": name,
            "status": "complete",
            "n_rows": len(selected),
        })
    combined = pd.concat(frames, ignore_index=True, sort=False)
    duplicates = combined.duplicated(keys, keep=False)
    if duplicates.any():
        spread = (
            combined.loc[duplicates]
            .groupby(keys)["target_margin_gain"]
            .agg(lambda values: float(values.max() - values.min()))
        )
        if (spread > 1e-6).any():
            raise ValueError("Repeated cross-interface and nuisance pair effects disagree")
    combined = (
        combined.sort_values("source_priority")
        .drop_duplicates(keys, keep="last")
        .drop(columns=["source_priority", "source_name"])
        .reset_index(drop=True)
    )
    return combined, pd.DataFrame(inventory)


def _equal_strata_bootstrap(
    frame: pd.DataFrame,
    *,
    strata_columns: list[str],
    n_boot: int,
    confidence: float,
    chunk_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values_by_stratum = [
        group["effect"].astype(float).to_numpy()
        for _, group in frame.groupby(strata_columns, sort=True)
    ]
    values_by_stratum = [values[np.isfinite(values)] for values in values_by_stratum]
    values_by_stratum = [values for values in values_by_stratum if len(values)]
    if not values_by_stratum:
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_excludes_zero": False,
            "n_pairs": 0,
            "n_strata": 0,
        }
    estimate = float(np.mean([values.mean() for values in values_by_stratum]))
    bootstrap = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        draws = np.zeros(stop - start, dtype=float)
        for values in values_by_stratum:
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            draws += values[indices].mean(axis=1) / len(values_by_stratum)
        bootstrap[start:stop] = draws
    tail = (1.0 - confidence) / 2.0
    low = float(np.quantile(bootstrap, tail))
    high = float(np.quantile(bootstrap, 1.0 - tail))
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "n_pairs": int(frame["pair_id"].nunique()),
        "n_pair_model_units": int(len(frame)),
        "n_strata": len(values_by_stratum),
    }


def _paired_modes(
    frame: pd.DataFrame,
    *,
    metric: str,
    comparisons: Iterable[tuple[str, str, str]],
    extra_keys: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keys = ["model_alias", "model_name", "pair_id", "pair_type", "interface", *extra_keys]
    rows: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    available = set(frame["mode"].astype(str).unique())
    for left_mode, right_mode, comparison in comparisons:
        if left_mode not in available or right_mode not in available:
            inventory.append({
                "analysis": "mode_comparison",
                "metric": metric,
                "comparison": comparison,
                "status": "skipped_missing_mode",
                "left_mode": left_mode,
                "right_mode": right_mode,
                "n_rows": 0,
            })
            continue
        left = frame[frame["mode"].eq(left_mode)][keys + [metric]].rename(columns={metric: "left"})
        right = frame[frame["mode"].eq(right_mode)][keys + [metric]].rename(columns={metric: "right"})
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
        merged["effect"] = merged["left"].astype(float) - merged["right"].astype(float)
        merged["metric"] = metric
        merged["comparison"] = comparison
        merged["left_mode"] = left_mode
        merged["right_mode"] = right_mode
        rows.append(merged)
        inventory.append({
            "analysis": "mode_comparison",
            "metric": metric,
            "comparison": comparison,
            "status": "complete" if len(merged) else "empty",
            "left_mode": left_mode,
            "right_mode": right_mode,
            "n_rows": len(merged),
        })
    paired = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return paired, inventory


def _paired_interfaces(
    frame: pd.DataFrame,
    *,
    metric: str,
    extra_keys: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keys = ["model_alias", "model_name", "pair_id", "pair_type", "mode", *extra_keys]
    rows: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    available = set(frame["interface"].astype(str).unique())
    for left_interface, right_interface, comparison in INTERFACE_COMPARISONS:
        if left_interface not in available or right_interface not in available:
            inventory.append({
                "analysis": "interface_comparison",
                "metric": metric,
                "comparison": comparison,
                "status": "skipped_missing_interface",
                "left_interface": left_interface,
                "right_interface": right_interface,
                "n_rows": 0,
            })
            continue
        left = frame[frame["interface"].eq(left_interface)][keys + [metric]].rename(columns={metric: "left"})
        right = frame[frame["interface"].eq(right_interface)][keys + [metric]].rename(columns={metric: "right"})
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
        merged["effect"] = merged["left"].astype(float) - merged["right"].astype(float)
        merged["metric"] = metric
        merged["comparison"] = comparison
        merged["interface"] = f"{left_interface}_vs_{right_interface}"
        merged["left_interface"] = left_interface
        merged["right_interface"] = right_interface
        rows.append(merged)
        inventory.append({
            "analysis": "interface_comparison",
            "metric": metric,
            "comparison": comparison,
            "status": "complete" if len(merged) else "empty",
            "left_interface": left_interface,
            "right_interface": right_interface,
            "n_rows": len(merged),
        })
    paired = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return paired, inventory


def _paired_interfaces_after_control(
    mode_comparison_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Compare interfaces after a direction-control paired subtraction.

    This is a difference-in-differences estimand. It must be kept separate
    from the unadjusted within-direction interface comparison because random
    and other controls can themselves differ across interfaces.
    """
    keys = [
        "model_alias", "model_name", "pair_id", "pair_type",
        "comparison", "metric", "left_mode", "right_mode",
    ]
    rows: list[pd.DataFrame] = []
    available = set(mode_comparison_pairs["interface"].astype(str).unique())
    for left_interface, right_interface, interface_comparison in INTERFACE_COMPARISONS:
        if left_interface not in available or right_interface not in available:
            continue
        left = (
            mode_comparison_pairs[mode_comparison_pairs["interface"].eq(left_interface)]
            [keys + ["effect"]]
            .rename(columns={"effect": "left_interface_effect"})
        )
        right = (
            mode_comparison_pairs[mode_comparison_pairs["interface"].eq(right_interface)]
            [keys + ["effect"]]
            .rename(columns={"effect": "right_interface_effect"})
        )
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
        merged["effect"] = (
            merged["left_interface_effect"].astype(float)
            - merged["right_interface_effect"].astype(float)
        )
        merged["interface_comparison"] = interface_comparison
        merged["interface"] = f"{left_interface}_vs_{right_interface}"
        merged["left_interface"] = left_interface
        merged["right_interface"] = right_interface
        rows.append(merged)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _bootstrap_grouped(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    config: InterfaceStatisticsConfig,
    rng: np.random.Generator,
    per_model: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        record = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        inference_frame = group
        strata = ["pair_type"]
        if not per_model:
            # The same strict pair is evaluated by every fixed model. Average
            # models first so pair resampling preserves that dependence while
            # retaining equal model weight.
            inference_frame = (
                group.groupby(["pair_id", "pair_type"], as_index=False)
                .agg(effect=("effect", "mean"))
            )
        record.update(_equal_strata_bootstrap(
            inference_frame,
            strata_columns=strata,
            n_boot=config.n_boot,
            confidence=config.confidence,
            chunk_size=config.bootstrap_chunk_size,
            rng=rng,
        ))
        record["n_models"] = int(group["model_alias"].nunique())
        record["n_pair_model_units"] = int(len(group))
        record["n_contrasts"] = int(group["pair_type"].nunique())
        record["confidence"] = config.confidence
        record["estimand"] = (
            "model_averaged_equal_contrast_pair_bootstrap"
            if not per_model
            else "equal_contrast_pair_bootstrap"
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _leave_one_group_out(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        base = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        for held_out in sorted(group["model_alias"].astype(str).unique()):
            selected = group[group["model_alias"].astype(str).ne(held_out)]
            record = dict(base)
            record.update({
                "held_out_model": held_out,
                "estimate": float(selected.groupby(["model_alias", "pair_type"])["effect"].mean().mean()),
                "n_models": int(selected["model_alias"].nunique()),
                "n_contrasts": int(selected["pair_type"].nunique()),
            })
            model_rows.append(record)
        for held_out in sorted(group["pair_type"].astype(str).unique()):
            selected = group[group["pair_type"].astype(str).ne(held_out)]
            record = dict(base)
            record.update({
                "held_out_contrast": held_out,
                "estimate": float(selected.groupby(["model_alias", "pair_type"])["effect"].mean().mean()),
                "n_models": int(selected["model_alias"].nunique()),
                "n_contrasts": int(selected["pair_type"].nunique()),
            })
            contrast_rows.append(record)
    return pd.DataFrame(model_rows), pd.DataFrame(contrast_rows)


def _leave_one_template_out(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        base = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        for held_out in sorted(group["template"].astype(str).unique()):
            selected = group[group["template"].astype(str).ne(held_out)]
            pair_averaged = (
                selected.groupby(
                    ["model_alias", "pair_id", "pair_type"],
                    as_index=False,
                )
                .agg(effect=("effect", "mean"))
            )
            record = dict(base)
            record.update({
                "held_out_template": held_out,
                "estimate": float(
                    pair_averaged.groupby(["model_alias", "pair_type"])["effect"].mean().mean()
                ),
                "n_templates": int(selected["template"].nunique()),
                "n_models": int(selected["model_alias"].nunique()),
                "n_contrasts": int(selected["pair_type"].nunique()),
            })
            rows.append(record)
    return pd.DataFrame(rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def _build_within_interface_calibration(
    mode_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize paired direction-minus-random effects within each score space.

    The resulting values are descriptive calibration statistics. They avoid
    treating raw letter, label, and codeword margins as metrically identical:
    every standardization is computed within one model, contrast, and
    interface before equal-weight aggregation.
    """
    selected = mode_pairs[
        mode_pairs["comparison"].isin(CALIBRATION_COMPARISONS)
        & mode_pairs["right_mode"].eq("random_direction_control")
    ].copy()
    group_columns = [
        "model_alias", "model_name", "pair_type", "comparison", "interface"
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(group_columns, sort=True):
        record = dict(zip(group_columns, keys))
        adjusted = pd.to_numeric(group["effect"], errors="coerce").to_numpy(dtype=float)
        random_values = pd.to_numeric(group["right"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(adjusted) & np.isfinite(random_values)
        adjusted = adjusted[valid]
        random_values = random_values[valid]
        n_pairs = int(len(adjusted))
        mean_adjusted = float(np.mean(adjusted)) if n_pairs else np.nan
        adjusted_sd = float(np.std(adjusted, ddof=1)) if n_pairs > 1 else np.nan
        random_sd = float(np.std(random_values, ddof=1)) if n_pairs > 1 else np.nan
        record.update(
            n_pairs=n_pairs,
            mean_adjusted_gain=mean_adjusted,
            adjusted_gain_sd=adjusted_sd,
            random_gain_sd=random_sd,
            paired_standardized_gain=_safe_ratio(mean_adjusted, adjusted_sd),
            random_scale_standardized_gain=_safe_ratio(mean_adjusted, random_sd),
            positive_pair_rate=float(np.mean(adjusted > 0.0)) if n_pairs else np.nan,
            negative_pair_rate=float(np.mean(adjusted < 0.0)) if n_pairs else np.nan,
            zero_pair_rate=float(np.mean(adjusted == 0.0)) if n_pairs else np.nan,
        )
        rows.append(record)
    by_stratum = pd.DataFrame(rows)
    summary_metrics = {
        "mean_adjusted_gain": ("mean_adjusted_gain", "mean"),
        "mean_paired_standardized_gain": ("paired_standardized_gain", "mean"),
        "median_paired_standardized_gain": ("paired_standardized_gain", "median"),
        "min_paired_standardized_gain": ("paired_standardized_gain", "min"),
        "max_paired_standardized_gain": ("paired_standardized_gain", "max"),
        "mean_random_scale_standardized_gain": ("random_scale_standardized_gain", "mean"),
        "mean_positive_pair_rate": ("positive_pair_rate", "mean"),
        "n_positive_strata": ("mean_adjusted_gain", lambda values: int((values > 0).sum())),
        "n_strata": ("mean_adjusted_gain", "size"),
        "n_models": ("model_alias", "nunique"),
        "n_contrasts": ("pair_type", "nunique"),
    }
    if by_stratum.empty:
        return by_stratum, pd.DataFrame(), pd.DataFrame()
    global_summary = (
        by_stratum.groupby(["comparison", "interface"], as_index=False)
        .agg(**summary_metrics)
        .sort_values(["comparison", "interface"])
    )
    per_model = (
        by_stratum.groupby(
            ["model_alias", "model_name", "comparison", "interface"], as_index=False
        )
        .agg(**summary_metrics)
        .sort_values(["comparison", "model_alias", "interface"])
    )
    return by_stratum, global_summary, per_model


def _build_item_level_interface_agreement(
    mode_pairs: pd.DataFrame,
    *,
    reference_interface: str = "letter_canonical",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare each pair's adjusted effect sign with the canonical interface."""
    selected = mode_pairs[
        mode_pairs["comparison"].isin(CALIBRATION_COMPARISONS)
        & mode_pairs["right_mode"].eq("random_direction_control")
    ].copy()
    keys = [
        "model_alias", "model_name", "pair_id", "pair_type", "comparison",
        "left_mode", "right_mode",
    ]
    reference = (
        selected[selected["interface"].eq(reference_interface)][keys + ["effect"]]
        .rename(columns={"effect": "reference_adjusted_gain"})
    )
    compared = selected.merge(reference, on=keys, how="inner", validate="many_to_one")
    compared = compared.rename(columns={"effect": "interface_adjusted_gain"})
    compared["reference_interface"] = reference_interface
    compared["reference_positive"] = compared["reference_adjusted_gain"].gt(0.0)
    compared["interface_positive"] = compared["interface_adjusted_gain"].gt(0.0)
    compared["both_positive"] = compared["reference_positive"] & compared["interface_positive"]
    compared["sign_eligible"] = (
        compared["reference_adjusted_gain"].ne(0.0)
        & compared["interface_adjusted_gain"].ne(0.0)
    )
    compared["same_nonzero_sign"] = (
        compared["reference_adjusted_gain"].mul(compared["interface_adjusted_gain"]).gt(0.0)
        & compared["sign_eligible"]
    )

    group_columns = [
        "model_alias", "model_name", "pair_type", "comparison", "interface"
    ]
    rows: list[dict[str, Any]] = []
    for group_keys, group in compared.groupby(group_columns, sort=True):
        record = dict(zip(group_columns, group_keys))
        record["reference_interface"] = reference_interface
        eligible = group[group["sign_eligible"]]
        record.update(
            n_pairs=int(len(group)),
            n_sign_eligible=int(len(eligible)),
            sign_agreement_rate=(
                float(eligible["same_nonzero_sign"].mean()) if len(eligible) else np.nan
            ),
            joint_positive_rate=float(group["both_positive"].mean()) if len(group) else np.nan,
            reference_positive_rate=float(group["reference_positive"].mean()) if len(group) else np.nan,
            interface_positive_rate=float(group["interface_positive"].mean()) if len(group) else np.nan,
        )
        rows.append(record)
    by_stratum = pd.DataFrame(rows)
    if by_stratum.empty:
        return compared, by_stratum, pd.DataFrame(), pd.DataFrame()
    summary_aggregations = {
        "mean_sign_agreement_rate": ("sign_agreement_rate", "mean"),
        "min_sign_agreement_rate": ("sign_agreement_rate", "min"),
        "max_sign_agreement_rate": ("sign_agreement_rate", "max"),
        "mean_joint_positive_rate": ("joint_positive_rate", "mean"),
        "mean_interface_positive_rate": ("interface_positive_rate", "mean"),
        "n_strata": ("pair_type", "size"),
        "n_models": ("model_alias", "nunique"),
        "n_contrasts": ("pair_type", "nunique"),
    }
    global_summary = (
        by_stratum.groupby(["comparison", "interface", "reference_interface"], as_index=False)
        .agg(**summary_aggregations)
        .sort_values(["comparison", "interface"])
    )
    per_model = (
        by_stratum.groupby(
            [
                "model_alias", "model_name", "comparison", "interface",
                "reference_interface",
            ],
            as_index=False,
        )
        .agg(**summary_aggregations)
        .sort_values(["comparison", "model_alias", "interface"])
    )
    return compared, by_stratum, global_summary, per_model


def build_interface_statistics(
    nuisance_pair_effects: pd.DataFrame,
    context_pair_rows: pd.DataFrame,
    config: InterfaceStatisticsConfig,
    *,
    cross_interface_pair_effects: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    common = {"model_alias", "model_name", "pair_id", "pair_type", "mode", "interface", "template"}
    _require_columns(
        nuisance_pair_effects,
        common | {"target_margin_gain"},
        source="nuisance_pair_effects",
    )
    _require_columns(
        context_pair_rows,
        common | set(CONTEXT_METRICS),
        source="context_pair_rows",
    )
    source_inventory = pd.DataFrame()
    nuisance_source = nuisance_pair_effects
    if cross_interface_pair_effects is not None:
        nuisance_source, source_inventory = _combine_pair_effect_sources(
            cross_interface_pair_effects,
            nuisance_pair_effects,
        )
    nuisance = _average_templates(nuisance_source, metrics=["target_margin_gain"])
    context = _average_templates(context_pair_rows, metrics=CONTEXT_METRICS)
    rng = np.random.default_rng(config.seed + 701)

    nuisance_mode_pairs, inventory = _paired_modes(
        nuisance,
        metric="target_margin_gain",
        comparisons=NUISANCE_MODE_COMPARISONS,
    )
    nuisance_effect_rows = nuisance.rename(columns={"target_margin_gain": "effect"}).copy()
    nuisance_effect_rows["metric"] = "target_margin_gain"
    nuisance_interface_pairs, interface_inventory = _paired_interfaces(
        nuisance,
        metric="target_margin_gain",
    )
    inventory.extend(interface_inventory)
    nuisance_control_adjusted_interface_pairs = _paired_interfaces_after_control(
        nuisance_mode_pairs
    )

    nuisance_template_mode_pairs, template_inventory = _paired_modes(
        nuisance_source,
        metric="target_margin_gain",
        comparisons=NUISANCE_MODE_COMPARISONS,
        extra_keys=["template"],
    )
    inventory.extend(template_inventory)
    nuisance_template_interface_pairs, template_interface_inventory = _paired_interfaces(
        nuisance_source,
        metric="target_margin_gain",
        extra_keys=["template"],
    )
    inventory.extend(template_interface_inventory)

    context_parts: list[pd.DataFrame] = []
    context_template_parts: list[pd.DataFrame] = []
    for metric in CONTEXT_METRICS:
        paired, metric_inventory = _paired_modes(
            context,
            metric=metric,
            comparisons=CONTEXT_MODE_COMPARISONS,
        )
        context_parts.append(paired)
        inventory.extend(metric_inventory)
        template_paired, context_template_inventory = _paired_modes(
            context_pair_rows,
            metric=metric,
            comparisons=CONTEXT_MODE_COMPARISONS,
            extra_keys=["template"],
        )
        context_template_parts.append(template_paired)
        inventory.extend(context_template_inventory)
    context_mode_pairs = pd.concat(context_parts, ignore_index=True, sort=False)
    context_template_mode_pairs = pd.concat(context_template_parts, ignore_index=True, sort=False)

    nuisance_global = _bootstrap_grouped(
        nuisance_mode_pairs,
        group_columns=["comparison", "metric", "interface"],
        config=config,
        rng=rng,
        per_model=False,
    )
    nuisance_effect_global = _bootstrap_grouped(
        nuisance_effect_rows,
        group_columns=["mode", "metric", "interface"],
        config=config,
        rng=rng,
        per_model=False,
    )
    nuisance_effect_per_model = _bootstrap_grouped(
        nuisance_effect_rows,
        group_columns=["model_alias", "model_name", "mode", "metric", "interface"],
        config=config,
        rng=rng,
        per_model=True,
    )
    nuisance_per_model = _bootstrap_grouped(
        nuisance_mode_pairs,
        group_columns=["model_alias", "model_name", "comparison", "metric", "interface"],
        config=config,
        rng=rng,
        per_model=True,
    )
    nuisance_per_contrast = _bootstrap_grouped(
        nuisance_mode_pairs,
        group_columns=["comparison", "metric", "interface", "pair_type"],
        config=config,
        rng=rng,
        per_model=False,
    )
    nuisance_per_template = _bootstrap_grouped(
        nuisance_template_mode_pairs,
        group_columns=["comparison", "metric", "interface", "template"],
        config=config,
        rng=rng,
        per_model=False,
    )
    nuisance_interface_global = _bootstrap_grouped(
        nuisance_interface_pairs,
        group_columns=["mode", "comparison", "metric", "interface"],
        config=config,
        rng=rng,
        per_model=False,
    )
    nuisance_control_adjusted_interface_global = _bootstrap_grouped(
        nuisance_control_adjusted_interface_pairs,
        group_columns=["comparison", "interface_comparison", "metric", "interface"],
        config=config,
        rng=rng,
        per_model=False,
    )
    nuisance_control_adjusted_interface_per_model = _bootstrap_grouped(
        nuisance_control_adjusted_interface_pairs,
        group_columns=[
            "model_alias", "model_name", "comparison",
            "interface_comparison", "metric", "interface",
        ],
        config=config,
        rng=rng,
        per_model=True,
    )
    nuisance_interface_per_template = _bootstrap_grouped(
        nuisance_template_interface_pairs,
        group_columns=["mode", "comparison", "metric", "interface", "template"],
        config=config,
        rng=rng,
        per_model=False,
    )
    context_global = _bootstrap_grouped(
        context_mode_pairs,
        group_columns=["comparison", "metric", "interface"],
        config=config,
        rng=rng,
        per_model=False,
    )
    context_per_model = _bootstrap_grouped(
        context_mode_pairs,
        group_columns=["model_alias", "model_name", "comparison", "metric", "interface"],
        config=config,
        rng=rng,
        per_model=True,
    )
    context_per_contrast = _bootstrap_grouped(
        context_mode_pairs,
        group_columns=["comparison", "metric", "interface", "pair_type"],
        config=config,
        rng=rng,
        per_model=False,
    )
    context_per_template = _bootstrap_grouped(
        context_template_mode_pairs,
        group_columns=["comparison", "metric", "interface", "template"],
        config=config,
        rng=rng,
        per_model=False,
    )

    nuisance_lomo, nuisance_loco = _leave_one_group_out(
        nuisance_mode_pairs,
        group_columns=["comparison", "metric", "interface"],
    )
    context_lomo, context_loco = _leave_one_group_out(
        context_mode_pairs,
        group_columns=["comparison", "metric", "interface"],
    )
    nuisance_loto = _leave_one_template_out(
        nuisance_template_mode_pairs,
        group_columns=["comparison", "metric", "interface"],
    )
    context_loto = _leave_one_template_out(
        context_template_mode_pairs,
        group_columns=["comparison", "metric", "interface"],
    )
    (
        calibration_by_stratum,
        calibration_global,
        calibration_per_model,
    ) = _build_within_interface_calibration(nuisance_mode_pairs)
    (
        agreement_pairs,
        agreement_by_stratum,
        agreement_global,
        agreement_per_model,
    ) = _build_item_level_interface_agreement(nuisance_mode_pairs)
    inventory_frame = pd.concat(
        [source_inventory, pd.DataFrame(inventory)],
        ignore_index=True,
        sort=False,
    )
    estimand_definitions = pd.DataFrame(
        [
            {
                "estimand": "direction_effect_vs_zero",
                "formula": "E[gain(direction, interface)]",
                "primary_table": "nuisance_mode_effect_global_bootstrap_ci.csv",
            },
            {
                "estimand": "direction_minus_control",
                "formula": "E[gain(direction, interface) - gain(control, interface)]",
                "primary_table": "nuisance_mode_global_bootstrap_ci.csv",
            },
            {
                "estimand": "unadjusted_interface_gap",
                "formula": "E[gain(direction, interface_a) - gain(direction, interface_b)]",
                "primary_table": "nuisance_interface_global_bootstrap_ci.csv",
            },
            {
                "estimand": "control_adjusted_interface_gap",
                "formula": "E[(gain(direction,a)-gain(control,a)) - (gain(direction,b)-gain(control,b))]",
                "primary_table": "nuisance_control_adjusted_interface_global_bootstrap_ci.csv",
            },
            {
                "estimand": "within_interface_standardized_gain",
                "formula": "d_z = mean_i(delta_i) / sample_sd_i(delta_i), delta_i = gain_i(direction)-gain_i(random), within model x contrast x interface; reported values equally average d_z across strata",
                "primary_table": "interface_calibration_global.csv",
            },
            {
                "estimand": "item_level_interface_co_direction_rate",
                "formula": "Pr[sign(adjusted_gain(interface)) = sign(adjusted_gain(original))]",
                "primary_table": "interface_sign_agreement_global.csv",
            },
            {
                "estimand": "pair_ranking_margin_change_vs_control",
                "formula": "E[(patched_gap-base_gap)_direction - (patched_gap-base_gap)_control]",
                "primary_table": "context_mode_global_bootstrap_ci.csv",
            },
        ]
    )
    return {
        "interface_statistics_inventory": inventory_frame,
        "interface_estimand_definitions": estimand_definitions,
        "nuisance_template_averaged_pair_rows": nuisance,
        "nuisance_mode_effect_rows": nuisance_effect_rows,
        "nuisance_mode_comparison_pairs": nuisance_mode_pairs,
        "nuisance_interface_comparison_pairs": nuisance_interface_pairs,
        "nuisance_control_adjusted_interface_pairs": nuisance_control_adjusted_interface_pairs,
        "nuisance_mode_effect_global_bootstrap_ci": nuisance_effect_global,
        "nuisance_mode_effect_per_model_bootstrap_ci": nuisance_effect_per_model,
        "nuisance_mode_global_bootstrap_ci": nuisance_global,
        "nuisance_mode_per_model_bootstrap_ci": nuisance_per_model,
        "nuisance_mode_per_contrast_bootstrap_ci": nuisance_per_contrast,
        "nuisance_mode_per_template_bootstrap_ci": nuisance_per_template,
        "nuisance_interface_global_bootstrap_ci": nuisance_interface_global,
        "nuisance_control_adjusted_interface_global_bootstrap_ci": nuisance_control_adjusted_interface_global,
        "nuisance_control_adjusted_interface_per_model_bootstrap_ci": nuisance_control_adjusted_interface_per_model,
        "nuisance_interface_per_template_bootstrap_ci": nuisance_interface_per_template,
        "nuisance_mode_leave_one_model_out": nuisance_lomo,
        "nuisance_mode_leave_one_contrast_out": nuisance_loco,
        "nuisance_mode_leave_one_template_out": nuisance_loto,
        "context_template_averaged_pair_rows": context,
        "context_mode_comparison_pairs": context_mode_pairs,
        "context_mode_global_bootstrap_ci": context_global,
        "context_mode_per_model_bootstrap_ci": context_per_model,
        "context_mode_per_contrast_bootstrap_ci": context_per_contrast,
        "context_mode_per_template_bootstrap_ci": context_per_template,
        "context_mode_leave_one_model_out": context_lomo,
        "context_mode_leave_one_contrast_out": context_loco,
        "context_mode_leave_one_template_out": context_loto,
        "interface_calibration_by_stratum": calibration_by_stratum,
        "interface_calibration_global": calibration_global,
        "interface_calibration_per_model": calibration_per_model,
        "interface_sign_agreement_pairs": agreement_pairs,
        "interface_sign_agreement_by_stratum": agreement_by_stratum,
        "interface_sign_agreement_global": agreement_global,
        "interface_sign_agreement_per_model": agreement_per_model,
    }


def run_interface_statistics_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = InterfaceStatisticsConfig.from_json(config_path, project_root=project_root)
    for path in (
        config.cross_interface_pair_effects,
        config.nuisance_pair_effects,
        config.context_pair_rows,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required interface-statistics input does not exist: {path}")
    tables = build_interface_statistics(
        pd.read_csv(config.nuisance_pair_effects),
        pd.read_csv(config.context_pair_rows),
        config,
        cross_interface_pair_effects=pd.read_csv(config.cross_interface_pair_effects),
    )
    tables["interface_statistics_config"] = pd.DataFrame([{
        "n_boot": config.n_boot,
        "confidence": config.confidence,
        "bootstrap_chunk_size": config.bootstrap_chunk_size,
        "seed": config.seed,
        "independent_unit": "strict_matched_pair",
        "template_policy": "equal_weight_then_average_within_pair",
        "model_policy": "equal_weight_then_average_within_pair",
        "stratification": "equal_contrast; pair bootstrap after template and model averaging",
        "cross_interface_pair_effects": str(config.cross_interface_pair_effects),
        "nuisance_pair_effects": str(config.nuisance_pair_effects),
        "context_pair_rows": str(config.context_pair_rows),
    }])
    write_tables(tables, config.output_dir)
    return tables
