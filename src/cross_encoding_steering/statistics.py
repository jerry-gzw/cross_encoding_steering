from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .io import write_tables
from .evaluation import filter_locked_test_pair_rows


PAPER_MODES = [
    "raw_caa_direction",
    "shared_only_direction",
    "residual_only_direction",
    "mixed_shared_residual",
    "loco_shared_direction",
    "loco_residual_direction",
]


@dataclass(frozen=True)
class PairedComparison:
    family: str
    left_mode: str
    right_mode: str

    @property
    def label(self) -> str:
        return f"{self.left_mode}_minus_{self.right_mode}"


DEFAULT_COMPARISONS = [
    PairedComparison("baseline", "raw_caa_direction", "zero_direction_control"),
    PairedComparison("baseline", "shared_only_direction", "zero_direction_control"),
    PairedComparison("baseline", "residual_only_direction", "zero_direction_control"),
    PairedComparison("baseline", "mixed_shared_residual", "zero_direction_control"),
    PairedComparison("control", "raw_caa_direction", "random_direction_control"),
    PairedComparison("control", "shared_only_direction", "random_direction_control"),
    PairedComparison("control", "residual_only_direction", "random_direction_control"),
    PairedComparison("control", "mixed_shared_residual", "random_direction_control"),
    PairedComparison("control", "raw_caa_direction", "wrong_direction_control"),
    PairedComparison("decomposition", "shared_only_direction", "residual_only_direction"),
    PairedComparison("decomposition", "raw_caa_direction", "shared_only_direction"),
    PairedComparison("decomposition", "raw_caa_direction", "residual_only_direction"),
    PairedComparison("decomposition", "mixed_shared_residual", "raw_caa_direction"),
    PairedComparison("loco", "loco_shared_direction", "loco_residual_direction"),
    PairedComparison("loco", "loco_shared_direction", "random_direction_control"),
    PairedComparison("loco", "raw_caa_direction", "loco_shared_direction"),
    PairedComparison("protocol", "shared_only_direction", "loco_shared_direction"),
]

PAIR_METRICS = [
    "delta_intended_acc",
    "delta_target_prob",
    "js_shift",
    "prediction_changed_rate",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.where(series.notna(), "").astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False, "": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"Cannot parse boolean values: {unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def validate_locked_pair_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "pair_id",
        "pair_type",
        "mode",
        "prompt_variant",
        "eval_direction",
        "base_intended_correct",
        "patched_intended_correct",
        "delta_target_prob",
        "js_shift",
        "prediction_changed",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Locked pair rows are missing columns: {missing}")
    if rows.empty:
        raise ValueError("Locked pair rows are empty")
    out = rows.copy()
    for column in ["base_intended_correct", "patched_intended_correct", "prediction_changed"]:
        out[column] = _as_bool(out[column])
    duplicated = out.duplicated(
        ["pair_id", "mode", "prompt_variant", "eval_direction"], keep=False
    )
    if duplicated.any():
        examples = out.loc[
            duplicated, ["pair_id", "mode", "prompt_variant", "eval_direction"]
        ].head(5)
        raise ValueError(f"Duplicate evaluation rows detected:\n{examples.to_string(index=False)}")
    return out


def build_pair_level_effects(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated prompt/direction observations to the independent pair unit."""
    clean = validate_locked_pair_rows(rows)
    clean["delta_intended_acc"] = (
        clean["patched_intended_correct"].astype(float)
        - clean["base_intended_correct"].astype(float)
    )
    clean["prediction_changed_rate"] = clean["prediction_changed"].astype(float)
    group_cols = ["pair_id", "pair_type", "mode"]
    return (
        clean.groupby(group_cols, as_index=False)
        .agg(
            n_repeated_observations=("delta_intended_acc", "size"),
            delta_intended_acc=("delta_intended_acc", "mean"),
            delta_target_prob=("delta_target_prob", "mean"),
            js_shift=("js_shift", "mean"),
            prediction_changed_rate=("prediction_changed_rate", "mean"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _bootstrap_mean_ci(
    values_by_stratum: Sequence[np.ndarray],
    *,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float,
) -> tuple[float, float]:
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between 0 and 1")
    total_n = sum(len(values) for values in values_by_stratum)
    if total_n == 0:
        return np.nan, np.nan
    boot_mean = np.zeros(int(n_boot), dtype=float)
    for values in values_by_stratum:
        if len(values) == 0:
            continue
        indices = rng.integers(0, len(values), size=(int(n_boot), len(values)))
        boot_mean += values[indices].sum(axis=1) / total_n
    tail = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(boot_mean, tail)), float(np.quantile(boot_mean, 1.0 - tail))


def _bootstrap_equal_strata_mean_ci(
    values_by_stratum: Sequence[np.ndarray],
    *,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float,
) -> tuple[float, float]:
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between 0 and 1")
    nonempty = [values for values in values_by_stratum if len(values)]
    if not nonempty:
        return np.nan, np.nan
    boot_mean = np.zeros(int(n_boot), dtype=float)
    for values in nonempty:
        indices = rng.integers(0, len(values), size=(int(n_boot), len(values)))
        boot_mean += values[indices].mean(axis=1) / len(nonempty)
    tail = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(boot_mean, tail)), float(np.quantile(boot_mean, 1.0 - tail))


def _stratified_values(group: pd.DataFrame, metric: str, strata_col: str | None) -> list[np.ndarray]:
    if strata_col is None:
        return [group[metric].astype(float).to_numpy()]
    return [part[metric].astype(float).to_numpy() for _, part in group.groupby(strata_col, sort=True)]


def build_mode_bootstrap_ci(
    pair_rows: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("mode",),
    strata_col: str | None = "pair_type",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> pd.DataFrame:
    if pair_rows.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records = []
    grouper = list(group_cols)
    for keys, group in pair_rows.groupby(grouper, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(grouper, key_tuple))
        record["n_pairs"] = int(group["pair_id"].nunique())
        record["n_groups"] = int(group["pair_type"].nunique())
        for metric in PAIR_METRICS:
            values = group[metric].astype(float).to_numpy()
            low, high = _bootstrap_mean_ci(
                _stratified_values(group, metric, strata_col),
                n_boot=n_boot,
                rng=rng,
                confidence=confidence,
            )
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
            record[f"{metric}_ci_excludes_zero"] = bool(low > 0 or high < 0)
        records.append(record)
    return pd.DataFrame(records)


def build_equal_group_mode_bootstrap_ci(
    pair_rows: pd.DataFrame,
    *,
    group_column: str = "pair_type",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> pd.DataFrame:
    if pair_rows.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records = []
    for mode, group in pair_rows.groupby("mode", sort=True):
        group_parts = [part for _, part in group.groupby(group_column, sort=True)]
        record = {
            "mode": mode,
            "n_pairs": int(group["pair_id"].nunique()),
            "n_groups": int(group[group_column].nunique()),
            "min_pairs_per_group": int(group.groupby(group_column)["pair_id"].nunique().min()),
            "max_pairs_per_group": int(group.groupby(group_column)["pair_id"].nunique().max()),
            "estimand": "equal_group_mean",
        }
        for metric in PAIR_METRICS:
            group_means = [float(part[metric].astype(float).mean()) for part in group_parts]
            low, high = _bootstrap_equal_strata_mean_ci(
                [part[metric].astype(float).to_numpy() for part in group_parts],
                n_boot=n_boot,
                rng=rng,
                confidence=confidence,
            )
            record[f"{metric}_mean"] = float(np.mean(group_means))
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
            record[f"{metric}_ci_excludes_zero"] = bool(low > 0 or high < 0)
        records.append(record)
    return pd.DataFrame(records)


def paired_sign_flip_pvalue(
    values: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Return a two-sided paired randomization p-value for a zero mean difference."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(float(values.mean()))
    if np.allclose(values, 0.0):
        return 1.0
    if len(values) <= 18:
        indices = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        bits = ((indices >> np.arange(len(values), dtype=np.uint64)) & 1).astype(float)
        signs = bits * 2.0 - 1.0
        permuted = np.abs((signs * values).mean(axis=1))
        return float(np.mean(permuted >= observed - 1e-12))
    if n_permutations < 1_000:
        raise ValueError("n_permutations must be at least 1000")
    exceed = 0
    remaining = int(n_permutations)
    while remaining:
        current = min(2_000, remaining)
        signs = rng.integers(0, 2, size=(current, len(values)), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * values).mean(axis=1))
        exceed += int(np.sum(permuted >= observed - 1e-12))
        remaining -= current
    return float((exceed + 1) / (int(n_permutations) + 1))


def paired_weighted_sign_flip_pvalue(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights)
    values = values[finite]
    weights = weights[finite]
    if len(values) == 0:
        return np.nan
    weights = weights / weights.sum()
    observed = abs(float(np.sum(weights * values)))
    if np.allclose(values, 0.0):
        return 1.0
    if n_permutations < 1_000:
        raise ValueError("n_permutations must be at least 1000")
    exceed = 0
    remaining = int(n_permutations)
    weighted_values = weights * values
    while remaining:
        current = min(2_000, remaining)
        signs = rng.integers(0, 2, size=(current, len(values)), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * weighted_values).sum(axis=1))
        exceed += int(np.sum(permuted >= observed - 1e-12))
        remaining -= current
    return float((exceed + 1) / (int(n_permutations) + 1))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _comparison_rows(
    pair_rows: pd.DataFrame,
    comparisons: Sequence[PairedComparison],
    *,
    subgroup_col: str | None,
    n_boot: int,
    n_permutations: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    subgroup_values: list[str | None]
    if subgroup_col is None:
        subgroup_values = [None]
    else:
        subgroup_values = sorted(pair_rows[subgroup_col].dropna().astype(str).unique())
    for subgroup in subgroup_values:
        current = pair_rows if subgroup is None else pair_rows[pair_rows[subgroup_col].astype(str).eq(subgroup)]
        metric_wide = current.pivot_table(
            index=["pair_id", "pair_type"],
            columns="mode",
            values=PAIR_METRICS,
            aggfunc="mean",
        )
        available_modes = set(metric_wide["delta_intended_acc"].columns)
        for comparison in comparisons:
            if comparison.left_mode not in available_modes or comparison.right_mode not in available_modes:
                continue
            difference_frame = pd.DataFrame(index=metric_wide.index).reset_index()
            for metric in PAIR_METRICS:
                difference_frame[metric] = (
                    metric_wide[(metric, comparison.left_mode)]
                    - metric_wide[(metric, comparison.right_mode)]
                ).to_numpy()
            difference_frame = difference_frame.dropna(subset=PAIR_METRICS)
            if difference_frame.empty:
                continue
            values = difference_frame["delta_intended_acc"].astype(float).to_numpy()
            low, high = _bootstrap_mean_ci(
                _stratified_values(
                    difference_frame,
                    "delta_intended_acc",
                    None if subgroup_col else "pair_type",
                ),
                n_boot=n_boot,
                rng=rng,
                confidence=confidence,
            )
            std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            record = {
                "comparison_family": comparison.family,
                "comparison": comparison.label,
                "left_mode": comparison.left_mode,
                "right_mode": comparison.right_mode,
                "n_pairs": int(len(values)),
                "mean_delta_intended_acc_difference": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "paired_effect_size_dz": float(values.mean() / std) if np.isfinite(std) and std > 0 else np.nan,
                "fraction_pairs_positive": float(np.mean(values > 0)),
                "fraction_pairs_negative": float(np.mean(values < 0)),
                "p_value_two_sided": paired_sign_flip_pvalue(
                    values,
                    n_permutations=n_permutations,
                    rng=rng,
                ),
            }
            if subgroup_col is not None:
                record[subgroup_col] = subgroup
            for metric in ["delta_target_prob", "js_shift", "prediction_changed_rate"]:
                record[f"mean_{metric}_difference"] = float(difference_frame[metric].mean())
            records.append(record)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    correction_groups = ["comparison_family"]
    if subgroup_col is not None:
        correction_groups.insert(0, subgroup_col)
    out["p_value_holm"] = np.nan
    for _, indices in out.groupby(correction_groups, sort=True).groups.items():
        index_list = list(indices)
        out.loc[index_list, "p_value_holm"] = holm_adjust(out.loc[index_list, "p_value_two_sided"])
    out["significant_holm_0_05"] = out["p_value_holm"].lt(0.05)
    sort_cols = ([subgroup_col] if subgroup_col else []) + ["comparison_family", "comparison"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def build_paired_comparisons(
    pair_rows: pd.DataFrame,
    *,
    comparisons: Sequence[PairedComparison] = DEFAULT_COMPARISONS,
    n_boot: int = 10_000,
    n_permutations: int = 20_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> pd.DataFrame:
    return _comparison_rows(
        pair_rows,
        comparisons,
        subgroup_col=None,
        n_boot=n_boot,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=seed,
    )


def build_paired_comparisons_by_group(
    pair_rows: pd.DataFrame,
    *,
    group_column: str = "pair_type",
    comparisons: Sequence[PairedComparison] = DEFAULT_COMPARISONS,
    n_boot: int = 10_000,
    n_permutations: int = 20_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> pd.DataFrame:
    return _comparison_rows(
        pair_rows,
        comparisons,
        subgroup_col=group_column,
        n_boot=n_boot,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=seed,
    )


def build_equal_group_paired_comparisons(
    pair_rows: pd.DataFrame,
    *,
    group_column: str = "pair_type",
    comparisons: Sequence[PairedComparison] = DEFAULT_COMPARISONS,
    n_boot: int = 10_000,
    n_permutations: int = 20_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> pd.DataFrame:
    if pair_rows.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    metric_wide = pair_rows.pivot_table(
        index=["pair_id", group_column],
        columns="mode",
        values=PAIR_METRICS,
        aggfunc="mean",
    )
    available_modes = set(metric_wide["delta_intended_acc"].columns)
    records = []
    for comparison in comparisons:
        if comparison.left_mode not in available_modes or comparison.right_mode not in available_modes:
            continue
        differences = pd.DataFrame(index=metric_wide.index).reset_index()
        for metric in PAIR_METRICS:
            differences[metric] = (
                metric_wide[(metric, comparison.left_mode)]
                - metric_wide[(metric, comparison.right_mode)]
            ).to_numpy()
        differences = differences.dropna(subset=PAIR_METRICS)
        if differences.empty:
            continue
        group_parts = [part for _, part in differences.groupby(group_column, sort=True)]
        values = differences["delta_intended_acc"].astype(float).to_numpy()
        group_counts = differences.groupby(group_column)["pair_id"].transform("count").astype(float)
        n_groups = int(differences[group_column].nunique())
        weights = (1.0 / (n_groups * group_counts)).to_numpy()
        mean = float(np.sum(weights * values))
        low, high = _bootstrap_equal_strata_mean_ci(
            [part["delta_intended_acc"].astype(float).to_numpy() for part in group_parts],
            n_boot=n_boot,
            rng=rng,
            confidence=confidence,
        )
        centered = values - mean
        weighted_std = float(np.sqrt(np.sum(weights * np.square(centered))))
        record = {
            "comparison_family": comparison.family,
            "comparison": comparison.label,
            "left_mode": comparison.left_mode,
            "right_mode": comparison.right_mode,
            "n_pairs": int(len(values)),
            "n_groups": n_groups,
            "estimand": "equal_group_mean",
            "mean_delta_intended_acc_difference": mean,
            "ci_low": low,
            "ci_high": high,
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "paired_effect_size_dz": mean / weighted_std if weighted_std > 0 else np.nan,
            "fraction_pairs_positive": float(
                np.mean([np.mean(part["delta_intended_acc"].astype(float) > 0) for part in group_parts])
            ),
            "fraction_pairs_negative": float(
                np.mean([np.mean(part["delta_intended_acc"].astype(float) < 0) for part in group_parts])
            ),
            "p_value_two_sided": paired_weighted_sign_flip_pvalue(
                values,
                weights,
                n_permutations=n_permutations,
                rng=rng,
            ),
        }
        for metric in ["delta_target_prob", "js_shift", "prediction_changed_rate"]:
            record[f"mean_{metric}_difference"] = float(
                np.mean([part[metric].astype(float).mean() for part in group_parts])
            )
        records.append(record)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["p_value_holm"] = np.nan
    for _, indices in out.groupby("comparison_family", sort=True).groups.items():
        index_list = list(indices)
        out.loc[index_list, "p_value_holm"] = holm_adjust(out.loc[index_list, "p_value_two_sided"])
    out["significant_holm_0_05"] = out["p_value_holm"].lt(0.05)
    return out.sort_values(["comparison_family", "comparison"]).reset_index(drop=True)


def load_locked_pair_rows(input_dir: str | Path) -> pd.DataFrame:
    root = Path(input_dir).expanduser().resolve()
    locked_path = root / "locked_test_pair_rows.csv"
    if locked_path.exists():
        return pd.read_csv(locked_path)
    test_path = root / "test_eval_rows.csv"
    settings_path = root / "selected_settings.csv"
    if test_path.exists() and settings_path.exists():
        return filter_locked_test_pair_rows(pd.read_csv(test_path), pd.read_csv(settings_path))
    raise FileNotFoundError(
        "Paired statistics require row-level test output. Run the configured decomposition evaluation so that "
        f"{locked_path.name} is written. Existing condition-level summaries are insufficient."
    )


def build_statistics_decision(
    comparisons: pd.DataFrame,
    equal_group_comparisons: pd.DataFrame | None = None,
) -> pd.DataFrame:
    def lookup(frame: pd.DataFrame, label: str, column: str) -> float | bool:
        row = frame[frame["comparison"].eq(label)]
        return row.iloc[0][column] if not row.empty else np.nan

    raw_label = "raw_caa_direction_minus_random_direction_control"
    shared_residual_label = "shared_only_direction_minus_residual_only_direction"
    shared_residual_mean = lookup(comparisons, shared_residual_label, "mean_delta_intended_acc_difference")
    shared_residual_significant = lookup(comparisons, shared_residual_label, "significant_holm_0_05")
    equal = equal_group_comparisons if equal_group_comparisons is not None else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "raw_minus_random_mean": lookup(comparisons, raw_label, "mean_delta_intended_acc_difference"),
                "raw_minus_random_ci_low": lookup(comparisons, raw_label, "ci_low"),
                "raw_minus_random_ci_high": lookup(comparisons, raw_label, "ci_high"),
                "raw_minus_random_p_holm": lookup(comparisons, raw_label, "p_value_holm"),
                "raw_signal_significant": bool(
                    lookup(comparisons, raw_label, "significant_holm_0_05")
                ),
                "shared_minus_residual_mean": shared_residual_mean,
                "shared_minus_residual_p_holm": lookup(
                    comparisons, shared_residual_label, "p_value_holm"
                ),
                "shared_stronger_than_residual": bool(
                    np.isfinite(shared_residual_mean)
                    and shared_residual_mean > 0
                    and bool(shared_residual_significant)
                ),
                "equal_group_raw_minus_random_mean": lookup(
                    equal, raw_label, "mean_delta_intended_acc_difference"
                )
                if not equal.empty
                else np.nan,
                "equal_group_raw_minus_random_ci_low": lookup(equal, raw_label, "ci_low")
                if not equal.empty
                else np.nan,
                "equal_group_raw_minus_random_ci_high": lookup(equal, raw_label, "ci_high")
                if not equal.empty
                else np.nan,
                "equal_group_raw_minus_random_p_holm": lookup(equal, raw_label, "p_value_holm")
                if not equal.empty
                else np.nan,
                "equal_group_shared_minus_residual_mean": lookup(
                    equal, shared_residual_label, "mean_delta_intended_acc_difference"
                )
                if not equal.empty
                else np.nan,
                "equal_group_shared_minus_residual_p_holm": lookup(
                    equal, shared_residual_label, "p_value_holm"
                )
                if not equal.empty
                else np.nan,
                "inference_unit": "pair_id_cluster",
                "multiple_testing": "Holm within comparison family",
            }
        ]
    )


def run_paired_statistics(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    group_column: str = "pair_type",
    n_boot: int = 10_000,
    n_permutations: int = 20_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> dict[str, pd.DataFrame]:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else input_root / "paired_statistics"
    )
    locked_rows = load_locked_pair_rows(input_root)
    pair_rows = build_pair_level_effects(locked_rows)
    mode_ci = build_mode_bootstrap_ci(
        pair_rows,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
    )
    mode_ci_by_group = build_mode_bootstrap_ci(
        pair_rows,
        group_cols=(group_column, "mode"),
        strata_col=None,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed + 1,
    )
    equal_group_mode_ci = build_equal_group_mode_bootstrap_ci(
        pair_rows,
        group_column=group_column,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed + 2,
    )
    comparisons = build_paired_comparisons(
        pair_rows,
        n_boot=n_boot,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=seed + 3,
    )
    comparisons_by_group = build_paired_comparisons_by_group(
        pair_rows,
        group_column=group_column,
        n_boot=n_boot,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=seed + 4,
    )
    equal_group_comparisons = build_equal_group_paired_comparisons(
        pair_rows,
        group_column=group_column,
        n_boot=n_boot,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=seed + 5,
    )
    decision = build_statistics_decision(comparisons, equal_group_comparisons)
    run_config = pd.DataFrame(
        [
            {
                "input_dir": str(input_root),
                "output_dir": str(output_root),
                "n_boot": int(n_boot),
                "n_permutations": int(n_permutations),
                "confidence": float(confidence),
                "seed": int(seed),
                "inference_unit": "pair_id_cluster",
                "group_column": group_column,
                "overall_bootstrap": f"stratified_by_{group_column}",
                "paired_test": "two_sided_sign_flip",
                "multiple_testing": "Holm_within_family",
            }
        ]
    )
    outputs = {
        "locked_test_pair_rows": locked_rows,
        "pair_level_effects": pair_rows,
        "mode_bootstrap_ci": mode_ci,
        "mode_bootstrap_ci_by_group": mode_ci_by_group,
        "mode_equal_group_bootstrap_ci": equal_group_mode_ci,
        "paired_comparisons": comparisons,
        "paired_comparisons_by_group": comparisons_by_group,
        "equal_group_paired_comparisons": equal_group_comparisons,
        "statistics_decision": decision,
        "statistics_run_config": run_config,
    }
    write_tables(outputs, output_root)
    return outputs
