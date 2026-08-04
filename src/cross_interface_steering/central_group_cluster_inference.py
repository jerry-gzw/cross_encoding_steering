"""Group-cluster uncertainty for the central NormBank evidence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io import write_tables


@dataclass(frozen=True)
class CentralGroupClusterConfig:
    project_root: Path
    output_dir: Path
    pairs_path: Path
    letter_effects_path: Path
    letter_pair_ci_path: Path
    factorial_effects_path: Path
    factorial_key_path: Path
    factorial_pair_ci_path: Path
    factorial_baseline_rows_path: Path
    readout_effects_path: Path
    readout_pair_ci_path: Path
    n_boot: int = 5_000
    confidence: float = 0.95
    seed: int = 13

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> "CentralGroupClusterConfig":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text())
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else config_path.parent.parent.resolve()
        )

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        inputs = data.get("inputs", {})
        statistics = data.get("statistics", {})
        config = cls(
            project_root=root,
            output_dir=resolve(
                data.get(
                    "output_dir",
                    "outputs/central_group_cluster_inference",
                )
            ),
            pairs_path=resolve(
                inputs.get(
                    "pairs_path",
                    "outputs/group_disjoint_normbank/"
                    "prepared/pairs.csv",
                )
            ),
            letter_effects_path=resolve(
                inputs.get(
                    "letter_effects_path",
                    "outputs/group_disjoint_normbank/"
                    "letter_permutations/statistics/"
                    "letter_permutation_pair_adjusted_effects.csv",
                )
            ),
            letter_pair_ci_path=resolve(
                inputs.get(
                    "letter_pair_ci_path",
                    "outputs/group_disjoint_normbank/"
                    "letter_permutations/statistics/"
                    "letter_permutation_bootstrap_ci.csv",
                )
            ),
            factorial_effects_path=resolve(
                inputs.get(
                    "factorial_effects_path",
                    "outputs/interface_factorial_audit/"
                    "factorial_pair_effects.csv",
                )
            ),
            factorial_key_path=resolve(
                inputs.get(
                    "factorial_key_path",
                    "outputs/interface_factorial_audit/"
                    "mapping_key_competence_summary.csv",
                )
            ),
            factorial_pair_ci_path=resolve(
                inputs.get(
                    "factorial_pair_ci_path",
                    "outputs/interface_factorial_audit/"
                    "statistics/factorial_pair_cluster_bootstrap_ci.csv",
                )
            ),
            factorial_baseline_rows_path=resolve(
                inputs.get(
                    "factorial_baseline_rows_path",
                    "outputs/interface_factorial_audit/"
                    "baseline_competence_rows.csv",
                )
            ),
            readout_effects_path=resolve(
                inputs.get(
                    "readout_effects_path",
                    "outputs/group_disjoint_normbank/"
                    "readout_geometry_audit_v2_full/readout_pair_effects.csv",
                )
            ),
            readout_pair_ci_path=resolve(
                inputs.get(
                    "readout_pair_ci_path",
                    "outputs/group_disjoint_normbank/"
                    "readout_geometry_audit_v2_full/"
                    "readout_component_paired_statistics.csv",
                )
            ),
            n_boot=int(statistics.get("n_boot", 5_000)),
            confidence=float(statistics.get("confidence", 0.95)),
            seed=int(statistics.get("seed", 13)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.n_boot <= 0:
            raise ValueError("statistics.n_boot must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("statistics.confidence must be in (0, 1)")
        missing = [
            str(path)
            for path in (
                self.pairs_path,
                self.letter_effects_path,
                self.letter_pair_ci_path,
                self.factorial_effects_path,
                self.factorial_key_path,
                self.factorial_pair_ci_path,
                self.factorial_baseline_rows_path,
                self.readout_effects_path,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing central-inference input files: " + ", ".join(missing)
            )


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _pair_group_map(pairs: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        pairs,
        {"pair_id", "pair_type", "group_key", "split"},
        name="Strict NormBank pair table",
    )
    mapping = pairs[["pair_id", "pair_type", "group_key", "split"]].drop_duplicates()
    if mapping["pair_id"].duplicated().any():
        conflicts = mapping.loc[mapping["pair_id"].duplicated(False), "pair_id"].unique()
        raise ValueError(
            "Each pair_id must map to one context group; conflicting IDs include "
            f"{list(conflicts[:5])}"
        )
    if mapping["group_key"].isna().any() or mapping["group_key"].astype(str).str.strip().eq("").any():
        raise ValueError("Every strict NormBank pair must have a non-empty group_key")
    return mapping


def _attach_groups(frame: pd.DataFrame, pair_map: pd.DataFrame, *, name: str) -> pd.DataFrame:
    _require_columns(frame, {"pair_id", "pair_type"}, name=name)
    merged = frame.merge(
        pair_map,
        on=["pair_id", "pair_type"],
        how="left",
        validate="many_to_one",
    )
    if merged["group_key"].isna().any():
        missing = merged.loc[merged["group_key"].isna(), "pair_id"].drop_duplicates()
        raise ValueError(
            f"{name} contains pair IDs absent from the strict pair table: "
            f"{missing.head(5).tolist()}"
        )
    non_test = sorted(merged.loc[~merged["split"].eq("test"), "split"].dropna().unique())
    if non_test:
        raise ValueError(f"{name} contains non-test pairs from splits: {non_test}")
    return merged.drop(columns="split")


def _pair_units(
    frame: pd.DataFrame,
    *,
    strata: tuple[str, ...],
    values: tuple[str, ...],
) -> pd.DataFrame:
    keys = [*strata, "pair_id", "group_key"]
    _require_columns(frame, {*keys, *values}, name="Cluster-bootstrap rows")
    groups_per_pair = frame.groupby([*strata, "pair_id"])["group_key"].nunique()
    if (groups_per_pair != 1).any():
        raise ValueError("A pair contributes to multiple context groups within a stratum")
    return (
        frame.groupby(keys, as_index=False)[list(values)]
        .mean()
        .dropna(subset=list(values))
    )


def _cluster_bootstrap_balanced_means(
    units: pd.DataFrame,
    *,
    strata: tuple[str, ...],
    values: tuple[str, ...],
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
    standardized_value: str | None = None,
    positive_value: str | None = None,
) -> dict[str, Any]:
    """Resample complete context groups and average fixed strata equally."""
    pair_units = _pair_units(units, strata=strata, values=values)
    cluster_values = sorted(pair_units["group_key"].astype(str).unique())
    stratum_rows = pair_units[list(strata)].drop_duplicates().sort_values(list(strata))
    stratum_keys = [tuple(row) for row in stratum_rows.itertuples(index=False, name=None)]
    cluster_index = {value: index for index, value in enumerate(cluster_values)}
    stratum_index = {value: index for index, value in enumerate(stratum_keys)}
    n_groups = len(cluster_values)
    n_strata = len(stratum_keys)
    if n_groups < 2 or n_strata < 1:
        raise ValueError("Group-cluster inference requires at least two groups and one stratum")

    counts = np.zeros((n_groups, n_strata), dtype=float)
    sums = {value: np.zeros((n_groups, n_strata), dtype=float) for value in values}
    sum_squares = {
        value: np.zeros((n_groups, n_strata), dtype=float) for value in values
    }
    positives = {
        value: np.zeros((n_groups, n_strata), dtype=float) for value in values
    }
    grouped = pair_units.groupby(["group_key", *strata], sort=False)
    for key, part in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_key = str(key_tuple[0])
        stratum_key = tuple(key_tuple[1:])
        group_i = cluster_index[group_key]
        stratum_i = stratum_index[stratum_key]
        counts[group_i, stratum_i] = len(part)
        for value in values:
            array = part[value].to_numpy(dtype=float)
            sums[value][group_i, stratum_i] = array.sum()
            sum_squares[value][group_i, stratum_i] = np.square(array).sum()
            positives[value][group_i, stratum_i] = (array > 0.0).sum()

    def summarize(
        weight: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
        def finite_row_mean(values: np.ndarray) -> np.ndarray:
            counts = np.isfinite(values).sum(axis=1)
            return np.divide(
                np.nansum(values, axis=1),
                counts,
                out=np.full(len(values), np.nan, dtype=float),
                where=counts > 0,
            )

        sampled_counts = weight @ counts
        valid = sampled_counts > 0
        balanced: dict[str, np.ndarray] = {}
        standardized = None
        positive = None
        for value in values:
            sampled_sum = weight @ sums[value]
            means = np.divide(
                sampled_sum,
                sampled_counts,
                out=np.full_like(sampled_sum, np.nan),
                where=valid,
            )
            balanced[value] = finite_row_mean(means)
            if value == standardized_value:
                sampled_squares = weight @ sum_squares[value]
                variance = np.divide(
                    sampled_squares
                    - np.divide(
                        np.square(sampled_sum),
                        sampled_counts,
                        out=np.zeros_like(sampled_sum),
                        where=valid,
                    ),
                    sampled_counts - 1.0,
                    out=np.full_like(sampled_sum, np.nan),
                    where=sampled_counts > 1.0,
                )
                scales = np.sqrt(np.maximum(variance, 0.0))
                cell_values = np.divide(
                    means,
                    scales,
                    out=np.full_like(means, np.nan),
                    where=scales > 1e-12,
                )
                standardized = finite_row_mean(cell_values)
            if value == positive_value:
                sampled_positive = weight @ positives[value]
                rates = np.divide(
                    sampled_positive,
                    sampled_counts,
                    out=np.full_like(sampled_positive, np.nan),
                    where=valid,
                )
                positive = finite_row_mean(rates)
        return balanced, standardized, positive

    observed_weight = np.ones((1, n_groups), dtype=float)
    observed, observed_standardized, observed_positive = summarize(observed_weight)
    boot = {value: np.zeros(n_boot, dtype=float) for value in values}
    boot_standardized = (
        np.zeros(n_boot, dtype=float) if standardized_value is not None else None
    )
    boot_positive = np.zeros(n_boot, dtype=float) if positive_value is not None else None
    probabilities = np.full(n_groups, 1.0 / n_groups)
    chunk_size = min(200, n_boot)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        weights = rng.multinomial(n_groups, probabilities, size=stop - start)
        sampled, sampled_standardized, sampled_positive = summarize(weights)
        for value in values:
            boot[value][start:stop] = sampled[value]
        if boot_standardized is not None and sampled_standardized is not None:
            boot_standardized[start:stop] = sampled_standardized
        if boot_positive is not None and sampled_positive is not None:
            boot_positive[start:stop] = sampled_positive

    tail = (1.0 - confidence) / 2.0

    def interval(array: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(array, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return np.nan, np.nan
        return (
            float(np.quantile(finite, tail)),
            float(np.quantile(finite, 1.0 - tail)),
        )

    return {
        "point": {value: float(observed[value][0]) for value in values},
        "boot": boot,
        "interval": {value: interval(boot[value]) for value in values},
        "standardized_point": (
            float(observed_standardized[0])
            if observed_standardized is not None
            else np.nan
        ),
        "standardized_interval": (
            interval(boot_standardized)
            if boot_standardized is not None
            else (np.nan, np.nan)
        ),
        "positive_point": (
            float(observed_positive[0]) if observed_positive is not None else np.nan
        ),
        "positive_interval": (
            interval(boot_positive)
            if boot_positive is not None
            else (np.nan, np.nan)
        ),
        "n_groups": n_groups,
        "n_pairs": int(pair_units["pair_id"].nunique()),
        "n_pair_rows": len(pair_units),
        "n_strata": n_strata,
    }


def _letter_group_cluster_statistics(
    effects: pd.DataFrame,
    *,
    n_boot: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 101)
    for (metric, interface), group in effects.groupby(["metric", "interface"], sort=True):
        result = _cluster_bootstrap_balanced_means(
            group,
            strata=("model_alias", "pair_type", "random_seed"),
            values=("adjusted_gain",),
            n_boot=n_boot,
            confidence=confidence,
            rng=rng,
            standardized_value="adjusted_gain",
            positive_value="adjusted_gain",
        )
        mean_low, mean_high = result["interval"]["adjusted_gain"]
        standardized_low, standardized_high = result["standardized_interval"]
        positive_low, positive_high = result["positive_interval"]
        rows.append(
            {
                "metric": metric,
                "interface": interface,
                "mean_adjusted_gain": result["point"]["adjusted_gain"],
                "mean_adjusted_ci_low": mean_low,
                "mean_adjusted_ci_high": mean_high,
                "standardized_detectability": result["standardized_point"],
                "standardized_ci_low": standardized_low,
                "standardized_ci_high": standardized_high,
                "positive_pair_rate": result["positive_point"],
                "positive_pair_rate_ci_low": positive_low,
                "positive_pair_rate_ci_high": positive_high,
                "n_pairs": result["n_pairs"],
                "n_groups": result["n_groups"],
                "n_model_contrast_seed_strata": result["n_strata"],
                "n_boot": n_boot,
                "confidence": confidence,
                "inference_unit": (
                    "setting_behavior_group; resampled jointly across contrasts; "
                    "equal model-contrast-random-seed stratum weight"
                ),
            }
        )
    return pd.DataFrame(rows)


FACTORIAL_METRICS = (
    "semantic_margin_gain",
    "extraction_identifier_margin_gain",
    "extraction_row_margin_gain",
    "identifier_minus_semantics",
    "identifier_minus_row",
)


def _factorial_group_cluster_statistics(
    pair_effects: pd.DataFrame,
    key_competence: pd.DataFrame,
    *,
    n_boot: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    gate_columns = [
        "model_alias",
        "semantic_mapping",
        "identifier_set",
        "competence_passed",
    ]
    _require_columns(key_competence, gate_columns, name="Factorial key competence")
    pair = pair_effects.merge(
        key_competence[gate_columns],
        on=["model_alias", "semantic_mapping", "identifier_set"],
        how="left",
        validate="many_to_one",
    )
    if pair["competence_passed"].isna().any():
        raise ValueError("Some factorial cells lack a competence-gate record")
    scopes = {
        "all_cells": pair,
        "competence_gated": pair.loc[pair["competence_passed"].astype(bool)].copy(),
    }
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 211)
    for scope, scope_frame in scopes.items():
        fixed = (
            scope_frame.groupby(
                ["model_alias", "pair_type", "pair_id", "group_key"],
                as_index=False,
            )
            .agg(
                semantic_margin_gain=("semantic_margin_gain", "mean"),
                extraction_identifier_margin_gain=(
                    "extraction_identifier_margin_gain",
                    "mean",
                ),
                extraction_row_margin_gain=("extraction_row_margin_gain", "mean"),
            )
        )
        fixed["identifier_minus_semantics"] = (
            fixed["extraction_identifier_margin_gain"]
            - fixed["semantic_margin_gain"]
        )
        fixed["identifier_minus_row"] = (
            fixed["extraction_identifier_margin_gain"]
            - fixed["extraction_row_margin_gain"]
        )
        for model_alias, group in fixed.groupby("model_alias", sort=True):
            result = _cluster_bootstrap_balanced_means(
                group,
                strata=("pair_type",),
                values=FACTORIAL_METRICS,
                n_boot=n_boot,
                confidence=confidence,
                rng=rng,
            )
            for metric in FACTORIAL_METRICS:
                low, high = result["interval"][metric]
                rows.append(
                    {
                        "scope": scope,
                        "model_alias": model_alias,
                        "metric": metric,
                        "mean": result["point"][metric],
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                        "n_pairs": result["n_pairs"],
                        "n_groups": result["n_groups"],
                        "n_contrasts": result["n_strata"],
                        "n_boot": n_boot,
                        "confidence": confidence,
                        "inference_unit": (
                            "setting_behavior_group; resampled jointly across contrasts; "
                            "equal contrast weight within model"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _build_factorial_competence_subsets(
    baseline_rows: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build pair-level competence subsets without using steered outcomes."""
    keys = [
        "model_alias",
        "pair_type",
        "pair_id",
        "semantic_mapping",
        "identifier_set",
        "row_order",
    ]
    _require_columns(
        baseline_rows,
        {
            *keys,
            "endpoint",
            "correct",
            "base_gold_mean_logprob",
        },
        name="Factorial baseline competence rows",
    )
    by_condition = (
        baseline_rows.groupby(keys, as_index=False)
        .agg(
            n_endpoints=("endpoint", "nunique"),
            both_endpoints_correct=("correct", "all"),
            min_gold_mean_logprob=("base_gold_mean_logprob", "min"),
            mean_gold_mean_logprob=("base_gold_mean_logprob", "mean"),
        )
    )
    by_condition["both_endpoints_correct"] = (
        by_condition["both_endpoints_correct"].astype(bool)
        & by_condition["n_endpoints"].eq(2)
    )
    pair_keys = ["model_alias", "pair_type", "pair_id"]
    canonical = by_condition.loc[
        by_condition["semantic_mapping"].eq("canonical_taboo_normal_expected")
        & by_condition["identifier_set"].eq("letters_abc")
        & by_condition["row_order"].eq("order_123")
        & by_condition["both_endpoints_correct"]
    ]
    all_six_source = by_condition.loc[
        by_condition["identifier_set"].eq("letters_abc")
        & by_condition["row_order"].eq("order_123")
    ]
    all_six = (
        all_six_source.groupby(pair_keys, as_index=False)
        .agg(
            n_semantic_mappings=("semantic_mapping", "nunique"),
            all_conditions_correct=("both_endpoints_correct", "all"),
            all_six_min_gold_logprob=("min_gold_mean_logprob", "min"),
            all_six_mean_gold_logprob=("mean_gold_mean_logprob", "mean"),
        )
    )
    all_six = all_six.loc[
        all_six["n_semantic_mappings"].eq(6)
        & all_six["all_conditions_correct"].astype(bool)
    ].copy()
    if not all_six.empty:
        all_six["confidence_rank"] = all_six.groupby(
            ["model_alias", "pair_type"]
        )["all_six_min_gold_logprob"].rank(method="first", pct=True)
    else:
        all_six["confidence_rank"] = pd.Series(dtype=float)
    subsets = {
        "canonical_both_endpoints_correct": canonical[pair_keys].drop_duplicates(),
        "all_six_abc_both_endpoints_correct": all_six[pair_keys].drop_duplicates(),
        "all_six_abc_high_confidence": all_six.loc[
            all_six["confidence_rank"].gt(0.5), pair_keys
        ].drop_duplicates(),
    }
    return subsets, all_six


def _factorial_competence_conditioned_statistics(
    pair_effects: pd.DataFrame,
    key_competence: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    *,
    n_boot: int,
    confidence: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate factorial attribution on competence-qualified pair subsets."""
    subsets, all_six = _build_factorial_competence_subsets(baseline_rows)
    gate_columns = [
        "model_alias",
        "semantic_mapping",
        "identifier_set",
        "competence_passed",
    ]
    pair = pair_effects.merge(
        key_competence[gate_columns],
        on=["model_alias", "semantic_mapping", "identifier_set"],
        how="left",
        validate="many_to_one",
    )
    if pair["competence_passed"].isna().any():
        raise ValueError("Some factorial cells lack a competence-gate record")
    pair = pair.loc[pair["competence_passed"].astype(bool)].copy()
    pair_keys = ["model_alias", "pair_type", "pair_id"]
    rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 907)
    for scope, eligible in subsets.items():
        selected = pair.merge(
            eligible.assign(_eligible=True),
            on=pair_keys,
            how="inner",
            validate="many_to_one",
        )
        fixed = (
            selected.groupby(
                ["model_alias", "pair_type", "pair_id", "group_key"],
                as_index=False,
            )
            .agg(
                semantic_margin_gain=("semantic_margin_gain", "mean"),
                extraction_identifier_margin_gain=(
                    "extraction_identifier_margin_gain",
                    "mean",
                ),
                extraction_row_margin_gain=("extraction_row_margin_gain", "mean"),
            )
        )
        fixed["identifier_minus_semantics"] = (
            fixed["extraction_identifier_margin_gain"]
            - fixed["semantic_margin_gain"]
        )
        fixed["identifier_minus_row"] = (
            fixed["extraction_identifier_margin_gain"]
            - fixed["extraction_row_margin_gain"]
        )
        model_groups = [("__pooled__", fixed)]
        model_groups.extend(list(fixed.groupby("model_alias", sort=True)))
        for model_alias, group in model_groups:
            if group.empty:
                continue
            strata = (
                ("model_alias", "pair_type")
                if model_alias == "__pooled__"
                else ("pair_type",)
            )
            result = _cluster_bootstrap_balanced_means(
                group,
                strata=strata,
                values=FACTORIAL_METRICS,
                n_boot=n_boot,
                confidence=confidence,
                rng=rng,
            )
            inventory_rows.append(
                {
                    "scope": scope,
                    "model_alias": model_alias,
                    "n_pairs": result["n_pairs"],
                    "n_groups": result["n_groups"],
                    "n_model_contrast_strata": result["n_strata"],
                }
            )
            for metric in FACTORIAL_METRICS:
                low, high = result["interval"][metric]
                rows.append(
                    {
                        "scope": scope,
                        "model_alias": model_alias,
                        "metric": metric,
                        "mean": result["point"][metric],
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                        "n_pairs": result["n_pairs"],
                        "n_groups": result["n_groups"],
                        "n_model_contrast_strata": result["n_strata"],
                        "n_boot": n_boot,
                        "confidence": confidence,
                        "selection_uses_steered_outcomes": False,
                        "inference_unit": (
                            "setting_behavior_group; competence selected from "
                            "unsteered predictions; equal model-contrast weight"
                        ),
                    }
                )
    inventory = pd.DataFrame(inventory_rows)
    if not all_six.empty:
        confidence_summary = (
            all_six.groupby(["model_alias", "pair_type"], as_index=False)
            .agg(
                n_all_six_correct_pairs=("pair_id", "nunique"),
                min_gold_logprob_median=("all_six_min_gold_logprob", "median"),
            )
        )
        inventory = inventory.merge(
            confidence_summary.groupby("model_alias", as_index=False).agg(
                n_all_six_correct_pairs=("n_all_six_correct_pairs", "sum"),
                min_gold_logprob_median=("min_gold_logprob_median", "median"),
            ),
            on="model_alias",
            how="left",
        )
    return pd.DataFrame(rows), inventory


def _balanced_readout_units(
    rows: pd.DataFrame,
    *,
    modes: dict[str, str],
) -> pd.DataFrame:
    selected = rows.loc[
        rows["mapping_name"].ne("canonical_taboo_normal_expected")
        & rows["mode"].isin(modes)
    ].copy()
    selected["component"] = selected["mode"].map(modes)
    wide = (
        selected.pivot_table(
            index=["model_alias", "pair_type", "pair_id", "group_key"],
            columns="component",
            values="original_letter_prob_gain",
            aggfunc="mean",
        )
        .reset_index()
    )
    return wide


def _readout_group_cluster_statistics(
    pair_effects: pd.DataFrame,
    *,
    n_boot: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    core = _balanced_readout_units(
        pair_effects,
        modes={
            "raw_canonical_direction": "raw",
            "readout_projection_natural": "projection",
            "readout_orthogonal_natural": "residual",
        },
    ).dropna(subset=["raw", "projection", "residual"])
    core["projection_minus_residual"] = core["projection"] - core["residual"]
    core["raw_minus_residual"] = core["raw"] - core["residual"]

    random_rows = pair_effects.loc[
        pair_effects["mapping_name"].ne("canonical_taboo_normal_expected")
        & pair_effects["mode"].astype(str).str.startswith("cov_readout_seed_")
    ].copy()
    random_mean = (
        random_rows.groupby(
            ["model_alias", "pair_type", "pair_id", "group_key"],
            as_index=False,
        )["original_letter_prob_gain"]
        .mean()
        .rename(columns={"original_letter_prob_gain": "subspace_random"})
    )
    projection_random = core[
        ["model_alias", "pair_type", "pair_id", "group_key", "projection"]
    ].merge(
        random_mean,
        on=["model_alias", "pair_type", "pair_id", "group_key"],
        how="inner",
        validate="one_to_one",
    )
    projection_random["projection_minus_subspace_random"] = (
        projection_random["projection"] - projection_random["subspace_random"]
    )

    rng = np.random.default_rng(seed + 307)
    core_result = _cluster_bootstrap_balanced_means(
        core,
        strata=("model_alias", "pair_type"),
        values=(
            "raw",
            "projection",
            "residual",
            "projection_minus_residual",
            "raw_minus_residual",
        ),
        n_boot=n_boot,
        confidence=confidence,
        rng=rng,
    )
    random_result = _cluster_bootstrap_balanced_means(
        projection_random,
        strata=("model_alias", "pair_type"),
        values=("projection", "subspace_random", "projection_minus_subspace_random"),
        n_boot=n_boot,
        confidence=confidence,
        rng=rng,
    )
    rows: list[dict[str, Any]] = []
    for comparison in (
        "projection_minus_residual",
        "raw_minus_residual",
    ):
        low, high = core_result["interval"][comparison]
        rows.append(
            {
                "comparison": comparison,
                "statistic": "paired_difference",
                "estimate": core_result["point"][comparison],
                "ci_low": low,
                "ci_high": high,
                "n_pair_units": core_result["n_pair_rows"],
                "n_unique_pairs": core_result["n_pairs"],
                "n_groups": core_result["n_groups"],
                "n_model_contrast_strata": core_result["n_strata"],
            }
        )
    comparison = "projection_minus_subspace_random"
    low, high = random_result["interval"][comparison]
    rows.append(
        {
            "comparison": comparison,
            "statistic": "paired_difference",
            "estimate": random_result["point"][comparison],
            "ci_low": low,
            "ci_high": high,
            "n_pair_units": random_result["n_pair_rows"],
            "n_unique_pairs": random_result["n_pairs"],
            "n_groups": random_result["n_groups"],
            "n_model_contrast_strata": random_result["n_strata"],
        }
    )
    for component in ("projection", "residual"):
        numerator = core_result["point"][component]
        denominator = core_result["point"]["raw"]
        estimate = numerator / denominator
        boot = np.divide(
            core_result["boot"][component],
            core_result["boot"]["raw"],
            out=np.full(n_boot, np.nan),
            where=np.abs(core_result["boot"]["raw"]) > 1e-12,
        )
        tail = (1.0 - confidence) / 2.0
        finite = boot[np.isfinite(boot)]
        rows.append(
            {
                "comparison": f"{component}_retention_ratio",
                "statistic": "ratio_of_balanced_means",
                "estimate": estimate,
                "ci_low": float(np.quantile(finite, tail)),
                "ci_high": float(np.quantile(finite, 1.0 - tail)),
                "n_pair_units": core_result["n_pair_rows"],
                "n_unique_pairs": core_result["n_pairs"],
                "n_groups": core_result["n_groups"],
                "n_model_contrast_strata": core_result["n_strata"],
            }
        )
    output = pd.DataFrame(rows)
    output["ci_excludes_zero"] = (
        output["ci_low"].gt(0.0) | output["ci_high"].lt(0.0)
    )
    output["n_boot"] = n_boot
    output["confidence"] = confidence
    output["inference_unit"] = (
        "setting_behavior_group; resampled jointly across contrasts; "
        "equal model-contrast stratum weight"
    )
    return output


def _pair_bootstrap_balanced_means(
    units: pd.DataFrame,
    *,
    values: tuple[str, ...],
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Bootstrap pair IDs independently within fixed model-contrast strata."""
    parts = [
        part[list(values)].dropna().to_numpy(dtype=float)
        for _, part in units.groupby(["model_alias", "pair_type"], sort=True)
        if len(part)
    ]
    if not parts:
        raise ValueError("Pair bootstrap requires at least one model-contrast stratum")
    point = {
        value: float(np.mean([part[:, index].mean() for part in parts]))
        for index, value in enumerate(values)
    }
    boot = {value: np.zeros(n_boot, dtype=float) for value in values}
    for part in parts:
        indices = rng.integers(0, len(part), size=(n_boot, len(part)))
        sampled = part[indices].mean(axis=1)
        for value_index, value in enumerate(values):
            boot[value] += sampled[:, value_index] / len(parts)
    tail = (1.0 - confidence) / 2.0
    return {
        "point": point,
        "boot": boot,
        "interval": {
            value: (
                float(np.quantile(boot[value], tail)),
                float(np.quantile(boot[value], 1.0 - tail)),
            )
            for value in values
        },
        "n_pair_units": int(sum(len(part) for part in parts)),
        "n_strata": len(parts),
    }


def _reconstruct_readout_pair_bootstrap(
    pair_effects: pd.DataFrame,
    *,
    n_boot: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Rebuild the legacy readout pair-bootstrap reference from pair effects."""
    core = _balanced_readout_units(
        pair_effects,
        modes={
            "raw_canonical_direction": "raw",
            "readout_projection_natural": "projection",
            "readout_orthogonal_natural": "residual",
        },
    ).dropna(subset=["raw", "projection", "residual"])
    core["projection_minus_residual"] = core["projection"] - core["residual"]
    core["raw_minus_residual"] = core["raw"] - core["residual"]

    random_rows = pair_effects.loc[
        pair_effects["mapping_name"].ne("canonical_taboo_normal_expected")
        & pair_effects["mode"].astype(str).str.startswith("cov_readout_seed_")
    ].copy()
    random_mean = (
        random_rows.groupby(
            ["model_alias", "pair_type", "pair_id", "group_key"],
            as_index=False,
        )["original_letter_prob_gain"]
        .mean()
        .rename(columns={"original_letter_prob_gain": "subspace_random"})
    )
    projection_random = core[
        ["model_alias", "pair_type", "pair_id", "group_key", "projection"]
    ].merge(
        random_mean,
        on=["model_alias", "pair_type", "pair_id", "group_key"],
        how="inner",
        validate="one_to_one",
    )
    projection_random["projection_minus_subspace_random"] = (
        projection_random["projection"] - projection_random["subspace_random"]
    )

    rng = np.random.default_rng(seed)
    core_result = _pair_bootstrap_balanced_means(
        core,
        values=(
            "raw",
            "projection",
            "residual",
            "projection_minus_residual",
            "raw_minus_residual",
        ),
        n_boot=n_boot,
        confidence=confidence,
        rng=rng,
    )
    random_result = _pair_bootstrap_balanced_means(
        projection_random,
        values=("projection", "subspace_random", "projection_minus_subspace_random"),
        n_boot=n_boot,
        confidence=confidence,
        rng=rng,
    )
    rows: list[dict[str, Any]] = []
    for comparison in ("projection_minus_residual", "raw_minus_residual"):
        low, high = core_result["interval"][comparison]
        rows.append(
            {
                "comparison": comparison,
                "statistic": "paired_difference",
                "estimate": core_result["point"][comparison],
                "ci_low": low,
                "ci_high": high,
                "n_pair_units": core_result["n_pair_units"],
                "n_model_contrast_strata": core_result["n_strata"],
            }
        )
    comparison = "projection_minus_subspace_random"
    low, high = random_result["interval"][comparison]
    rows.append(
        {
            "comparison": comparison,
            "statistic": "paired_difference",
            "estimate": random_result["point"][comparison],
            "ci_low": low,
            "ci_high": high,
            "n_pair_units": random_result["n_pair_units"],
            "n_model_contrast_strata": random_result["n_strata"],
        }
    )
    for component in ("projection", "residual"):
        boot = np.divide(
            core_result["boot"][component],
            core_result["boot"]["raw"],
            out=np.full(n_boot, np.nan),
            where=np.abs(core_result["boot"]["raw"]) > 1e-12,
        )
        finite = boot[np.isfinite(boot)]
        tail = (1.0 - confidence) / 2.0
        rows.append(
            {
                "comparison": f"{component}_retention_ratio",
                "statistic": "ratio_of_balanced_means",
                "estimate": (
                    core_result["point"][component] / core_result["point"]["raw"]
                ),
                "ci_low": float(np.quantile(finite, tail)),
                "ci_high": float(np.quantile(finite, 1.0 - tail)),
                "n_pair_units": core_result["n_pair_units"],
                "n_model_contrast_strata": core_result["n_strata"],
            }
        )
    output = pd.DataFrame(rows)
    output["confidence"] = confidence
    output["n_boot"] = n_boot
    output["inference_unit"] = (
        "pair_id; bootstrap independently within model-by-contrast strata; "
        "equal stratum weight"
    )
    return output


def _decision_comparison(
    cluster: pd.DataFrame,
    pair: pd.DataFrame,
    *,
    analysis: str,
    keys: list[str],
    cluster_low: str,
    cluster_high: str,
    pair_low: str,
    pair_high: str,
) -> pd.DataFrame:
    required_cluster = {*keys, cluster_low, cluster_high}
    required_pair = {*keys, pair_low, pair_high}
    missing_cluster = sorted(required_cluster - set(cluster.columns))
    missing_pair = sorted(required_pair - set(pair.columns))
    if missing_cluster or missing_pair:
        raise ValueError(
            f"{analysis} inference comparison has incompatible schemas; "
            f"missing cluster columns={missing_cluster}, missing pair columns={missing_pair}"
        )
    left = cluster[[*keys, cluster_low, cluster_high]].copy().rename(
        columns={
            cluster_low: "group_cluster_ci_low",
            cluster_high: "group_cluster_ci_high",
        }
    )
    right = pair[[*keys, pair_low, pair_high]].copy().rename(
        columns={
            pair_low: "pair_bootstrap_ci_low",
            pair_high: "pair_bootstrap_ci_high",
        }
    )
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    merged.insert(0, "analysis", analysis)
    merged["group_cluster_excludes_zero"] = (
        merged["group_cluster_ci_low"].gt(0.0)
        | merged["group_cluster_ci_high"].lt(0.0)
    )
    merged["pair_bootstrap_excludes_zero"] = (
        merged["pair_bootstrap_ci_low"].gt(0.0)
        | merged["pair_bootstrap_ci_high"].lt(0.0)
    )
    merged["decision_agrees"] = (
        merged["group_cluster_excludes_zero"]
        == merged["pair_bootstrap_excludes_zero"]
    )
    merged["group_cluster_width"] = (
        merged["group_cluster_ci_high"] - merged["group_cluster_ci_low"]
    )
    merged["pair_bootstrap_width"] = (
        merged["pair_bootstrap_ci_high"] - merged["pair_bootstrap_ci_low"]
    )
    merged["width_ratio_group_over_pair"] = np.divide(
        merged["group_cluster_width"],
        merged["pair_bootstrap_width"],
        out=np.full(len(merged), np.nan),
        where=merged["pair_bootstrap_width"].abs().to_numpy() > 1e-12,
    )
    return merged


def run_central_group_cluster_inference(
    config: CentralGroupClusterConfig,
) -> dict[str, pd.DataFrame]:
    pairs = pd.read_csv(config.pairs_path)
    pair_map = _pair_group_map(pairs)
    inventory = (
        pair_map.groupby(["split", "pair_type"], as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            n_groups=("group_key", "nunique"),
            max_pairs_per_group=("group_key", lambda values: int(values.value_counts().max())),
        )
    )

    letter = _attach_groups(
        pd.read_csv(config.letter_effects_path),
        pair_map,
        name="Letter-permutation adjusted effects",
    )
    letter_cluster = _letter_group_cluster_statistics(
        letter,
        n_boot=config.n_boot,
        confidence=config.confidence,
        seed=config.seed,
    )

    factorial = _attach_groups(
        pd.read_csv(config.factorial_effects_path),
        pair_map,
        name="Factorial pair effects",
    )
    factorial_cluster = _factorial_group_cluster_statistics(
        factorial,
        pd.read_csv(config.factorial_key_path),
        n_boot=config.n_boot,
        confidence=config.confidence,
        seed=config.seed,
    )
    factorial_competence_cluster, factorial_competence_inventory = (
        _factorial_competence_conditioned_statistics(
            factorial,
            pd.read_csv(config.factorial_key_path),
            pd.read_csv(config.factorial_baseline_rows_path),
            n_boot=config.n_boot,
            confidence=config.confidence,
            seed=config.seed,
        )
    )

    readout_pair_effects = pd.read_csv(config.readout_effects_path)
    readout = _attach_groups(
        readout_pair_effects,
        pair_map,
        name="Readout-geometry pair effects",
    )
    readout_cluster = _readout_group_cluster_statistics(
        readout,
        n_boot=config.n_boot,
        confidence=config.confidence,
        seed=config.seed,
    )

    comparisons = []
    comparisons.append(
        _decision_comparison(
            letter_cluster,
            pd.read_csv(config.letter_pair_ci_path),
            analysis="letter_permutation",
            keys=["metric", "interface"],
            cluster_low="standardized_ci_low",
            cluster_high="standardized_ci_high",
            pair_low="standardized_ci_low",
            pair_high="standardized_ci_high",
        )
    )
    comparisons.append(
        _decision_comparison(
            factorial_cluster,
            pd.read_csv(config.factorial_pair_ci_path),
            analysis="factorial",
            keys=["scope", "model_alias", "metric"],
            cluster_low="ci_low",
            cluster_high="ci_high",
            pair_low="ci_low",
            pair_high="ci_high",
        )
    )
    if config.readout_pair_ci_path.exists():
        readout_pair_reference = pd.read_csv(config.readout_pair_ci_path)
        readout_pair_reference_source = "existing_readout_component_paired_statistics"
    else:
        readout_pair_reference = _reconstruct_readout_pair_bootstrap(
            readout,
            n_boot=config.n_boot,
            confidence=config.confidence,
            seed=config.seed,
        )
        if readout_pair_reference.empty:
            raise ValueError(
                "Could not reconstruct readout pair-bootstrap statistics from "
                f"{config.readout_effects_path}"
            )
        readout_pair_reference_source = "reconstructed_from_readout_pair_effects"
    comparisons.append(
        _decision_comparison(
            readout_cluster,
            readout_pair_reference,
            analysis="readout_geometry",
            keys=["comparison", "statistic"],
            cluster_low="ci_low",
            cluster_high="ci_high",
            pair_low="ci_low",
            pair_high="ci_high",
        )
    )
    comparison = pd.concat(
        [frame for frame in comparisons if not frame.empty],
        ignore_index=True,
        sort=False,
    ) if any(not frame.empty for frame in comparisons) else pd.DataFrame()
    decision = pd.DataFrame(
        [
            {
                "status": "complete",
                "n_boot": config.n_boot,
                "confidence": config.confidence,
                "n_test_pairs": int((pair_map["split"] == "test").sum()),
                "n_test_groups": int(
                    pair_map.loc[pair_map["split"].eq("test"), "group_key"].nunique()
                ),
                "n_comparisons": len(comparison),
                "n_decision_agreements": (
                    int(comparison["decision_agrees"].sum()) if not comparison.empty else 0
                ),
                "all_pair_vs_group_decisions_agree": (
                    bool(comparison["decision_agrees"].all())
                    if not comparison.empty
                    else False
                ),
                "readout_pair_reference_source": readout_pair_reference_source,
                "note": (
                    "Complete setting-behavior groups are resampled jointly across "
                    "contrasts; no GPU outputs are recomputed."
                ),
            }
        ]
    )
    tables = {
        "central_group_cluster_inventory": inventory,
        "letter_group_cluster_bootstrap_ci": letter_cluster,
        "factorial_group_cluster_bootstrap_ci": factorial_cluster,
        "factorial_competence_conditioned_group_cluster_ci":
            factorial_competence_cluster,
        "factorial_competence_subset_inventory": factorial_competence_inventory,
        "readout_group_cluster_bootstrap_ci": readout_cluster,
        "readout_pair_bootstrap_reference": readout_pair_reference,
        "pair_vs_group_cluster_inference": comparison,
        "central_group_cluster_decision": decision,
    }
    write_tables(tables, config.output_dir)
    return tables


def run_central_group_cluster_inference_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = CentralGroupClusterConfig.from_json(
        config_path,
        project_root=project_root,
    )
    return run_central_group_cluster_inference(config)
