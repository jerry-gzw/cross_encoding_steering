"""Exhaustive fixed-direction audit over all three-label letter mappings."""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cross_interface_audit import CrossInterfaceConfig
from .io import write_tables
from .mapping_audit import _load_mapping_audit_config


RANDOM_MODE_PATTERN = re.compile(r"^random_direction_control_seed_(-?\d+)$")


@dataclass(frozen=True)
class LetterPermutationStatisticsConfig:
    audit: CrossInterfaceConfig
    output_dir: Path
    n_boot: int
    confidence: float
    seed: int

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "LetterPermutationStatisticsConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        audit = CrossInterfaceConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        spec = dict(data.get("letter_permutation_statistics", {}))
        output_value = spec.get("output_dir", str(audit.audit.output_dir / "statistics"))
        output_dir = Path(str(output_value)).expanduser()
        if not output_dir.is_absolute():
            output_dir = audit.audit.project_root / output_dir
        confidence = float(spec.get("confidence", 0.95))
        if not 0.0 < confidence < 1.0:
            raise ValueError("letter_permutation_statistics.confidence must be in (0, 1)")
        config = cls(
            audit=audit,
            output_dir=output_dir.resolve(),
            n_boot=int(spec.get("n_boot", 5_000)),
            confidence=confidence,
            seed=int(spec.get("seed", 13)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.n_boot <= 0:
            raise ValueError("letter_permutation_statistics.n_boot must be positive")
        if "raw_pre_answer" not in self.audit.direction_sources:
            raise ValueError("Letter-permutation audit requires raw_pre_answer")
        if "random_ensemble" not in self.audit.direction_sources:
            raise ValueError("Letter-permutation audit requires random_ensemble")
        if len(self.audit.random_control_seeds) < 2:
            raise ValueError("Letter-permutation audit requires at least two random controls")

        labels = tuple(self.audit.audit.dataset.label_ranks)
        if len(labels) != 3:
            raise ValueError("Exhaustive letter-permutation audit currently requires three labels")
        letter_interfaces = [
            interface for interface in self.audit.interfaces if interface.kind == "letter_mcq"
        ]
        if len(letter_interfaces) != math.factorial(len(labels)):
            raise ValueError("Configure exactly all six three-label letter mappings")
        mappings = {mapping.name: mapping for mapping in self.audit.audit.mappings}
        orders = []
        for interface in letter_interfaces:
            if not interface.mapping_name or interface.mapping_name not in mappings:
                raise ValueError(f"Interface {interface.name!r} lacks a known mapping_name")
            orders.append(mappings[interface.mapping_name].option_order)
        expected = set(itertools.permutations(labels))
        if set(orders) != expected:
            raise ValueError("Configured letter interfaces do not exhaust all label permutations")


def _random_seed(mode: Any) -> int | None:
    match = RANDOM_MODE_PATTERN.match(str(mode))
    return int(match.group(1)) if match else None


def _mapping_inventory(config: LetterPermutationStatisticsConfig) -> pd.DataFrame:
    canonical_name = config.audit.audit.dataset.canonical_mapping
    mappings = {mapping.name: mapping for mapping in config.audit.audit.mappings}
    canonical = mappings[canonical_name].option_order
    rows = []
    for interface in config.audit.interfaces:
        if interface.kind != "letter_mcq":
            continue
        mapping = mappings[str(interface.mapping_name)]
        moved = sum(left != right for left, right in zip(canonical, mapping.option_order))
        permutation_class = {0: "identity", 2: "transposition", 3: "three_cycle"}.get(
            moved, f"moved_{moved}"
        )
        rows.append(
            {
                "interface": interface.name,
                "mapping_name": mapping.name,
                "option_a_label": mapping.option_order[0],
                "option_b_label": mapping.option_order[1],
                "option_c_label": mapping.option_order[2],
                "option_order": ",".join(mapping.option_order),
                "is_source_mapping": mapping.name == canonical_name,
                "n_moved_slots": moved,
                "permutation_class": permutation_class,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["is_source_mapping", "mapping_name"], ascending=[False, True]
    ).reset_index(drop=True)


def _pair_adjusted_effects(
    rows: pd.DataFrame,
    config: LetterPermutationStatisticsConfig,
) -> pd.DataFrame:
    required = {
        "model_alias",
        "model_name",
        "pair_id",
        "pair_type",
        "mode",
        "interface",
        "evaluation_policy",
        "target_margin_gain",
        "original_slot_margin_gain",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Letter-permutation eval rows are missing columns: {missing}")
    interfaces = {item.name for item in config.audit.interfaces if item.kind == "letter_mcq"}
    selected = rows.loc[
        rows["evaluation_policy"].eq("counterfactual")
        & rows["interface"].isin(interfaces)
    ].copy()
    selected["random_seed"] = selected["mode"].map(_random_seed)
    keys = ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface"]
    pair = (
        selected.groupby(keys, as_index=False)
        .agg(
            semantic_gain=("target_margin_gain", "mean"),
            original_slot_gain=("original_slot_margin_gain", "mean"),
            n_templates=("template", "nunique"),
        )
    )
    target = pair.loc[pair["mode"].eq("raw_pre_answer")].copy()
    random = pair.loc[pair["mode"].map(_random_seed).notna()].copy()
    random["random_seed"] = random["mode"].map(_random_seed).astype(int)
    if target.empty or random.empty:
        raise ValueError("Both raw_pre_answer and seeded random rows are required")
    observed_seeds = set(random["random_seed"].unique())
    expected_seeds = set(config.audit.random_control_seeds)
    if observed_seeds != expected_seeds:
        raise ValueError(
            "Seeded random rows are incomplete: "
            f"expected {sorted(expected_seeds)}, observed {sorted(observed_seeds)}"
        )
    merged = target.merge(
        random,
        on=["model_alias", "model_name", "pair_id", "pair_type", "interface"],
        how="inner",
        suffixes=("_target", "_random"),
        validate="one_to_many",
    )
    metric_frames = []
    for metric, column in (
        ("semantic_target_margin", "semantic_gain"),
        ("original_slot_margin", "original_slot_gain"),
    ):
        frame = merged[
            ["model_alias", "model_name", "pair_id", "pair_type", "interface", "random_seed"]
        ].copy()
        frame["metric"] = metric
        frame["target_gain"] = merged[f"{column}_target"]
        frame["random_gain"] = merged[f"{column}_random"]
        frame["adjusted_gain"] = frame["target_gain"] - frame["random_gain"]
        metric_frames.append(frame)
    effects = pd.concat(metric_frames, ignore_index=True)
    semantic = effects.loc[effects["metric"].eq("semantic_target_margin")].copy()
    original = effects.loc[effects["metric"].eq("original_slot_margin")].copy()
    comparison_keys = [
        "model_alias", "model_name", "pair_id", "pair_type", "interface", "random_seed"
    ]
    comparison = semantic.merge(
        original,
        on=comparison_keys,
        suffixes=("_semantic", "_original"),
        validate="one_to_one",
    )
    difference = comparison[comparison_keys].copy()
    for column in ("target_gain", "random_gain", "adjusted_gain"):
        difference[column] = (
            comparison[f"{column}_semantic"].to_numpy(dtype=float)
            - comparison[f"{column}_original"].to_numpy(dtype=float)
        )
    difference["metric"] = "semantic_minus_original_slot"
    return pd.concat([effects, difference], ignore_index=True, sort=False)


def _stratum_statistics(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["metric", "model_alias", "model_name", "pair_type", "interface", "random_seed"]
    strata = (
        effects.groupby(keys, as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_adjusted_gain=("adjusted_gain", "mean"),
            adjusted_gain_sd=("adjusted_gain", "std"),
            positive_pair_rate=("adjusted_gain", lambda values: float((values > 0).mean())),
        )
    )
    strata["standardized_detectability"] = np.divide(
        strata["mean_adjusted_gain"],
        strata["adjusted_gain_sd"],
        out=np.full(len(strata), np.nan, dtype=float),
        where=strata["adjusted_gain_sd"].to_numpy(dtype=float) > 1e-12,
    )
    per_seed = (
        strata.groupby(["metric", "interface", "random_seed"], as_index=False)
        .agg(
            n_strata=("pair_type", "size"),
            n_models=("model_alias", "nunique"),
            n_contrasts=("pair_type", "nunique"),
            mean_adjusted_gain=("mean_adjusted_gain", "mean"),
            mean_standardized_detectability=("standardized_detectability", "mean"),
            mean_positive_pair_rate=("positive_pair_rate", "mean"),
            n_positive_strata=("mean_adjusted_gain", lambda values: int((values > 0).sum())),
        )
    )
    summary = (
        per_seed.groupby(["metric", "interface"], as_index=False)
        .agg(
            n_random_seeds=("random_seed", "nunique"),
            adjusted_gain_seed_mean=("mean_adjusted_gain", "mean"),
            adjusted_gain_seed_sd=("mean_adjusted_gain", "std"),
            adjusted_gain_seed_min=("mean_adjusted_gain", "min"),
            adjusted_gain_seed_max=("mean_adjusted_gain", "max"),
            standardized_seed_mean=("mean_standardized_detectability", "mean"),
            standardized_seed_sd=("mean_standardized_detectability", "std"),
            standardized_seed_min=("mean_standardized_detectability", "min"),
            standardized_seed_max=("mean_standardized_detectability", "max"),
            positive_pair_rate_seed_mean=("mean_positive_pair_rate", "mean"),
            min_positive_strata=("n_positive_strata", "min"),
            max_positive_strata=("n_positive_strata", "max"),
        )
    )
    return strata, per_seed, summary


def _bootstrap_statistics(
    effects: pd.DataFrame,
    config: LetterPermutationStatisticsConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 701)
    output = []
    for (metric, interface), group in effects.groupby(["metric", "interface"], sort=True):
        arrays = []
        for _, contrast in group.groupby("pair_type", sort=True):
            pivot = contrast.pivot_table(
                index="pair_id",
                columns=["model_alias", "random_seed"],
                values="adjusted_gain",
                aggfunc="mean",
            ).dropna(axis=0, how="any")
            if len(pivot) >= 2:
                arrays.append(pivot.to_numpy(dtype=float))
        if not arrays:
            continue

        def observed(values: list[np.ndarray]) -> tuple[float, float, float]:
            means, standardized, positive = [], [], []
            for matrix in values:
                cell_means = matrix.mean(axis=0)
                scales = matrix.std(axis=0, ddof=1)
                cell_standardized = np.divide(
                    cell_means,
                    scales,
                    out=np.full_like(cell_means, np.nan, dtype=float),
                    where=scales > 1e-12,
                )
                means.append(float(np.mean(cell_means)))
                finite_standardized = cell_standardized[np.isfinite(cell_standardized)]
                standardized.append(
                    float(np.mean(finite_standardized)) if len(finite_standardized) else np.nan
                )
                positive.append(float(np.mean(matrix > 0.0)))
            finite_standardized = np.asarray(standardized, dtype=float)
            finite_standardized = finite_standardized[np.isfinite(finite_standardized)]
            return (
                float(np.mean(means)),
                float(np.mean(finite_standardized)) if len(finite_standardized) else np.nan,
                float(np.mean(positive)),
            )

        mean_estimate, standardized_estimate, positive_estimate = observed(arrays)
        boot_mean = np.zeros(config.n_boot, dtype=float)
        boot_standardized = np.zeros(config.n_boot, dtype=float)
        boot_positive = np.zeros(config.n_boot, dtype=float)
        chunk_size = min(200, config.n_boot)
        for start in range(0, config.n_boot, chunk_size):
            stop = min(start + chunk_size, config.n_boot)
            size = stop - start
            chunk_mean = np.zeros(size, dtype=float)
            chunk_standardized = np.zeros(size, dtype=float)
            chunk_positive = np.zeros(size, dtype=float)
            for matrix in arrays:
                indices = rng.integers(0, len(matrix), size=(size, len(matrix)))
                sampled = matrix[indices]
                cell_means = sampled.mean(axis=1)
                scales = sampled.std(axis=1, ddof=1)
                standardized = np.divide(
                    cell_means,
                    scales,
                    out=np.full_like(cell_means, np.nan, dtype=float),
                    where=scales > 1e-12,
                )
                finite = np.isfinite(standardized).sum(axis=1)
                standardized_mean = np.divide(
                    np.nansum(standardized, axis=1),
                    finite,
                    out=np.full(size, np.nan, dtype=float),
                    where=finite > 0,
                )
                chunk_mean += cell_means.mean(axis=1) / len(arrays)
                chunk_standardized += standardized_mean / len(arrays)
                chunk_positive += (sampled > 0.0).mean(axis=(1, 2)) / len(arrays)
            boot_mean[start:stop] = chunk_mean
            boot_standardized[start:stop] = chunk_standardized
            boot_positive[start:stop] = chunk_positive

        tail = (1.0 - config.confidence) / 2.0
        def bounds(values: np.ndarray) -> tuple[float, float]:
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if not len(finite):
                return np.nan, np.nan
            return float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))
        mean_low, mean_high = bounds(boot_mean)
        std_low, std_high = bounds(boot_standardized)
        positive_low, positive_high = bounds(boot_positive)
        output.append(
            {
                "metric": metric,
                "interface": interface,
                "mean_adjusted_gain": mean_estimate,
                "mean_adjusted_ci_low": mean_low,
                "mean_adjusted_ci_high": mean_high,
                "standardized_detectability": standardized_estimate,
                "standardized_ci_low": std_low,
                "standardized_ci_high": std_high,
                "positive_pair_rate": positive_estimate,
                "positive_pair_rate_ci_low": positive_low,
                "positive_pair_rate_ci_high": positive_high,
                "confidence": config.confidence,
                "n_boot": config.n_boot,
                "n_pairs": int(sum(len(array) for array in arrays)),
                "n_contrasts": len(arrays),
                "n_models": int(group["model_alias"].nunique()),
                "n_random_seeds": int(group["random_seed"].nunique()),
                "estimand": "equal_contrast_mean_over_model_seed_pair_effects",
            }
        )
    return pd.DataFrame(output)


def _exclude_exact_swap_contrasts(
    effects: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the contrast that is an algebraic sign flip under each transposition."""
    canonical_rows = inventory.loc[inventory["is_source_mapping"]]
    if len(canonical_rows) != 1:
        raise ValueError("Expected exactly one source mapping")
    canonical_order = str(canonical_rows.iloc[0]["option_order"]).split(",")
    canonical_position = {label: index for index, label in enumerate(canonical_order)}
    audit_rows = []
    for item in inventory.loc[~inventory["is_source_mapping"]].itertuples(index=False):
        current_order = str(item.option_order).split(",")
        current_position = {label: index for index, label in enumerate(current_order)}
        pair_types = sorted(
            effects.loc[effects["interface"].eq(item.interface), "pair_type"].unique()
        )
        for pair_type in pair_types:
            labels = str(pair_type).split("_vs_", maxsplit=1)
            if len(labels) != 2:
                raise ValueError(f"Cannot parse label contrast {pair_type!r}")
            lower, higher = labels
            exact_swap = bool(
                item.permutation_class == "transposition"
                and current_position[lower] == canonical_position[higher]
                and current_position[higher] == canonical_position[lower]
            )
            audit_rows.append(
                {
                    "interface": item.interface,
                    "mapping_name": item.mapping_name,
                    "option_order": item.option_order,
                    "permutation_class": item.permutation_class,
                    "pair_type": pair_type,
                    "is_exact_source_target_swap": exact_swap,
                    "sensitivity_includes_contrast": not exact_swap,
                }
            )
    audit = pd.DataFrame(audit_rows)
    if audit.empty:
        raise ValueError("No non-source mapping contrasts were available for sensitivity analysis")
    filtered = effects.merge(
        audit[["interface", "pair_type", "sensitivity_includes_contrast"]],
        on=["interface", "pair_type"],
        how="inner",
        validate="many_to_one",
    )
    filtered = filtered.loc[filtered["sensitivity_includes_contrast"]].drop(
        columns="sensitivity_includes_contrast"
    )
    return filtered.reset_index(drop=True), audit


def _decision_table(
    bootstrap: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    value_columns = [
        "mean_adjusted_gain",
        "mean_adjusted_ci_low",
        "mean_adjusted_ci_high",
        "standardized_detectability",
        "standardized_ci_low",
        "standardized_ci_high",
        "positive_pair_rate",
    ]
    wide = bootstrap.pivot_table(
        index="interface",
        columns="metric",
        values=value_columns,
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{value}" for value, metric in wide.columns]
    wide = wide.reset_index().merge(inventory, on="interface", how="left", validate="one_to_one")
    semantic_low = wide["semantic_target_margin__standardized_ci_low"]
    original_low = wide["original_slot_margin__standardized_ci_low"]
    difference_low = wide["semantic_minus_original_slot__mean_adjusted_ci_low"]
    difference_high = wide["semantic_minus_original_slot__mean_adjusted_ci_high"]
    wide["semantic_detected"] = semantic_low.gt(0.0)
    wide["original_slot_detected"] = original_low.gt(0.0)
    wide["semantic_exceeds_original_slot"] = difference_low.gt(0.0)
    wide["original_slot_exceeds_semantic"] = difference_high.lt(0.0)

    def signature(row: pd.Series) -> str:
        if row["semantic_exceeds_original_slot"] and row["semantic_detected"]:
            return "semantic_dominant"
        if row["original_slot_exceeds_semantic"] and row["original_slot_detected"]:
            return "original_slot_dominant"
        if row["semantic_detected"] and row["original_slot_detected"]:
            return "both_detected_no_clear_dominance"
        if row["semantic_detected"]:
            return "semantic_only_detected"
        if row["original_slot_detected"]:
            return "original_slot_only_detected"
        return "neither_detected"

    wide["evidence_signature"] = wide.apply(signature, axis=1)
    return wide.sort_values(
        ["is_source_mapping", "mapping_name"], ascending=[False, True]
    ).reset_index(drop=True)


def summarize_letter_permutation_audit(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    config = LetterPermutationStatisticsConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    rows_path = config.audit.audit.output_dir / "cross_interface_eval_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing completed letter-permutation rows: {rows_path}")
    rows = pd.read_csv(rows_path)
    inventory = _mapping_inventory(config)
    effects = _pair_adjusted_effects(rows, config)
    strata, per_seed, summary = _stratum_statistics(effects)
    bootstrap = _bootstrap_statistics(effects, config)
    decision = _decision_table(bootstrap, inventory)
    swap_filtered, swap_exclusion_audit = _exclude_exact_swap_contrasts(
        effects,
        inventory,
    )
    swap_exclusion_bootstrap = _bootstrap_statistics(swap_filtered, config)
    swap_exclusion_decision = _decision_table(swap_exclusion_bootstrap, inventory)
    tables = {
        "letter_permutation_mapping_inventory": inventory,
        "letter_permutation_pair_adjusted_effects": effects,
        "letter_permutation_by_stratum": strata,
        "letter_permutation_by_seed": per_seed,
        "letter_permutation_summary": summary,
        "letter_permutation_bootstrap_ci": bootstrap,
        "letter_permutation_decision": decision,
        "letter_permutation_exact_swap_exclusion_audit": swap_exclusion_audit,
        "letter_permutation_swap_exclusion_bootstrap_ci": swap_exclusion_bootstrap,
        "letter_permutation_swap_exclusion_decision": swap_exclusion_decision,
    }
    write_tables(tables, config.output_dir)
    return tables


def run_letter_permutation_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: list[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    """Run, aggregate, or summarize the exhaustive three-label audit."""
    from .cross_interface_audit import (
        CrossInterfaceConfig,
        aggregate_cross_interface_outputs,
        run_cross_interface_audit_from_json,
    )

    allowed = {"run", "aggregate", "summarize", "all"}
    if phase not in allowed:
        raise ValueError(f"Unsupported letter-permutation phase {phase!r}")

    tables: dict[str, pd.DataFrame] = {}
    if phase in {"run", "all"}:
        outputs = run_cross_interface_audit_from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
            model_aliases=model_aliases,
        )
        tables.update({f"audit__{key}": value for key, value in outputs.items()})
    if phase in {"aggregate", "summarize", "all"}:
        config = CrossInterfaceConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        outputs = aggregate_cross_interface_outputs(config)
        tables.update({f"audit__{key}": value for key, value in outputs.items()})
    if phase in {"summarize", "all"}:
        tables.update(
            summarize_letter_permutation_audit(
                config_path,
                project_root=project_root,
                model_source_overrides=model_source_overrides,
            )
        )
    return tables
