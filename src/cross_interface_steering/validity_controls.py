"""Submission-facing validity controls for cross-interface steering evidence."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .context_discrimination import build_context_pair_rows
from .io import write_tables
from .mapping_audit import _load_mapping_audit_config
from .published_caa_audit import BEHAVIOR_RUBRICS


RANDOM_MODE_PATTERN = re.compile(r"^random_direction_control_seed_(-?\d+)$")


@dataclass(frozen=True)
class ValidityControlsConfig:
    project_root: Path
    output_dir: Path
    multi_random_output_dir: Path
    target_pair_effects: Path
    target_context_rows: Path
    context_pair_metadata: Path
    published_caa_judgments: Path
    annotation_files: tuple[Path, Path]
    target_modes: tuple[str, ...]
    interfaces: tuple[str, ...]
    required_multipliers: tuple[float, ...]
    judge_sample_per_model_behavior: int
    minimum_base_gap: float
    n_boot: int
    confidence: float
    seed: int

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> "ValidityControlsConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        spec = dict(data.get("validity_controls", {}))
        output_dir = resolve(
            str(spec.get("output_dir", "outputs/validity_controls"))
        )
        multi_random_output = resolve(str(data.get("output_dir")))
        statistics = dict(spec.get("statistics", {}))
        confidence = float(statistics.get("confidence", 0.95))
        if not 0.0 < confidence < 1.0:
            raise ValueError("validity_controls.statistics.confidence must be between zero and one")
        annotation_values = spec.get(
            "annotation_files",
            [
                str(output_dir / "judge_validation_annotator_1.csv"),
                str(output_dir / "judge_validation_annotator_2.csv"),
            ],
        )
        if len(annotation_values) != 2:
            raise ValueError("validity_controls.annotation_files must contain exactly two paths")
        required_multipliers = tuple(
            float(value) for value in spec.get("required_multipliers", [-2.0, 0.0, 2.0])
        )
        if 0.0 not in required_multipliers or len(required_multipliers) < 2:
            raise ValueError("required_multipliers must contain baseline 0 and an intervention")
        return cls(
            project_root=root,
            output_dir=output_dir,
            multi_random_output_dir=multi_random_output,
            target_pair_effects=resolve(
                str(spec.get(
                    "target_pair_effects",
                    "outputs/normbank/nuisance_baselines/cross_interface_pair_effects.csv",
                ))
            ),
            target_context_rows=resolve(
                str(spec.get(
                    "target_context_rows",
                    "outputs/normbank/context_selectivity/context_discrimination_pair_rows.csv",
                ))
            ),
            context_pair_metadata=resolve(
                str(spec.get(
                    "context_pair_metadata",
                    "outputs/prepared/normbank/pairs.csv",
                ))
            ),
            published_caa_judgments=resolve(
                str(spec.get(
                    "published_caa_judgments",
                    "outputs/published_caa/published_caa_open_ended_judgments.csv",
                ))
            ),
            annotation_files=tuple(resolve(str(value)) for value in annotation_values),
            target_modes=tuple(str(value) for value in spec.get(
                "target_modes",
                ["raw_pre_answer", "mapping_balanced_direction", "label_only_direction"],
            )),
            interfaces=tuple(str(value) for value in spec.get(
                "interfaces",
                ["letter_canonical", "letter_reversed", "direct_label_completion", "opaque_completion"],
            )),
            required_multipliers=required_multipliers,
            judge_sample_per_model_behavior=int(spec.get("judge_sample_per_model_behavior", 10)),
            minimum_base_gap=float(spec.get("minimum_base_gap", 0.05)),
            n_boot=int(statistics.get("n_boot", 5_000)),
            confidence=confidence,
            seed=int(statistics.get("seed", 13)),
        )


def _read_required(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing validity-control input: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def _random_seed(mode: Any) -> int | None:
    match = RANDOM_MODE_PATTERN.match(str(mode))
    return int(match.group(1)) if match else None


def _safe_standardized(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return np.nan
    scale = float(np.std(array, ddof=1))
    return float(np.mean(array) / scale) if scale > 1e-12 else np.nan


def _bootstrap_bounds(
    estimates: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    finite = np.asarray(estimates, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    return float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))


def _bootstrap_multi_random_standardized_gain(
    pair_effects: pd.DataFrame,
    config: ValidityControlsConfig,
) -> pd.DataFrame:
    """Pair-bootstrap the five-draw, equal-model, equal-contrast standardized gain."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 101)
    for (target_mode, interface), group in pair_effects.groupby(
        ["target_mode", "interface"], sort=True
    ):
        contrast_arrays: list[np.ndarray] = []
        for _, contrast in group.groupby("pair_type", sort=True):
            pivot = contrast.pivot_table(
                index="pair_id",
                columns=["model_alias", "random_seed"],
                values="adjusted_gain",
                aggfunc="mean",
            ).dropna(axis=0, how="any")
            if len(pivot) >= 2:
                contrast_arrays.append(pivot.to_numpy(dtype=float))
        if not contrast_arrays:
            continue

        def standardized(matrix: np.ndarray) -> float:
            scales = np.std(matrix, axis=0, ddof=1)
            values = np.divide(
                np.mean(matrix, axis=0),
                scales,
                out=np.full_like(scales, np.nan, dtype=float),
                where=scales > 1e-12,
            )
            return float(np.nanmean(values))

        observed = float(np.mean([standardized(values) for values in contrast_arrays]))
        bootstrap = np.empty(config.n_boot, dtype=float)
        chunk_size = min(250, config.n_boot)
        for start in range(0, config.n_boot, chunk_size):
            stop = min(start + chunk_size, config.n_boot)
            chunk = np.zeros(stop - start, dtype=float)
            for values in contrast_arrays:
                indices = rng.integers(0, len(values), size=(stop - start, len(values)))
                sampled = values[indices]
                scales = np.std(sampled, axis=1, ddof=1)
                standardized_samples = np.divide(
                    np.mean(sampled, axis=1),
                    scales,
                    out=np.full_like(scales, np.nan, dtype=float),
                    where=scales > 1e-12,
                )
                finite_counts = np.isfinite(standardized_samples).sum(axis=1)
                stratum_means = np.divide(
                    np.nansum(standardized_samples, axis=1),
                    finite_counts,
                    out=np.full(stop - start, np.nan, dtype=float),
                    where=finite_counts > 0,
                )
                chunk += stratum_means / len(contrast_arrays)
            bootstrap[start:stop] = chunk
        low, high = _bootstrap_bounds(bootstrap, confidence=config.confidence)
        rows.append({
            "target_mode": target_mode,
            "interface": interface,
            "estimate": observed,
            "ci_low": low,
            "ci_high": high,
            "confidence": config.confidence,
            "n_boot": config.n_boot,
            "n_pairs": int(sum(len(values) for values in contrast_arrays)),
            "n_contrasts": len(contrast_arrays),
            "n_models": int(group["model_alias"].nunique()),
            "n_random_seeds": int(group["random_seed"].nunique()),
            "estimand": "equal_contrast_mean_of_model_seed_pair_standardized_gains",
        })
    return pd.DataFrame(rows)


def _bootstrap_multi_random_context_effects(
    pair_effects: pd.DataFrame,
    config: ValidityControlsConfig,
) -> pd.DataFrame:
    """Pair-bootstrap context effects after averaging nuisance draws and models."""
    averaged = (
        pair_effects.groupby(
            ["model_alias", "pair_id", "pair_type", "target_mode", "interface", "metric"],
            as_index=False,
        )["effect"].mean()
        .groupby(["pair_id", "pair_type", "target_mode", "interface", "metric"], as_index=False)["effect"].mean()
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 211)
    for (target_mode, interface, metric), group in averaged.groupby(
        ["target_mode", "interface", "metric"], sort=True
    ):
        contrast_arrays = [
            contrast["effect"].to_numpy(dtype=float)
            for _, contrast in group.groupby("pair_type", sort=True)
            if len(contrast)
        ]
        if not contrast_arrays:
            continue
        observed = float(np.mean([values.mean() for values in contrast_arrays]))
        bootstrap = np.zeros(config.n_boot, dtype=float)
        chunk_size = min(500, config.n_boot)
        for start in range(0, config.n_boot, chunk_size):
            stop = min(start + chunk_size, config.n_boot)
            chunk = np.zeros(stop - start, dtype=float)
            for values in contrast_arrays:
                indices = rng.integers(0, len(values), size=(stop - start, len(values)))
                chunk += values[indices].mean(axis=1) / len(contrast_arrays)
            bootstrap[start:stop] = chunk
        low, high = _bootstrap_bounds(bootstrap, confidence=config.confidence)
        rows.append({
            "target_mode": target_mode,
            "interface": interface,
            "metric": metric,
            "estimate": observed,
            "ci_low": low,
            "ci_high": high,
            "confidence": config.confidence,
            "n_boot": config.n_boot,
            "n_pairs": int(sum(len(values) for values in contrast_arrays)),
            "n_contrasts": len(contrast_arrays),
            "n_random_seeds": int(pair_effects["random_seed"].nunique()),
            "estimand": "equal_contrast_pair_bootstrap_after_seed_and_model_averaging",
        })
    return pd.DataFrame(rows)


def _load_context_pair_metadata(
    config: ValidityControlsConfig,
) -> pd.DataFrame:
    """Load RQ2 clustering metadata and identify invalid identical-text pairs."""
    metadata = _read_required(
        config.context_pair_metadata,
        {"pair_id", "pair_type", "group_key", "negative_text", "positive_text"},
    ).copy()
    metadata = metadata[
        ["pair_id", "pair_type", "group_key", "negative_text", "positive_text"]
    ].drop_duplicates(["pair_id", "pair_type"])
    metadata["identical_endpoint_text"] = (
        metadata["negative_text"].fillna("").astype(str).str.strip()
        == metadata["positive_text"].fillna("").astype(str).str.strip()
    )
    metadata["context_cluster_id"] = metadata["group_key"].fillna("").astype(str)
    if metadata["context_cluster_id"].eq("").any():
        raise ValueError("Context pair metadata contains empty group_key values")
    return metadata


def _bootstrap_multi_random_context_effects_by_cluster(
    pair_effects: pd.DataFrame,
    config: ValidityControlsConfig,
) -> pd.DataFrame:
    """Cluster-bootstrap RQ2 effects by setting--behavior group.

    The point estimand remains the equal-contrast mean. Resampling complete
    setting--behavior groups preserves dependence from groups represented in
    more than one contrast.
    """
    required = {"context_cluster_id"}
    if not required.issubset(pair_effects.columns):
        raise ValueError("Cluster bootstrap requires context_cluster_id metadata")
    averaged = (
        pair_effects.groupby(
            [
                "model_alias", "pair_id", "pair_type", "context_cluster_id",
                "target_mode", "interface", "metric",
            ],
            as_index=False,
        )["effect"].mean()
        .groupby(
            [
                "pair_id", "pair_type", "context_cluster_id",
                "target_mode", "interface", "metric",
            ],
            as_index=False,
        )["effect"].mean()
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 257)
    for (target_mode, interface, metric), group in averaged.groupby(
        ["target_mode", "interface", "metric"], sort=True
    ):
        contrasts = sorted(group["pair_type"].unique())
        clusters = sorted(group["context_cluster_id"].unique())
        if not contrasts or not clusters:
            continue
        contrast_index = {value: index for index, value in enumerate(contrasts)}
        cluster_index = {value: index for index, value in enumerate(clusters)}
        sums = np.zeros((len(clusters), len(contrasts)), dtype=float)
        counts = np.zeros((len(clusters), len(contrasts)), dtype=float)
        for row in group.itertuples(index=False):
            i = cluster_index[row.context_cluster_id]
            j = contrast_index[row.pair_type]
            sums[i, j] += float(row.effect)
            counts[i, j] += 1.0
        contrast_counts = counts.sum(axis=0)
        observed_contrast_means = np.divide(
            sums.sum(axis=0), contrast_counts,
            out=np.full(len(contrasts), np.nan), where=contrast_counts > 0,
        )
        observed = float(np.nanmean(observed_contrast_means))
        bootstrap = np.zeros(config.n_boot, dtype=float)
        chunk_size = min(500, config.n_boot)
        probabilities = np.full(len(clusters), 1.0 / len(clusters))
        for start in range(0, config.n_boot, chunk_size):
            stop = min(start + chunk_size, config.n_boot)
            weights = rng.multinomial(len(clusters), probabilities, size=stop - start)
            sampled_sums = weights @ sums
            sampled_counts = weights @ counts
            sampled_means = np.divide(
                sampled_sums,
                sampled_counts,
                out=np.full_like(sampled_sums, np.nan),
                where=sampled_counts > 0,
            )
            bootstrap[start:stop] = np.nanmean(sampled_means, axis=1)
        low, high = _bootstrap_bounds(bootstrap, confidence=config.confidence)
        rows.append({
            "target_mode": target_mode,
            "interface": interface,
            "metric": metric,
            "estimate": observed,
            "ci_low": low,
            "ci_high": high,
            "confidence": config.confidence,
            "n_boot": config.n_boot,
            "n_pairs": int(group["pair_id"].nunique()),
            "n_clusters": len(clusters),
            "n_contrasts": len(contrasts),
            "n_random_seeds": int(pair_effects["random_seed"].nunique()),
            "estimand": (
                "equal_contrast_setting_behavior_cluster_bootstrap_"
                "after_seed_and_model_averaging"
            ),
        })
    return pd.DataFrame(rows)


def _bootstrap_context_baselines(
    target_rows: pd.DataFrame,
    config: ValidityControlsConfig,
) -> pd.DataFrame:
    """Summarize unsteered strict-pair discrimination on the RQ2 test set."""
    keys = [
        "model_alias", "model_name", "pair_id", "pair_type", "interface", "template",
    ]
    base = target_rows.groupby(keys, as_index=False).agg(
        base_discrimination_gap=("base_discrimination_gap", "mean"),
        base_rank_accuracy=("base_rank_correct", "mean"),
        base_gap_spread=("base_discrimination_gap", lambda values: float(values.max() - values.min())),
        base_rank_spread=(
            "base_rank_correct",
            lambda values: float(values.astype(float).max() - values.astype(float).min()),
        ),
    )
    if (base[["base_gap_spread", "base_rank_spread"]].to_numpy(dtype=float) > 1e-6).any():
        raise ValueError("Unsteered context baselines disagree across target direction modes")
    base = (
        base.groupby(
            ["model_alias", "model_name", "pair_id", "pair_type", "interface"],
            as_index=False,
        )[["base_discrimination_gap", "base_rank_accuracy"]]
        .mean()
    )
    model_averaged = (
        base.groupby(["pair_id", "pair_type", "interface"], as_index=False)[
            ["base_discrimination_gap", "base_rank_accuracy"]
        ]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 307)
    for interface, group in model_averaged.groupby("interface", sort=True):
        for metric in ("base_discrimination_gap", "base_rank_accuracy"):
            contrast_arrays = [
                contrast[metric].to_numpy(dtype=float)
                for _, contrast in group.groupby("pair_type", sort=True)
                if len(contrast)
            ]
            observed = float(np.mean([values.mean() for values in contrast_arrays]))
            bootstrap = np.zeros(config.n_boot, dtype=float)
            for values in contrast_arrays:
                indices = rng.integers(0, len(values), size=(config.n_boot, len(values)))
                bootstrap += values[indices].mean(axis=1) / len(contrast_arrays)
            low, high = _bootstrap_bounds(bootstrap, confidence=config.confidence)
            rows.append({
                "interface": interface,
                "metric": metric,
                "estimate": observed,
                "ci_low": low,
                "ci_high": high,
                "confidence": config.confidence,
                "n_boot": config.n_boot,
                "n_pairs": int(sum(len(values) for values in contrast_arrays)),
                "n_models": int(base["model_alias"].nunique()),
                "n_contrasts": len(contrast_arrays),
                "estimand": "fixed_model_equal_contrast_pair_bootstrap_after_template_averaging",
            })
    return pd.DataFrame(rows)


def _bootstrap_multi_random_context_per_model(
    pair_effects: pd.DataFrame,
    config: ValidityControlsConfig,
) -> pd.DataFrame:
    """Expose model heterogeneity after averaging the five random controls."""
    averaged = (
        pair_effects.groupby(
            [
                "model_alias", "model_name", "pair_id", "pair_type",
                "target_mode", "interface", "metric",
            ],
            as_index=False,
        )["effect"]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 401)
    for keys, group in averaged.groupby(
        ["model_alias", "model_name", "target_mode", "interface", "metric"],
        sort=True,
    ):
        model_alias, model_name, target_mode, interface, metric = keys
        contrast_arrays = [
            contrast["effect"].to_numpy(dtype=float)
            for _, contrast in group.groupby("pair_type", sort=True)
            if len(contrast)
        ]
        observed = float(np.mean([values.mean() for values in contrast_arrays]))
        bootstrap = np.zeros(config.n_boot, dtype=float)
        for values in contrast_arrays:
            indices = rng.integers(0, len(values), size=(config.n_boot, len(values)))
            bootstrap += values[indices].mean(axis=1) / len(contrast_arrays)
        low, high = _bootstrap_bounds(bootstrap, confidence=config.confidence)
        rows.append({
            "model_alias": model_alias,
            "model_name": model_name,
            "target_mode": target_mode,
            "interface": interface,
            "metric": metric,
            "estimate": observed,
            "ci_low": low,
            "ci_high": high,
            "confidence": config.confidence,
            "n_boot": config.n_boot,
            "n_pairs": int(sum(len(values) for values in contrast_arrays)),
            "n_contrasts": len(contrast_arrays),
            "n_random_seeds": int(pair_effects["random_seed"].nunique()),
            "estimand": "fixed_model_equal_contrast_pair_bootstrap_after_seed_averaging",
        })
    return pd.DataFrame(rows)


def build_multi_random_margin_statistics(config: ValidityControlsConfig) -> dict[str, pd.DataFrame]:
    target = _read_required(
        config.target_pair_effects,
        {"model_alias", "model_name", "pair_id", "pair_type", "mode", "interface", "template", "target_margin_gain"},
    )
    random_path = config.multi_random_output_dir / "cross_interface_pair_effects.csv"
    random = _read_required(
        random_path,
        {"model_alias", "model_name", "pair_id", "pair_type", "mode", "interface", "template", "target_margin_gain"},
    )
    target = target[
        target["mode"].isin(config.target_modes) & target["interface"].isin(config.interfaces)
    ].copy()
    random["random_seed"] = random["mode"].map(_random_seed)
    random = random[
        random["random_seed"].notna() & random["interface"].isin(config.interfaces)
    ].copy()
    if random.empty:
        raise ValueError("No random_direction_control_seed_* rows were found")
    random["random_seed"] = random["random_seed"].astype(int)

    pair_keys = ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface"]
    target_pair = target.groupby(pair_keys, as_index=False).agg(
        target_gain=("target_margin_gain", "mean"),
        n_target_templates=("template", "nunique"),
    )
    random_pair = random.groupby([*pair_keys, "random_seed"], as_index=False).agg(
        random_gain=("target_margin_gain", "mean"),
        n_random_templates=("template", "nunique"),
    ).rename(columns={"mode": "random_mode"})
    merged = target_pair.merge(
        random_pair,
        on=["model_alias", "model_name", "pair_id", "pair_type", "interface"],
        how="inner",
        validate="many_to_many",
    ).rename(columns={"mode": "target_mode"})
    merged["adjusted_gain"] = merged["target_gain"] - merged["random_gain"]

    strata_keys = ["model_alias", "model_name", "pair_type", "target_mode", "interface", "random_seed"]
    strata = (
        merged.groupby(strata_keys, as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_adjusted_gain=("adjusted_gain", "mean"),
            adjusted_gain_sd=("adjusted_gain", "std"),
            positive_pair_rate=("adjusted_gain", lambda values: float((values > 0).mean())),
        )
    )
    strata["paired_standardized_gain"] = (
        strata["mean_adjusted_gain"] / strata["adjusted_gain_sd"]
    )
    per_seed = (
        strata.groupby(["target_mode", "interface", "random_seed"], as_index=False)
        .agg(
            n_strata=("pair_type", "size"),
            n_models=("model_alias", "nunique"),
            n_contrasts=("pair_type", "nunique"),
            mean_adjusted_gain=("mean_adjusted_gain", "mean"),
            mean_paired_standardized_gain=("paired_standardized_gain", "mean"),
            mean_positive_pair_rate=("positive_pair_rate", "mean"),
            n_positive_strata=("mean_adjusted_gain", lambda values: int((values > 0).sum())),
        )
    )
    across_seed = (
        per_seed.groupby(["target_mode", "interface"], as_index=False)
        .agg(
            n_random_seeds=("random_seed", "nunique"),
            adjusted_gain_seed_mean=("mean_adjusted_gain", "mean"),
            adjusted_gain_seed_sd=("mean_adjusted_gain", "std"),
            adjusted_gain_seed_min=("mean_adjusted_gain", "min"),
            adjusted_gain_seed_max=("mean_adjusted_gain", "max"),
            standardized_gain_seed_mean=("mean_paired_standardized_gain", "mean"),
            standardized_gain_seed_sd=("mean_paired_standardized_gain", "std"),
            standardized_gain_seed_min=("mean_paired_standardized_gain", "min"),
            standardized_gain_seed_max=("mean_paired_standardized_gain", "max"),
            min_positive_strata=("n_positive_strata", "min"),
            max_positive_strata=("n_positive_strata", "max"),
        )
    )
    across_seed["all_seed_means_positive"] = across_seed["adjusted_gain_seed_min"].gt(0.0)
    return {
        "multi_random_pair_adjusted_effects": merged,
        "multi_random_standardized_gain_by_stratum": strata,
        "multi_random_standardized_gain_by_seed": per_seed,
        "multi_random_standardized_gain_summary": across_seed,
        "multi_random_standardized_gain_bootstrap_ci": _bootstrap_multi_random_standardized_gain(
            merged, config
        ),
    }


def build_multi_random_context_statistics(config: ValidityControlsConfig) -> dict[str, pd.DataFrame]:
    pair_metadata = _load_context_pair_metadata(config)
    target = _read_required(
        config.target_context_rows,
        {
            "model_alias", "model_name", "pair_id", "pair_type", "mode", "interface",
            "template", "base_discrimination_gap", "base_rank_correct",
            "discrimination_gap_change", "patched_rank_correct",
        },
    )
    random_eval = _read_required(
        config.multi_random_output_dir / "cross_interface_eval_rows.csv",
        {"model_alias", "model_name", "pair_id", "pair_type", "endpoint", "evaluation_policy", "mode", "interface", "template", "base_high_low_margin", "patched_high_low_margin"},
    )
    random_eval = random_eval[
        random_eval["evaluation_policy"].eq("context_plus")
        & random_eval["mode"].map(_random_seed).notna()
        & random_eval["interface"].isin(config.interfaces)
    ].copy()
    random_context = build_context_pair_rows(
        random_eval,
        minimum_base_gap=config.minimum_base_gap,
    )
    random_context["random_seed"] = random_context["mode"].map(_random_seed).astype(int)
    target = target[
        target["mode"].isin(config.target_modes) & target["interface"].isin(config.interfaces)
    ].copy()
    input_pairs = target[["pair_id", "pair_type"]].drop_duplicates()
    audit_metadata = input_pairs.merge(
        pair_metadata,
        on=["pair_id", "pair_type"],
        how="left",
        validate="one_to_one",
    )
    if audit_metadata["context_cluster_id"].isna().any():
        raise ValueError("RQ2 input pairs could not be matched to NormBank pair metadata")
    exclusions = audit_metadata.loc[audit_metadata["identical_endpoint_text"]].copy()
    excluded_ids = set(exclusions["pair_id"].astype(str))
    target = target.loc[~target["pair_id"].astype(str).isin(excluded_ids)].copy()
    random_context = random_context.loc[
        ~random_context["pair_id"].astype(str).isin(excluded_ids)
    ].copy()
    retained_metadata = audit_metadata.loc[~audit_metadata["identical_endpoint_text"]]
    filter_audit = pd.DataFrame([
        {
            "scope": "rq2_matched_context_diagnostic",
            "n_input_pairs": int(len(audit_metadata)),
            "n_excluded_identical_text_pairs": int(len(exclusions)),
            "n_retained_pairs": int(len(retained_metadata)),
            "n_retained_setting_behavior_clusters": int(
                retained_metadata["context_cluster_id"].nunique()
            ),
            "cluster_key": "setting_plus_behavior_group_key",
            "exclusion_rule": "normalized rendered negative_text equals positive_text",
        }
    ])
    keys = ["model_alias", "model_name", "pair_id", "pair_type", "interface"]
    metrics = ("discrimination_gap_change", "patched_rank_correct")
    target_pair = target.groupby([*keys, "mode"], as_index=False).agg(
        **{f"target_{metric}": (metric, "mean") for metric in metrics}
    ).rename(columns={"mode": "target_mode"})
    random_pair = random_context.groupby([*keys, "mode", "random_seed"], as_index=False).agg(
        **{f"random_{metric}": (metric, "mean") for metric in metrics}
    ).rename(columns={"mode": "random_mode"})
    merged = target_pair.merge(random_pair, on=keys, how="inner", validate="many_to_many")
    long_rows: list[pd.DataFrame] = []
    for metric in metrics:
        part = merged[[*keys, "target_mode", "random_mode", "random_seed"]].copy()
        part["metric"] = metric
        part["effect"] = merged[f"target_{metric}"] - merged[f"random_{metric}"]
        long_rows.append(part)
    long = pd.concat(long_rows, ignore_index=True)
    long = long.merge(
        pair_metadata[["pair_id", "pair_type", "context_cluster_id"]],
        on=["pair_id", "pair_type"],
        how="left",
        validate="many_to_one",
    )
    if long["context_cluster_id"].isna().any():
        raise ValueError("RQ2 effects could not be matched to NormBank group metadata")
    stratum = (
        long.groupby(["model_alias", "model_name", "pair_type", "target_mode", "interface", "random_seed", "metric"], as_index=False)
        .agg(n_pairs=("pair_id", "nunique"), estimate=("effect", "mean"))
    )
    per_seed = (
        stratum.groupby(["target_mode", "interface", "random_seed", "metric"], as_index=False)
        .agg(n_strata=("pair_type", "size"), estimate=("estimate", "mean"))
    )
    across_seed = (
        per_seed.groupby(["target_mode", "interface", "metric"], as_index=False)
        .agg(
            n_random_seeds=("random_seed", "nunique"),
            seed_mean=("estimate", "mean"),
            seed_sd=("estimate", "std"),
            seed_min=("estimate", "min"),
            seed_max=("estimate", "max"),
            n_positive_seeds=("estimate", lambda values: int((values > 0).sum())),
            n_negative_seeds=("estimate", lambda values: int((values < 0).sum())),
        )
    )
    return {
        "context_pair_filter_audit": filter_audit,
        "context_pair_exclusions": exclusions,
        "multi_random_context_pair_effects": long,
        "multi_random_context_by_stratum": stratum,
        "multi_random_context_by_seed": per_seed,
        "multi_random_context_summary": across_seed,
        "multi_random_context_bootstrap_ci": _bootstrap_multi_random_context_effects(long, config),
        "multi_random_context_cluster_bootstrap_ci": (
            _bootstrap_multi_random_context_effects_by_cluster(long, config)
        ),
        "multi_random_context_baseline_bootstrap_ci": _bootstrap_context_baselines(target, config),
        "multi_random_context_per_model_bootstrap_ci": _bootstrap_multi_random_context_per_model(
            long, config
        ),
    }


def prepare_judge_validation(config: ValidityControlsConfig) -> dict[str, pd.DataFrame]:
    judgments = _read_required(
        config.published_caa_judgments,
        {"model_alias", "model_name", "behavior", "item_id", "multiplier", "question", "response", "judge_score"},
    )
    judgments["multiplier"] = pd.to_numeric(judgments["multiplier"], errors="coerce")
    judgments["judge_score"] = pd.to_numeric(judgments["judge_score"], errors="coerce")
    judgments = judgments[
        judgments["multiplier"].isin(config.required_multipliers)
        & judgments["judge_score"].notna()
    ].copy()
    rng = np.random.default_rng(config.seed)
    selected: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for (model_alias, behavior), group in judgments.groupby(["model_alias", "behavior"], sort=True):
        multiplier_sets = group.groupby("item_id")["multiplier"].agg(lambda values: set(values))
        eligible = sorted(
            item_id for item_id, values in multiplier_sets.items()
            if set(config.required_multipliers).issubset(values)
        )
        n_select = min(config.judge_sample_per_model_behavior, len(eligible))
        chosen = sorted(rng.choice(eligible, size=n_select, replace=False).tolist()) if n_select else []
        selected.append(group[group["item_id"].isin(chosen)].copy())
        inventory.append(
            {
                "model_alias": model_alias,
                "behavior": behavior,
                "n_eligible_items": len(eligible),
                "n_sampled_items": n_select,
                "n_sampled_responses": n_select * len(config.required_multipliers),
            }
        )
    sample = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()
    if sample.empty:
        raise ValueError("No complete model-behavior-multiplier groups are available for judge validation")
    sample["annotation_id"] = sample.apply(
        lambda row: hashlib.sha256(
            f"{row['model_alias']}|{row['behavior']}|{row['item_id']}|{row['multiplier']}".encode("utf-8")
        ).hexdigest()[:16],
        axis=1,
    )
    sample["rubric"] = sample["behavior"].map(BEHAVIOR_RUBRICS)
    key_columns = [
        "annotation_id", "model_alias", "model_name", "behavior", "item_id",
        "multiplier", "judge_score",
    ]
    key = sample[key_columns].sort_values("annotation_id").reset_index(drop=True)
    form_columns = ["annotation_id", "behavior", "rubric", "question", "response"]
    base_form = sample[form_columns].copy()
    base_form["human_score"] = np.nan
    base_form["human_notes"] = ""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for annotator_index, path in enumerate(config.annotation_files):
        if path.exists():
            existing = pd.read_csv(path)
            if "human_score" in existing and existing["human_score"].notna().any():
                continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shuffled = base_form.sample(frac=1.0, random_state=config.seed + annotator_index + 1)
        shuffled.to_csv(path, index=False)
    tables = {
        "judge_validation_sample_key": key,
        "judge_validation_sampling_inventory": pd.DataFrame(inventory),
        "judge_validation_instructions": pd.DataFrame([
            {
                "instruction": (
                    "Two annotators independently fill human_score with a number from 0 to 10. "
                    "Do not change annotation_id, behavior, rubric, question, or response. "
                    "Annotators must not inspect judge_validation_sample_key.csv before scoring."
                ),
                "annotator_1_file": str(config.annotation_files[0]),
                "annotator_2_file": str(config.annotation_files[1]),
            }
        ]),
    }
    write_tables(tables, config.output_dir)
    return tables


def _quadratic_weighted_kappa(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(np.rint(left).astype(int), 0, 10)
    right = np.clip(np.rint(right).astype(int), 0, 10)
    if not len(left):
        return np.nan
    observed = np.zeros((11, 11), dtype=float)
    for first, second in zip(left, right):
        observed[first, second] += 1.0
    expected = np.outer(np.bincount(left, minlength=11), np.bincount(right, minlength=11)) / len(left)
    indices = np.arange(11, dtype=float)
    weights = np.square(indices[:, None] - indices[None, :]) / 100.0
    denominator = float(np.sum(weights * expected))
    return float(1.0 - np.sum(weights * observed) / denominator) if denominator > 1e-12 else np.nan


def _agreement_record(scope: str, left: pd.Series, right: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if pair.empty:
        return {"scope": scope, "n_items": 0}
    delta = pair["left"] - pair["right"]
    # Ranking then applying Pearson is exactly Spearman correlation and avoids
    # making the CPU-only validity package depend on scipy.
    spearman = pair["left"].rank(method="average").corr(
        pair["right"].rank(method="average"), method="pearson"
    )
    return {
        "scope": scope,
        "n_items": len(pair),
        "pearson_r": pair["left"].corr(pair["right"], method="pearson"),
        "spearman_rho": spearman,
        "mean_absolute_difference": delta.abs().mean(),
        "exact_rounded_agreement": np.rint(pair["left"]).eq(np.rint(pair["right"])).mean(),
        "within_one_point_rate": delta.abs().le(1.0).mean(),
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(
            pair["left"].to_numpy(dtype=float), pair["right"].to_numpy(dtype=float)
        ),
    }


def _bootstrap_mean(values: np.ndarray, *, n_boot: int, confidence: float, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    estimates = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))


def summarize_judge_validation(config: ValidityControlsConfig) -> dict[str, pd.DataFrame]:
    key_path = config.output_dir / "judge_validation_sample_key.csv"
    key = _read_required(
        key_path,
        {"annotation_id", "model_alias", "model_name", "behavior", "item_id", "multiplier", "judge_score"},
    )
    annotations: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    for index, path in enumerate(config.annotation_files, start=1):
        if not path.exists():
            status_rows.append({"annotator": index, "file": str(path), "status": "missing", "n_scored": 0})
            continue
        frame = _read_required(path, {"annotation_id", "human_score"})
        frame["human_score"] = pd.to_numeric(frame["human_score"], errors="coerce")
        invalid = frame["human_score"].notna() & ~frame["human_score"].between(0.0, 10.0)
        if invalid.any():
            raise ValueError(f"{path} contains human_score values outside [0, 10]")
        score_name = f"human_score_{index}"
        annotations.append(frame[["annotation_id", "human_score"]].rename(columns={"human_score": score_name}))
        status_rows.append({
            "annotator": index,
            "file": str(path),
            "status": "complete" if frame["human_score"].notna().all() else "partial",
            "n_scored": int(frame["human_score"].notna().sum()),
            "n_expected": len(key),
        })
    status = pd.DataFrame(status_rows)
    if len(annotations) < 2:
        return {"judge_validation_status": status}
    merged = key.copy()
    for frame in annotations:
        merged = merged.merge(frame, on="annotation_id", how="left", validate="one_to_one")
    complete = merged.dropna(subset=["human_score_1", "human_score_2", "judge_score"]).copy()
    if complete.empty:
        return {"judge_validation_status": status, "judge_validation_scored_rows": merged}
    complete["human_mean_score"] = complete[["human_score_1", "human_score_2"]].mean(axis=1)

    agreement_rows = [
        _agreement_record("human_1_vs_human_2", complete["human_score_1"], complete["human_score_2"]),
        _agreement_record("llm_judge_vs_human_mean", complete["judge_score"], complete["human_mean_score"]),
    ]
    for behavior, group in complete.groupby("behavior", sort=True):
        record = _agreement_record(
            f"llm_judge_vs_human_mean:{behavior}", group["judge_score"], group["human_mean_score"]
        )
        agreement_rows.append(record)

    paired_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 500)
    for (model_alias, model_name, behavior), group in complete.groupby(
        ["model_alias", "model_name", "behavior"], sort=True
    ):
        pivot_human = group.pivot_table(index="item_id", columns="multiplier", values="human_mean_score", aggfunc="first")
        pivot_llm = group.pivot_table(index="item_id", columns="multiplier", values="judge_score", aggfunc="first")
        for multiplier in config.required_multipliers:
            if multiplier == 0.0 or multiplier not in pivot_human or 0.0 not in pivot_human:
                continue
            item_ids = pivot_human[[0.0, multiplier]].dropna().index.intersection(
                pivot_llm[[0.0, multiplier]].dropna().index
            )
            for source, pivot in (("human_mean", pivot_human), ("llm_judge", pivot_llm)):
                values = (pivot.loc[item_ids, multiplier] - pivot.loc[item_ids, 0.0]).to_numpy(dtype=float)
                estimate, low, high = _bootstrap_mean(
                    values, n_boot=config.n_boot, confidence=config.confidence, rng=rng
                )
                verdict = "positive" if low > 0 else "negative" if high < 0 else "inconclusive"
                paired_rows.append({
                    "model_alias": model_alias,
                    "model_name": model_name,
                    "behavior": behavior,
                    "multiplier": multiplier,
                    "score_source": source,
                    "n_paired_items": len(values),
                    "paired_effect": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "verdict": verdict,
                })
    paired = pd.DataFrame(paired_rows)
    verdict = pd.DataFrame()
    if not paired.empty:
        verdict = paired.pivot_table(
            index=["model_alias", "model_name", "behavior", "multiplier"],
            columns="score_source",
            values="verdict",
            aggfunc="first",
        ).reset_index()
        if {"human_mean", "llm_judge"}.issubset(verdict.columns):
            verdict["verdict_agrees"] = verdict["human_mean"].eq(verdict["llm_judge"])
    return {
        "judge_validation_status": status,
        "judge_validation_scored_rows": complete,
        "judge_validation_agreement": pd.DataFrame(agreement_rows),
        "judge_validation_paired_effects": paired,
        "judge_validation_verdict_agreement": verdict,
    }


def run_validity_control_synthesis(
    config: ValidityControlsConfig,
    *,
    include_judge: bool = True,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables.update(build_multi_random_margin_statistics(config))
    tables.update(build_multi_random_context_statistics(config))
    competence_path = config.multi_random_output_dir / "cross_interface_key_competence_summary.csv"
    if competence_path.exists():
        tables["normbank_key_competence_summary"] = pd.read_csv(competence_path)
    if include_judge and (config.output_dir / "judge_validation_sample_key.csv").exists():
        tables.update(summarize_judge_validation(config))
    write_tables(tables, config.output_dir)
    return tables


def run_evaluation_validity_controls_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: list[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    """Run random controls, prepare judge forms, or summarize validity checks."""
    from .cross_interface_audit import (
        CrossInterfaceConfig,
        aggregate_cross_interface_outputs,
        run_cross_interface_audit_from_json,
    )

    allowed = {"random", "prepare-judge", "summarize", "all"}
    if phase not in allowed:
        raise ValueError(f"Unsupported validity-control phase {phase!r}")

    tables: dict[str, pd.DataFrame] = {}
    if phase in {"random", "all"}:
        outputs = run_cross_interface_audit_from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
            model_aliases=model_aliases,
        )
        tables.update({f"random_audit__{key}": value for key, value in outputs.items()})

    config = ValidityControlsConfig.from_json(config_path, project_root=project_root)
    if phase in {"prepare-judge", "all"}:
        tables.update(prepare_judge_validation(config))
    if phase in {"summarize", "all"}:
        random_config = CrossInterfaceConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        aggregate_cross_interface_outputs(random_config)
        tables.update(run_validity_control_synthesis(config))
    return tables
