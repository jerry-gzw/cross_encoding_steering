"""Submission-facing validity controls for cross-encoding steering evidence."""
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
    published_caa_multi_judge_scores: Path
    annotation_files: tuple[Path, Path]
    adjudication_file: Path
    panel_judge_aliases: tuple[str, ...]
    target_modes: tuple[str, ...]
    interfaces: tuple[str, ...]
    required_multipliers: tuple[float, ...]
    judge_sample_per_model_behavior: int
    judge_pair_items_across_models: bool
    adjudication_score_gap: int
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
            str(spec.get("output_dir", "outputs/evaluation_validity_controls"))
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
        adjudication_score_gap = int(spec.get("adjudication_score_gap", 3))
        if adjudication_score_gap < 1:
            raise ValueError("validity_controls.adjudication_score_gap must be positive")
        return cls(
            project_root=root,
            output_dir=output_dir,
            multi_random_output_dir=multi_random_output,
            target_pair_effects=resolve(
                str(spec.get(
                    "target_pair_effects",
                    "outputs/interface_nuisance_baselines/normbank/cross_interface_pair_effects.csv",
                ))
            ),
            target_context_rows=resolve(
                str(spec.get(
                    "target_context_rows",
                    "outputs/context_conditioned_discrimination/normbank/context_discrimination_pair_rows.csv",
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
                    "outputs/published_caa_protocol_audit/published_caa_open_ended_judgments.csv",
                ))
            ),
            published_caa_multi_judge_scores=resolve(
                str(spec.get(
                    "published_caa_multi_judge_scores",
                    "outputs/published_caa_protocol_audit/published_caa_multi_judge_scores.csv",
                ))
            ),
            annotation_files=tuple(resolve(str(value)) for value in annotation_values),
            adjudication_file=resolve(
                str(spec.get(
                    "adjudication_file",
                    output_dir / "adjudicator_blind.csv",
                ))
            ),
            panel_judge_aliases=tuple(str(value) for value in spec.get(
                "panel_judge_aliases",
                [
                    "gpt5_1_rubric_v3",
                    "claude_sonnet_4_6_rubric_v3",
                    "gemini_3_5_flash_rubric_v3_batch10",
                ],
            )),
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
            judge_pair_items_across_models=bool(
                spec.get("judge_pair_items_across_models", False)
            ),
            adjudication_score_gap=adjudication_score_gap,
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
    if config.judge_pair_items_across_models:
        model_aliases = tuple(sorted(judgments["model_alias"].astype(str).unique()))
        required = set(config.required_multipliers)
        for behavior, group in judgments.groupby("behavior", sort=True):
            complete_by_model = {}
            for model_alias, model_group in group.groupby("model_alias", sort=True):
                multiplier_sets = model_group.groupby("item_id")["multiplier"].agg(
                    lambda values: set(values)
                )
                complete_by_model[str(model_alias)] = {
                    str(item_id)
                    for item_id, values in multiplier_sets.items()
                    if required.issubset(values)
                }
            eligible = sorted(
                set.intersection(
                    *(complete_by_model.get(alias, set()) for alias in model_aliases)
                )
            )
            n_select = min(config.judge_sample_per_model_behavior, len(eligible))
            chosen = (
                sorted(rng.choice(eligible, size=n_select, replace=False).tolist())
                if n_select
                else []
            )
            selected.append(group[group["item_id"].astype(str).isin(chosen)].copy())
            inventory.append(
                {
                    "model_alias": "__paired_across_models__",
                    "behavior": behavior,
                    "n_models": len(model_aliases),
                    "n_eligible_items": len(eligible),
                    "n_sampled_items": n_select,
                    "n_sampled_responses": (
                        n_select * len(model_aliases) * len(config.required_multipliers)
                    ),
                    "sampling_unit": "behavior_item_block_shared_across_models",
                }
            )
    else:
        for (model_alias, behavior), group in judgments.groupby(
            ["model_alias", "behavior"], sort=True
        ):
            multiplier_sets = group.groupby("item_id")["multiplier"].agg(
                lambda values: set(values)
            )
            eligible = sorted(
                item_id
                for item_id, values in multiplier_sets.items()
                if set(config.required_multipliers).issubset(values)
            )
            n_select = min(config.judge_sample_per_model_behavior, len(eligible))
            chosen = (
                sorted(rng.choice(eligible, size=n_select, replace=False).tolist())
                if n_select
                else []
            )
            selected.append(group[group["item_id"].isin(chosen)].copy())
            inventory.append(
                {
                    "model_alias": model_alias,
                    "behavior": behavior,
                    "n_models": 1,
                    "n_eligible_items": len(eligible),
                    "n_sampled_items": n_select,
                    "n_sampled_responses": (
                        n_select * len(config.required_multipliers)
                    ),
                    "sampling_unit": "model_behavior_item_block",
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
    base_form["evidence_span"] = ""
    base_form["confidence"] = ""
    base_form["uncertainty_reason"] = ""
    base_form["adjudication_flag"] = ""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for annotator_index, path in enumerate(config.annotation_files):
        if path.exists():
            existing = pd.read_csv(path)
            if "human_score" in existing and existing["human_score"].notna().any():
                continue
        path.parent.mkdir(parents=True, exist_ok=True)
        ordering = sample[["behavior", "item_id"]].copy()
        ordering["block_id"] = (
            ordering["behavior"].astype(str) + "|" + ordering["item_id"].astype(str)
        )
        ordering["_within_block"] = ordering.groupby("block_id", sort=True).cumcount()
        ordering["_random"] = np.random.default_rng(
            config.seed + annotator_index + 1
        ).random(len(ordering))
        shuffled = (
            base_form.assign(
                _within_block=ordering["_within_block"].to_numpy(),
                _random=ordering["_random"].to_numpy(),
            )
            .sort_values(["_within_block", "_random"])
            .drop(columns=["_within_block", "_random"])
        )
        shuffled.to_csv(path, index=False)
    tables = {
        "judge_validation_sample_key": key,
        "judge_validation_sampling_inventory": pd.DataFrame(inventory),
        "judge_validation_instructions": pd.DataFrame([
            {
                "instruction": (
                    "Two annotators independently fill human_score (integer 0-10), "
                    "evidence_span, confidence, and uncertainty_reason. "
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


def _merge_panel_scores(
    scored_rows: pd.DataFrame,
    config: ValidityControlsConfig,
) -> tuple[pd.DataFrame, list[str]]:
    if not config.published_caa_multi_judge_scores.exists():
        return scored_rows, []
    panel = _read_required(
        config.published_caa_multi_judge_scores,
        {"model_alias", "behavior", "item_id", "multiplier", "judge_alias", "judge_score"},
    )
    aliases = [
        alias
        for alias in config.panel_judge_aliases
        if alias in set(panel["judge_alias"].astype(str))
    ]
    if not aliases:
        return scored_rows, []
    panel = panel[panel["judge_alias"].astype(str).isin(aliases)].copy()
    panel["judge_score"] = pd.to_numeric(panel["judge_score"], errors="coerce")
    panel["multiplier"] = pd.to_numeric(panel["multiplier"], errors="coerce")
    key_columns = ["model_alias", "behavior", "item_id", "multiplier", "judge_alias"]
    duplicates = panel.duplicated(key_columns, keep=False)
    if duplicates.any():
        example = panel.loc[duplicates, key_columns].head(3).to_dict("records")
        raise ValueError(f"Multi-judge scores contain duplicate keys: {example}")
    wide = panel.pivot(
        index=["model_alias", "behavior", "item_id", "multiplier"],
        columns="judge_alias",
        values="judge_score",
    ).reset_index()
    for alias in aliases:
        if alias not in wide:
            wide[alias] = np.nan
    wide["three_judge_panel_score"] = wide[aliases].mean(axis=1)
    wide.loc[wide[aliases].isna().any(axis=1), "three_judge_panel_score"] = np.nan
    merged = scored_rows.merge(
        wide[
            ["model_alias", "behavior", "item_id", "multiplier"]
            + aliases
            + ["three_judge_panel_score"]
        ],
        on=["model_alias", "behavior", "item_id", "multiplier"],
        how="left",
        validate="one_to_one",
    )
    return merged, aliases


def _paired_effect_table(
    scored_rows: pd.DataFrame,
    config: ValidityControlsConfig,
    score_columns: dict[str, str],
    *,
    seed_offset: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + seed_offset)
    for (model_alias, model_name, behavior), group in scored_rows.groupby(
        ["model_alias", "model_name", "behavior"], sort=True
    ):
        pivots = {
            source: group.pivot_table(
                index="item_id",
                columns="multiplier",
                values=column,
                aggfunc="first",
            )
            for source, column in score_columns.items()
            if column in group
        }
        for multiplier in config.required_multipliers:
            if multiplier == 0.0:
                continue
            valid_pivots = {
                source: pivot
                for source, pivot in pivots.items()
                if multiplier in pivot and 0.0 in pivot
            }
            if not valid_pivots:
                continue
            item_ids: pd.Index | None = None
            for pivot in valid_pivots.values():
                available = pivot[[0.0, multiplier]].dropna().index
                item_ids = available if item_ids is None else item_ids.intersection(available)
            if item_ids is None:
                continue
            for source, pivot in valid_pivots.items():
                values = (
                    pivot.loc[item_ids, multiplier] - pivot.loc[item_ids, 0.0]
                ).to_numpy(dtype=float)
                estimate, low, high = _bootstrap_mean(
                    values,
                    n_boot=config.n_boot,
                    confidence=config.confidence,
                    rng=rng,
                )
                verdict = "positive" if low > 0 else "negative" if high < 0 else "inconclusive"
                rows.append({
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
    return pd.DataFrame(rows)


def _verdict_agreement_table(
    paired: pd.DataFrame,
    *,
    human_source: str,
    judge_source: str,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    verdict = paired.pivot_table(
        index=["model_alias", "model_name", "behavior", "multiplier"],
        columns="score_source",
        values="verdict",
        aggfunc="first",
    ).reset_index()
    if {human_source, judge_source}.issubset(verdict.columns):
        verdict["verdict_agrees"] = verdict[human_source].eq(verdict[judge_source])
    return verdict


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
        frame = _read_required(
            path,
            {
                "annotation_id",
                "human_score",
                "evidence_span",
                "confidence",
                "uncertainty_reason",
                "adjudication_flag",
            },
        )
        if frame["annotation_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate annotation_id values")
        unknown_ids = sorted(set(frame["annotation_id"]) - set(key["annotation_id"]))
        missing_ids = sorted(set(key["annotation_id"]) - set(frame["annotation_id"]))
        if unknown_ids or missing_ids:
            raise ValueError(
                f"{path} annotation IDs do not match the private key; "
                f"unknown={unknown_ids[:3]}, missing={missing_ids[:3]}"
            )
        frame["human_score"] = pd.to_numeric(frame["human_score"], errors="coerce")
        invalid = frame["human_score"].notna() & ~frame["human_score"].between(0.0, 10.0)
        if invalid.any():
            raise ValueError(f"{path} contains human_score values outside [0, 10]")
        noninteger = frame["human_score"].notna() & ~np.isclose(
            frame["human_score"], np.rint(frame["human_score"])
        )
        if noninteger.any():
            raise ValueError(f"{path} contains non-integer human_score values")
        evidence = frame["evidence_span"].fillna("").astype(str).str.strip()
        missing_evidence = frame["human_score"].notna() & evidence.eq("")
        if missing_evidence.any():
            raise ValueError(
                f"{path} has scored rows without evidence_span"
            )
        if "confidence" in frame:
            normalized_confidence = frame["confidence"].fillna("").astype(str).str.strip().str.lower()
            scored = frame["human_score"].notna()
            invalid_confidence = scored & ~normalized_confidence.isin(
                ["high", "medium", "low"]
            )
            if invalid_confidence.any():
                raise ValueError(
                    f"{path} must use high, medium, or low confidence for every scored row"
                )
            if "uncertainty_reason" in frame:
                reasons = frame["uncertainty_reason"].fillna("").astype(str).str.strip()
                missing_reason = scored & normalized_confidence.eq("low") & reasons.eq("")
                if missing_reason.any():
                    raise ValueError(
                        f"{path} has low-confidence rows without uncertainty_reason"
                    )
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
    complete, panel_aliases = _merge_panel_scores(complete, config)

    agreement_rows = [
        _agreement_record("human_1_vs_human_2", complete["human_score_1"], complete["human_score_2"]),
        _agreement_record("llm_judge_vs_human_mean", complete["judge_score"], complete["human_mean_score"]),
    ]
    for behavior, group in complete.groupby("behavior", sort=True):
        record = _agreement_record(
            f"llm_judge_vs_human_mean:{behavior}", group["judge_score"], group["human_mean_score"]
        )
        agreement_rows.append(record)
    if panel_aliases:
        for alias in panel_aliases:
            agreement_rows.append(
                _agreement_record(
                    f"{alias}_vs_human_mean",
                    complete[alias],
                    complete["human_mean_score"],
                )
            )
        agreement_rows.append(
            _agreement_record(
                "three_judge_panel_vs_human_mean",
                complete["three_judge_panel_score"],
                complete["human_mean_score"],
            )
        )
        for behavior, group in complete.groupby("behavior", sort=True):
            agreement_rows.append(
                _agreement_record(
                    f"three_judge_panel_vs_human_mean:{behavior}",
                    group["three_judge_panel_score"],
                    group["human_mean_score"],
                )
            )
    score_columns = {
        "human_mean": "human_mean_score",
        "llm_judge": "judge_score",
    }
    if "three_judge_panel_score" in complete:
        score_columns["three_judge_panel"] = "three_judge_panel_score"
    paired = _paired_effect_table(
        complete,
        config,
        score_columns,
        seed_offset=500,
    )
    judge_source = (
        "three_judge_panel"
        if "three_judge_panel" in set(paired.get("score_source", pd.Series(dtype=str)))
        else "llm_judge"
    )
    verdict = _verdict_agreement_table(
        paired,
        human_source="human_mean",
        judge_source=judge_source,
    )
    return {
        "judge_validation_status": status,
        "judge_validation_scored_rows": complete,
        "judge_validation_agreement": pd.DataFrame(agreement_rows),
        "judge_validation_paired_effects": paired,
        "judge_validation_verdict_agreement": verdict,
    }


def _adjudication_candidates(
    config: ValidityControlsConfig,
    summary: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    scored = summary.get("judge_validation_scored_rows", pd.DataFrame()).copy()
    if scored.empty:
        raise ValueError("Two complete annotation forms are required before adjudication")
    forms: list[pd.DataFrame] = []
    detail_columns = [
        "annotation_id",
        "rubric",
        "question",
        "response",
        "human_score",
        "evidence_span",
        "confidence",
        "uncertainty_reason",
        "adjudication_flag",
    ]
    for index, path in enumerate(config.annotation_files, start=1):
        frame = _read_required(path, detail_columns)
        renamed = {
            column: f"{column}_{index}"
            for column in detail_columns
            if column not in {
                "annotation_id",
                "rubric",
                "question",
                "response",
                "human_score",
            }
        }
        static = ["annotation_id"]
        if index == 1:
            static += ["rubric", "question", "response"]
        forms.append(frame[static + list(renamed)].rename(columns=renamed))
    candidates = scored.merge(forms[0], on="annotation_id", how="left", validate="one_to_one")
    candidates = candidates.merge(forms[1], on="annotation_id", how="left", validate="one_to_one")
    candidates["absolute_score_gap"] = (
        candidates["human_score_1"] - candidates["human_score_2"]
    ).abs()
    candidates["absence_substantial_conflict"] = (
        (candidates["human_score_1"].le(2) & candidates["human_score_2"].ge(6))
        | (candidates["human_score_2"].le(2) & candidates["human_score_1"].ge(6))
    )
    flag_1 = (
        candidates["adjudication_flag_1"].fillna("").astype(str).str.strip().str.lower().eq("yes")
    )
    flag_2 = (
        candidates["adjudication_flag_2"].fillna("").astype(str).str.strip().str.lower().eq("yes")
    )
    candidates["annotator_flag"] = flag_1 | flag_2
    selected = (
        candidates["absolute_score_gap"].ge(config.adjudication_score_gap)
        | candidates["absence_substantial_conflict"]
        | candidates["annotator_flag"]
    )
    candidates = candidates.loc[selected].copy()

    def reasons(row: pd.Series) -> str:
        values = []
        if row["absolute_score_gap"] >= config.adjudication_score_gap:
            values.append(f"score_gap>={config.adjudication_score_gap}")
        if bool(row["absence_substantial_conflict"]):
            values.append("absence_vs_substantial")
        if bool(row["annotator_flag"]):
            values.append("annotator_flag")
        return ";".join(values)

    candidates["adjudication_trigger"] = candidates.apply(reasons, axis=1)
    return candidates.sort_values(["behavior", "annotation_id"]).reset_index(drop=True)


def prepare_human_adjudication(
    config: ValidityControlsConfig,
) -> dict[str, pd.DataFrame]:
    summary = summarize_judge_validation(config)
    candidates = _adjudication_candidates(config, summary)
    inventory = (
        candidates.groupby("behavior", as_index=False)
        .agg(
            n_adjudication_rows=("annotation_id", "size"),
            mean_absolute_score_gap=("absolute_score_gap", "mean"),
            max_absolute_score_gap=("absolute_score_gap", "max"),
        )
        if not candidates.empty
        else pd.DataFrame(columns=[
            "behavior",
            "n_adjudication_rows",
            "mean_absolute_score_gap",
            "max_absolute_score_gap",
        ])
    )
    inventory = pd.concat(
        [
            inventory,
            pd.DataFrame([{
                "behavior": "__all__",
                "n_adjudication_rows": len(candidates),
                "mean_absolute_score_gap": candidates["absolute_score_gap"].mean(),
                "max_absolute_score_gap": candidates["absolute_score_gap"].max(),
            }]),
        ],
        ignore_index=True,
    )
    key_columns = [
        "annotation_id",
        "model_alias",
        "model_name",
        "behavior",
        "item_id",
        "multiplier",
        "human_score_1",
        "human_score_2",
        "evidence_span_1",
        "evidence_span_2",
        "confidence_1",
        "confidence_2",
        "absolute_score_gap",
        "absence_substantial_conflict",
        "annotator_flag",
        "adjudication_trigger",
    ]
    private_key = candidates[key_columns].copy()
    rng = np.random.default_rng(config.seed + 701)
    swap = rng.random(len(candidates)) < 0.5
    blind = candidates[
        ["annotation_id", "behavior", "rubric", "question", "response", "adjudication_trigger"]
    ].copy()
    blind["candidate_score_a"] = np.where(
        swap, candidates["human_score_2"], candidates["human_score_1"]
    )
    blind["candidate_evidence_a"] = np.where(
        swap, candidates["evidence_span_2"], candidates["evidence_span_1"]
    )
    blind["candidate_score_b"] = np.where(
        swap, candidates["human_score_1"], candidates["human_score_2"]
    )
    blind["candidate_evidence_b"] = np.where(
        swap, candidates["evidence_span_1"], candidates["evidence_span_2"]
    )
    blind["adjudicated_score"] = np.nan
    blind["adjudication_reason"] = ""
    blind["confidence"] = ""
    blind = blind.sample(frac=1.0, random_state=config.seed + 702).reset_index(drop=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.adjudication_file.exists():
        existing = pd.read_csv(config.adjudication_file)
        if (
            "adjudicated_score" in existing
            and pd.to_numeric(existing["adjudicated_score"], errors="coerce").notna().any()
        ):
            blind = existing
        else:
            blind.to_csv(config.adjudication_file, index=False)
    else:
        config.adjudication_file.parent.mkdir(parents=True, exist_ok=True)
        blind.to_csv(config.adjudication_file, index=False)
    instructions = pd.DataFrame([{
        "instruction": (
            "The adjudicator independently assigns adjudicated_score (integer 0-10) "
            "after reviewing the response, rubric, and both anonymized evidence spans. "
            "The adjudicator remains blind to model, multiplier, item_id, and automatic judges. "
            "Do not change annotation_id or source text; provide adjudication_reason and confidence."
        ),
        "adjudication_file": str(config.adjudication_file),
        "n_rows": len(candidates),
    }])
    tables = {
        "judge_validation_adjudication_private_key": private_key,
        "judge_validation_adjudication_inventory": inventory,
        "judge_validation_adjudication_instructions": instructions,
    }
    write_tables(tables, config.output_dir)
    return tables


def summarize_human_adjudication(
    config: ValidityControlsConfig,
) -> dict[str, pd.DataFrame]:
    base = summarize_judge_validation(config)
    candidates = _adjudication_candidates(config, base)
    if not config.adjudication_file.exists():
        raise FileNotFoundError(
            f"Missing adjudication form: {config.adjudication_file}. "
            "Run prepare-caa-human-adjudication first."
        )
    adjudication = _read_required(
        config.adjudication_file,
        {
            "annotation_id",
            "adjudicated_score",
            "adjudication_reason",
            "confidence",
        },
    )
    if adjudication["annotation_id"].duplicated().any():
        raise ValueError(f"{config.adjudication_file} contains duplicate annotation_id values")
    expected = set(candidates["annotation_id"])
    observed = set(adjudication["annotation_id"])
    if expected != observed:
        raise ValueError(
            "Adjudication IDs do not match current trigger rows; "
            f"unknown={sorted(observed - expected)[:3]}, missing={sorted(expected - observed)[:3]}"
        )
    adjudication["adjudicated_score"] = pd.to_numeric(
        adjudication["adjudicated_score"], errors="coerce"
    )
    scored = adjudication["adjudicated_score"].notna()
    invalid = scored & ~adjudication["adjudicated_score"].between(0.0, 10.0)
    noninteger = scored & ~np.isclose(
        adjudication["adjudicated_score"],
        np.rint(adjudication["adjudicated_score"]),
    )
    if invalid.any() or noninteger.any():
        raise ValueError("adjudicated_score must be an integer in [0, 10]")
    reasons = adjudication["adjudication_reason"].fillna("").astype(str).str.strip()
    confidence = adjudication["confidence"].fillna("").astype(str).str.strip().str.lower()
    if (scored & reasons.eq("")).any():
        raise ValueError("Every adjudicated row must include adjudication_reason")
    if (scored & ~confidence.isin(["high", "medium", "low"])).any():
        raise ValueError("Every adjudicated row must use high, medium, or low confidence")
    status = pd.DataFrame([{
        "file": str(config.adjudication_file),
        "status": "complete" if scored.all() else "partial",
        "n_scored": int(scored.sum()),
        "n_expected": len(candidates),
    }])
    tables = dict(base)
    tables["judge_validation_adjudication_status"] = status
    if not scored.all():
        return tables
    final = base["judge_validation_scored_rows"].merge(
        adjudication[
            ["annotation_id", "adjudicated_score", "adjudication_reason", "confidence"]
        ].rename(columns={"confidence": "adjudication_confidence"}),
        on="annotation_id",
        how="left",
        validate="one_to_one",
    )
    final["human_final_score"] = final["adjudicated_score"].fillna(
        final["human_mean_score"]
    )
    final["human_score_source"] = np.where(
        final["adjudicated_score"].notna(),
        "adjudicated",
        "two_annotator_mean",
    )
    agreement_rows = [
        _agreement_record(
            "human_final_vs_human_mean",
            final["human_final_score"],
            final["human_mean_score"],
        )
    ]
    if "three_judge_panel_score" in final:
        agreement_rows.append(
            _agreement_record(
                "three_judge_panel_vs_human_final",
                final["three_judge_panel_score"],
                final["human_final_score"],
            )
        )
        for behavior, group in final.groupby("behavior", sort=True):
            agreement_rows.append(
                _agreement_record(
                    f"three_judge_panel_vs_human_final:{behavior}",
                    group["three_judge_panel_score"],
                    group["human_final_score"],
                )
            )
    score_columns = {"human_final": "human_final_score"}
    if "three_judge_panel_score" in final:
        score_columns["three_judge_panel"] = "three_judge_panel_score"
    else:
        score_columns["llm_judge"] = "judge_score"
    paired = _paired_effect_table(
        final,
        config,
        score_columns,
        seed_offset=800,
    )
    judge_source = (
        "three_judge_panel"
        if "three_judge_panel" in set(paired.get("score_source", pd.Series(dtype=str)))
        else "llm_judge"
    )
    tables.update({
        "judge_validation_final_scored_rows": final,
        "judge_validation_final_agreement": pd.DataFrame(agreement_rows),
        "judge_validation_final_paired_effects": paired,
        "judge_validation_final_verdict_agreement": _verdict_agreement_table(
            paired,
            human_source="human_final",
            judge_source=judge_source,
        ),
    })
    return tables


def prepare_caa_human_annotation_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = ValidityControlsConfig.from_json(config_path, project_root=project_root)
    return prepare_judge_validation(config)


def run_evaluation_validity_controls_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    from .cross_interface_audit import (
        CrossInterfaceConfig,
        aggregate_cross_interface_outputs,
        run_cross_interface_audit_from_json,
    )

    allowed = {"random", "prepare-judge", "summarize", "all"}
    if phase not in allowed:
        raise ValueError(f"Unsupported phase {phase!r}; expected {sorted(allowed)}")
    tables: dict[str, pd.DataFrame] = {}
    if phase in {"random", "all"}:
        result = run_cross_interface_audit_from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
            model_aliases=model_aliases,
        )
        tables.update({f"random_audit__{key}": value for key, value in result.items()})
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


def summarize_caa_human_annotation_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = ValidityControlsConfig.from_json(config_path, project_root=project_root)
    tables = summarize_judge_validation(config)
    write_tables(tables, config.output_dir)
    return tables


def prepare_caa_human_adjudication_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = ValidityControlsConfig.from_json(config_path, project_root=project_root)
    return prepare_human_adjudication(config)


def summarize_caa_human_adjudication_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = ValidityControlsConfig.from_json(config_path, project_root=project_root)
    tables = summarize_human_adjudication(config)
    write_tables(tables, config.output_dir)
    return tables


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
