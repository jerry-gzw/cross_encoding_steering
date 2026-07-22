"""Pair-level context-conditioned discrimination analysis.

This module consumes the endpoint-level ``context_plus`` observations already
written by the cross-interface evaluator.  It does not rerun a language model:
the required baseline and steered high-vs-low margins are already present for
both endpoints of every strict counterfactual pair.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import write_tables


@dataclass(frozen=True)
class ContextDiscriminationConfig:
    project_root: Path
    output_dir: Path
    input_dirs: tuple[Path, ...]
    optional_input_dirs: tuple[Path, ...]
    modes: tuple[str, ...]
    interfaces: tuple[str, ...]
    templates: tuple[str, ...]
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
    ) -> "ContextDiscriminationConfig":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        statistics = dict(data.get("statistics", {}))
        confidence = float(statistics.get("confidence", 0.95))
        n_boot = int(statistics.get("n_boot", 2_000))
        if not 0.0 < confidence < 1.0:
            raise ValueError("statistics.confidence must be between zero and one")
        if n_boot < 100:
            raise ValueError("statistics.n_boot must be at least 100")
        return cls(
            project_root=root,
            output_dir=resolve(str(data.get("output_dir", "outputs/normbank/context_selectivity"))),
            input_dirs=tuple(resolve(str(value)) for value in data.get("input_dirs", [])),
            optional_input_dirs=tuple(resolve(str(value)) for value in data.get("optional_input_dirs", [])),
            modes=tuple(str(value) for value in data.get("modes", [])),
            interfaces=tuple(str(value) for value in data.get("interfaces", [])),
            templates=tuple(str(value) for value in data.get("templates", [])),
            minimum_base_gap=float(data.get("minimum_base_gap", 0.05)),
            n_boot=n_boot,
            confidence=confidence,
            seed=int(statistics.get("seed", 13)),
        )


def _discover_eval_files(directory: Path, *, required: bool) -> list[Path]:
    if directory.is_file():
        return [directory]
    if not directory.exists():
        if required:
            raise FileNotFoundError(f"Context-discrimination input does not exist: {directory}")
        return []
    aggregate = directory / "cross_interface_eval_rows.csv"
    if aggregate.exists():
        return [aggregate]
    files = sorted(directory.glob("*/cross_interface_eval_rows.csv"))
    if required and not files:
        raise FileNotFoundError(f"No cross_interface_eval_rows.csv found under {directory}")
    return files


def _load_context_rows(config: ContextDiscriminationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    files: list[tuple[Path, bool]] = []
    for directory in config.input_dirs:
        files.extend((path, True) for path in _discover_eval_files(directory, required=True))
    for directory in config.optional_input_dirs:
        files.extend((path, False) for path in _discover_eval_files(directory, required=False))
    if not files:
        raise FileNotFoundError("No context-discrimination input files were discovered")

    frames, inventory = [], []
    for source_index, (path, required) in enumerate(files):
        frame = pd.read_csv(path)
        required_columns = {
            "model_alias", "model_name", "pair_id", "pair_type", "endpoint",
            "evaluation_policy", "mode", "interface", "template",
            "base_high_low_margin", "patched_high_low_margin",
        }
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            if required:
                raise ValueError(f"{path} is missing required columns: {missing}")
            continue
        selected = frame[frame["evaluation_policy"].astype(str).eq("context_plus")].copy()
        if config.modes:
            selected = selected[selected["mode"].astype(str).isin(config.modes)]
        if config.interfaces:
            selected = selected[selected["interface"].astype(str).isin(config.interfaces)]
        if config.templates:
            selected = selected[selected["template"].astype(str).isin(config.templates)]
        selected["source_index"] = source_index
        selected["source_file"] = str(path)
        frames.append(selected)
        inventory.append(
            {
                "source_file": str(path),
                "required": required,
                "n_input_rows": len(frame),
                "n_context_rows_selected": len(selected),
                "n_models": selected["model_alias"].nunique(),
                "modes": ",".join(sorted(selected["mode"].astype(str).unique())),
            }
        )
    rows = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if rows.empty:
        raise ValueError("No context_plus rows remain after filtering")

    keys = ["model_alias", "pair_id", "pair_type", "endpoint", "mode", "interface", "template"]
    duplicates = rows.duplicated(keys, keep=False)
    if duplicates.any():
        spread = (
            rows.loc[duplicates]
            .groupby(keys, as_index=False)
            .agg(
                base_spread=("base_high_low_margin", lambda values: float(values.max() - values.min())),
                patched_spread=("patched_high_low_margin", lambda values: float(values.max() - values.min())),
            )
        )
        if (spread[["base_spread", "patched_spread"]].to_numpy(dtype=float) > 1e-5).any():
            raise ValueError("Repeated context rows disagree across input sources; do not mix incompatible runs")
        rows = rows.sort_values("source_index").drop_duplicates(keys, keep="last")
    return rows.reset_index(drop=True), pd.DataFrame(inventory)


def build_context_pair_rows(rows: pd.DataFrame, *, minimum_base_gap: float) -> pd.DataFrame:
    keys = ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface", "template"]
    values = ["base_high_low_margin", "patched_high_low_margin"]
    endpoint_counts = rows.groupby(keys)["endpoint"].nunique()
    invalid = endpoint_counts[endpoint_counts.ne(2)]
    if not invalid.empty:
        raise ValueError(f"Expected low/high endpoints for every context pair; invalid groups: {len(invalid)}")
    wide = rows.pivot_table(index=keys, columns="endpoint", values=values, aggfunc="first")
    required = {(metric, endpoint) for metric in values for endpoint in ("low", "high")}
    if not required.issubset(set(wide.columns)):
        raise ValueError("Context rows must include low and high endpoint margins")
    wide.columns = [f"{metric}_{endpoint}" for metric, endpoint in wide.columns]
    pair = wide.reset_index()
    pair["low_margin_shift"] = pair["patched_high_low_margin_low"] - pair["base_high_low_margin_low"]
    pair["high_margin_shift"] = pair["patched_high_low_margin_high"] - pair["base_high_low_margin_high"]
    pair["context_interaction"] = pair["low_margin_shift"] - pair["high_margin_shift"]
    pair["base_discrimination_gap"] = pair["base_high_low_margin_high"] - pair["base_high_low_margin_low"]
    pair["patched_discrimination_gap"] = pair["patched_high_low_margin_high"] - pair["patched_high_low_margin_low"]
    pair["discrimination_gap_change"] = pair["patched_discrimination_gap"] - pair["base_discrimination_gap"]
    pair["base_rank_correct"] = pair["base_discrimination_gap"].gt(0)
    pair["patched_rank_correct"] = pair["patched_discrimination_gap"].gt(0)
    pair["rank_sign_preserved"] = np.sign(pair["patched_discrimination_gap"]) == np.sign(pair["base_discrimination_gap"])
    eligible = pair["base_discrimination_gap"].abs().ge(float(minimum_base_gap))
    pair["discrimination_retention_ratio"] = np.where(
        eligible,
        pair["patched_discrimination_gap"] / pair["base_discrimination_gap"],
        np.nan,
    )
    pair["retention_eligible"] = eligible
    return pair


PAIR_METRICS = (
    "context_interaction",
    "low_margin_shift",
    "high_margin_shift",
    "base_discrimination_gap",
    "patched_discrimination_gap",
    "discrimination_gap_change",
    "base_rank_correct",
    "patched_rank_correct",
    "rank_sign_preserved",
)


def _bootstrap_interval(values: np.ndarray, *, n_boot: int, confidence: float, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    estimates = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))


def summarize_context_pairs(pair_rows: pd.DataFrame, config: ContextDiscriminationConfig) -> dict[str, pd.DataFrame]:
    group_columns = ["model_alias", "model_name", "mode", "interface", "template", "pair_type"]
    summary = (
        pair_rows.groupby(group_columns, as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            context_interaction=("context_interaction", "mean"),
            low_margin_shift=("low_margin_shift", "mean"),
            high_margin_shift=("high_margin_shift", "mean"),
            base_discrimination_gap=("base_discrimination_gap", "mean"),
            patched_discrimination_gap=("patched_discrimination_gap", "mean"),
            discrimination_gap_change=("discrimination_gap_change", "mean"),
            base_rank_accuracy=("base_rank_correct", "mean"),
            patched_rank_accuracy=("patched_rank_correct", "mean"),
            rank_sign_preservation=("rank_sign_preserved", "mean"),
            median_discrimination_retention=("discrimination_retention_ratio", "median"),
            n_retention_eligible=("retention_eligible", "sum"),
        )
    )
    rng = np.random.default_rng(config.seed)
    ci_rows: list[dict[str, Any]] = []
    for keys, group in pair_rows.groupby(group_columns, sort=True):
        record = dict(zip(group_columns, keys))
        record["n_pairs"] = int(group["pair_id"].nunique())
        for metric in PAIR_METRICS:
            values = group[metric].astype(float).to_numpy()
            low, high = _bootstrap_interval(
                values,
                n_boot=config.n_boot,
                confidence=config.confidence,
                rng=rng,
            )
            record[f"{metric}_mean"] = float(np.nanmean(values))
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
            record[f"{metric}_ci_excludes_zero"] = bool(low > 0 or high < 0)
        ci_rows.append(record)
    return {
        "context_discrimination_summary": summary,
        "context_discrimination_bootstrap_ci": pd.DataFrame(ci_rows),
    }


def run_context_discrimination_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = ContextDiscriminationConfig.from_json(config_path, project_root=project_root)
    rows, inventory = _load_context_rows(config)
    pair_rows = build_context_pair_rows(rows, minimum_base_gap=config.minimum_base_gap)
    tables = summarize_context_pairs(pair_rows, config)
    tables.update(
        {
            "context_discrimination_pair_rows": pair_rows,
            "context_discrimination_input_inventory": inventory,
            "context_discrimination_config": pd.DataFrame(
                [
                    {
                        "minimum_base_gap": config.minimum_base_gap,
                        "n_boot": config.n_boot,
                        "confidence": config.confidence,
                        "seed": config.seed,
                    }
                ]
            ),
        }
    )
    write_tables(tables, config.output_dir)
    return tables
