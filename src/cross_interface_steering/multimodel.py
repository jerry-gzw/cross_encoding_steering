from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .io import write_tables
from .evaluation import EvaluationSpec, run_decomposition_evaluation
from .steering import decoder_layers
from .statistics import run_paired_statistics


def resolve_fractional_layers(model: Any, fractions: Sequence[float]) -> tuple[list[int], pd.DataFrame]:
    layers = decoder_layers(model)
    n_layers = len(layers)
    if n_layers < 2:
        raise ValueError(f"Expected at least two decoder layers, found {n_layers}")
    rows = []
    for fraction in fractions:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Layer fraction must be in [0, 1], found {value}")
        index = int(round(value * (n_layers - 1)))
        rows.append(
            {
                "requested_layer_fraction": value,
                "resolved_layer_index": index,
                "resolved_layer_fraction": index / (n_layers - 1),
                "n_decoder_layers": n_layers,
            }
        )
    mapping = pd.DataFrame(rows).drop_duplicates("resolved_layer_index").sort_values("resolved_layer_index")
    return mapping["resolved_layer_index"].astype(int).tolist(), mapping.reset_index(drop=True)


def _annotate_tables(
    tables: dict[str, pd.DataFrame],
    *,
    model_name: str,
    model_alias: str,
) -> dict[str, pd.DataFrame]:
    annotated = {}
    for name, frame in tables.items():
        out = frame.copy()
        if "model_alias" not in out.columns:
            out.insert(0, "model_alias", model_alias)
        if "model_name" not in out.columns:
            out.insert(1, "model_name", model_name)
        annotated[name] = out
    return annotated


def model_run_is_complete(model_dir: str | Path) -> bool:
    root = Path(model_dir).expanduser().resolve()
    fixed_required = [
        root / "model_run_complete.csv",
        root / "locked_test_pair_rows.csv",
        root / "specificity_by_mode.csv",
    ]
    statistics_candidates = [
        [
            root / "paired_statistics" / "mode_equal_group_bootstrap_ci.csv",
            root / "paired_statistics" / "mic_mode_equal_axis_bootstrap_ci.csv",
        ],
        [
            root / "paired_statistics" / "equal_group_paired_comparisons.csv",
            root / "paired_statistics" / "mic_equal_axis_paired_comparisons.csv",
        ],
    ]
    return all(path.exists() for path in fixed_required) and all(
        any(path.exists() for path in candidates) for candidates in statistics_candidates
    )


def run_single_model_experiment(
    model: Any,
    tokenizer: Any,
    pairs: pd.DataFrame,
    *,
    evaluation_spec: EvaluationSpec,
    output_root: str | Path,
    model_name: str,
    model_alias: str,
    layer_fractions: Sequence[float],
    alpha_grid: list[float],
    subspace_dims: list[int],
    mixing_grid: list[float],
    prompt_variants: list[str],
    eval_directions: list[str],
    max_train_pairs_per_group: int,
    max_validation_pairs_per_group: int,
    max_test_pairs_per_group: int,
    batch_size: int,
    max_length: int,
    n_boot: int,
    n_permutations: int,
    confidence: float,
    seed: int,
    include_loco: bool = False,
    loco_subspace_dim: int = 1,
    extraction_position: str = "pre_answer",
) -> dict[str, dict[str, pd.DataFrame]]:
    model_dir = Path(output_root).expanduser().resolve() / model_alias
    model_dir.mkdir(parents=True, exist_ok=True)
    resolved_layers, layer_map = resolve_fractional_layers(model, layer_fractions)
    layer_map.insert(0, "model_alias", model_alias)
    layer_map.insert(1, "model_name", model_name)
    layer_map.to_csv(model_dir / "layer_map.csv", index=False)

    validation_tables = run_decomposition_evaluation(
        model,
        tokenizer,
        pairs,
        spec=evaluation_spec,
        output_dir=model_dir,
        layer_indices=resolved_layers,
        alpha_grid=alpha_grid,
        subspace_dims=subspace_dims,
        mixing_grid=mixing_grid,
        prompt_variants=prompt_variants,
        eval_directions=eval_directions,
        max_train_pairs_per_group=max_train_pairs_per_group,
        max_validation_pairs_per_group=max_validation_pairs_per_group,
        max_test_pairs_per_group=max_test_pairs_per_group,
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
        include_loco=include_loco,
        loco_subspace_dim=loco_subspace_dim,
        extraction_position=extraction_position,
    )
    validation_tables = _annotate_tables(
        validation_tables,
        model_name=model_name,
        model_alias=model_alias,
    )
    write_tables(validation_tables, model_dir)

    statistics_tables = run_paired_statistics(
        model_dir,
        output_dir=model_dir / "paired_statistics",
        group_column="pair_type",
        n_boot=n_boot,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=seed,
    )
    statistics_tables = _annotate_tables(
        statistics_tables,
        model_name=model_name,
        model_alias=model_alias,
    )
    write_tables(statistics_tables, model_dir / "paired_statistics")

    selected = validation_tables["selected_settings"]
    completion = pd.DataFrame(
        [
            {
                "model_alias": model_alias,
                "model_name": model_name,
                "status": "complete",
                "n_decoder_layers": int(layer_map["n_decoder_layers"].iloc[0]),
                "candidate_layers": ",".join(map(str, resolved_layers)),
                "selected_layer": int(selected["layer_index"].iloc[0]),
                "selected_layer_fraction": float(
                    int(selected["layer_index"].iloc[0]) / (int(layer_map["n_decoder_layers"].iloc[0]) - 1)
                ),
                "n_selected_pairs": int(validation_tables["selected_pairs"]["pair_id"].nunique()),
                "n_boot": int(n_boot),
                "n_permutations": int(n_permutations),
                "seed": int(seed),
                "include_loco": bool(include_loco),
                "loco_subspace_dim": int(loco_subspace_dim),
                "extraction_position": extraction_position,
            }
        ]
    )
    completion.to_csv(model_dir / "model_run_complete.csv", index=False)
    return {"validation": validation_tables, "statistics": statistics_tables}


def _read_completed_table(model_dirs: list[Path], relative_paths: str | Sequence[str]) -> pd.DataFrame:
    candidates = [relative_paths] if isinstance(relative_paths, str) else list(relative_paths)
    parts = []
    for model_dir in model_dirs:
        for relative_path in candidates:
            path = model_dir / relative_path
            if path.exists():
                parts.append(pd.read_csv(path))
                break
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _normalize_legacy_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.rename(
        columns={
            "n_axes": "n_groups",
            "min_pairs_per_axis": "min_pairs_per_group",
            "max_pairs_per_axis": "max_pairs_per_group",
        }
    ).copy()
    if "estimand" in out.columns:
        out["estimand"] = out["estimand"].replace("equal_axis_mean", "equal_group_mean")
    return out


def aggregate_multimodel_outputs(
    output_root: str | Path,
    *,
    expected_models: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_dirs = sorted(path for path in root.iterdir() if path.is_dir() and model_run_is_complete(path))
    completion = _read_completed_table(model_dirs, "model_run_complete.csv")
    selected = _read_completed_table(model_dirs, "selected_settings.csv")
    pair_weighted_ci = _read_completed_table(
        model_dirs,
        ["paired_statistics/mode_bootstrap_ci.csv", "paired_statistics/mic_mode_bootstrap_ci.csv"],
    )
    equal_group_ci = _read_completed_table(
        model_dirs,
        "paired_statistics/mode_equal_group_bootstrap_ci.csv",
    )
    pair_weighted_comparisons = _read_completed_table(
        model_dirs, "paired_statistics/paired_comparisons.csv"
    )
    equal_group_comparisons = _read_completed_table(
        model_dirs, "paired_statistics/equal_group_paired_comparisons.csv"
    )
    equal_group_ci = _normalize_legacy_group_columns(equal_group_ci)
    equal_group_comparisons = _normalize_legacy_group_columns(equal_group_comparisons)
    decisions = _read_completed_table(model_dirs, "paired_statistics/statistics_decision.csv")
    specificity = _read_completed_table(model_dirs, "specificity_by_mode.csv")
    direction_inventory = _read_completed_table(model_dirs, "selected_direction_inventory.csv")

    if expected_models is None:
        inventory = completion[["model_alias", "model_name", "status"]].copy() if not completion.empty else pd.DataFrame()
    else:
        inventory = expected_models[["model_alias", "model_name"]].copy()
        complete_aliases = set(completion.get("model_alias", pd.Series(dtype=str)).astype(str))
        inventory["status"] = inventory["model_alias"].astype(str).map(
            lambda alias: "complete" if alias in complete_aliases else "missing"
        )

    raw_label = "raw_caa_direction_minus_random_direction_control"
    shared_label = "shared_only_direction_minus_residual_only_direction"
    loco_label = "loco_shared_direction_minus_loco_residual_direction"
    loco_random_label = "loco_shared_direction_minus_random_direction_control"
    raw_rows = equal_group_comparisons[
        equal_group_comparisons.get("comparison", pd.Series(dtype=str)).eq(raw_label)
    ]
    shared_rows = equal_group_comparisons[
        equal_group_comparisons.get("comparison", pd.Series(dtype=str)).eq(shared_label)
    ]
    loco_rows = equal_group_comparisons[
        equal_group_comparisons.get("comparison", pd.Series(dtype=str)).eq(loco_label)
    ]
    loco_random_rows = equal_group_comparisons[
        equal_group_comparisons.get("comparison", pd.Series(dtype=str)).eq(loco_random_label)
    ]
    n_models = int(completion["model_alias"].nunique()) if not completion.empty else 0
    n_raw_significant = int(raw_rows.get("significant_holm_0_05", pd.Series(dtype=bool)).fillna(False).sum())
    n_shared_significant = int(
        (
            shared_rows.get("significant_holm_0_05", pd.Series(dtype=bool)).fillna(False)
            & shared_rows.get("mean_delta_intended_acc_difference", pd.Series(dtype=float)).gt(0)
        ).sum()
    )
    n_loco_significant = int(
        (
            loco_rows.get("significant_holm_0_05", pd.Series(dtype=bool)).fillna(False)
            & loco_rows.get("mean_delta_intended_acc_difference", pd.Series(dtype=float)).gt(0)
        ).sum()
    )
    n_loco_random_significant = int(
        (
            loco_random_rows.get("significant_holm_0_05", pd.Series(dtype=bool)).fillna(False)
            & loco_random_rows.get("mean_delta_intended_acc_difference", pd.Series(dtype=float)).gt(0)
        ).sum()
    )
    global_decision = pd.DataFrame(
        [
            {
                "n_models_complete": n_models,
                "n_models_expected": int(len(expected_models)) if expected_models is not None else n_models,
                "n_models_raw_minus_random_significant": n_raw_significant,
                "n_models_shared_minus_residual_significant": n_shared_significant,
                "all_models_raw_signal_supported": bool(n_models > 0 and n_raw_significant == n_models),
                "majority_models_shared_dominance_supported": bool(
                    n_models > 0 and n_shared_significant >= math.ceil(n_models / 2)
                ),
                "n_models_loco_shared_minus_residual_significant": n_loco_significant,
                "n_models_loco_shared_minus_random_significant": n_loco_random_significant,
                "majority_models_loco_shared_dominance_supported": bool(
                    n_models > 0 and n_loco_significant >= math.ceil(n_models / 2)
                ),
                "majority_models_loco_signal_supported": bool(
                    n_models > 0 and n_loco_random_significant >= math.ceil(n_models / 2)
                ),
                "mean_equal_group_raw_minus_random": float(
                    raw_rows["mean_delta_intended_acc_difference"].mean()
                )
                if not raw_rows.empty
                else np.nan,
                "min_equal_group_raw_minus_random": float(
                    raw_rows["mean_delta_intended_acc_difference"].min()
                )
                if not raw_rows.empty
                else np.nan,
                "mean_equal_group_shared_minus_residual": float(
                    shared_rows["mean_delta_intended_acc_difference"].mean()
                )
                if not shared_rows.empty
                else np.nan,
                "aggregation_note": "Models are summarized as independent replications; pair rows are not pooled across models.",
            }
        ]
    )
    outputs = {
        "multimodel_inventory": inventory,
        "multimodel_completion": completion,
        "multimodel_selected_settings": selected,
        "multimodel_pair_weighted_ci": pair_weighted_ci,
        "multimodel_equal_group_ci": equal_group_ci,
        "multimodel_pair_weighted_comparisons": pair_weighted_comparisons,
        "multimodel_equal_group_comparisons": equal_group_comparisons,
        "multimodel_statistics_decisions": decisions,
        "multimodel_specificity": specificity,
        "multimodel_direction_inventory": direction_inventory,
        "multimodel_global_decision": global_decision,
    }
    write_tables(outputs, root)
    return outputs
