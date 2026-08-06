"""Group-cluster inference for the layer-wise attribution audit."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .central_group_cluster_inference import (
    _attach_groups,
    _cluster_bootstrap_balanced_means,
    _pair_group_map,
)


def build_layer_group_cluster_statistics(
    effects: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    canonical_mapping: str,
    n_boot: int,
    confidence: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate layer profiles by resampling complete context groups."""
    pair_map = _pair_group_map(pairs)
    grouped = _attach_groups(
        effects,
        pair_map,
        name="Layer-attribution pair effects",
    )
    grouped["extraction_identifier_advantage"] = (
        -grouped["semantic_minus_original_prob_gain_moved"]
    )
    profile_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 1701)
    for (layer_fraction, mode), frame in grouped.groupby(
        ["requested_layer_fraction", "mode"],
        sort=True,
    ):
        scopes = {
            "source_interface_effect": (
                frame.loc[frame["mapping_name"].eq(canonical_mapping)],
                "semantic_prob_gain",
                ("model_alias", "pair_type"),
            ),
            "current_label_effect": (
                frame.loc[frame["mapping_name"].ne(canonical_mapping)],
                "semantic_prob_gain",
                ("model_alias", "pair_type", "mapping_name"),
            ),
            "extraction_identifier_effect": (
                frame.loc[frame["mapping_name"].ne(canonical_mapping)],
                "original_letter_prob_gain",
                ("model_alias", "pair_type", "mapping_name"),
            ),
            "extraction_identifier_advantage": (
                frame.loc[
                    frame["mapping_name"].ne(canonical_mapping)
                    & frame["extraction_identifier_advantage"].notna()
                ],
                "extraction_identifier_advantage",
                ("model_alias", "pair_type", "mapping_name"),
            ),
        }
        for metric, (metric_frame, value, strata) in scopes.items():
            result = _cluster_bootstrap_balanced_means(
                metric_frame,
                strata=strata,
                values=(value,),
                n_boot=n_boot,
                confidence=confidence,
                rng=rng,
            )
            low, high = result["interval"][value]
            profile_rows.append(
                {
                    "requested_layer_fraction": float(layer_fraction),
                    "mode": mode,
                    "metric": metric,
                    "mean": result["point"][value],
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "n_pairs": result["n_pairs"],
                    "n_groups": result["n_groups"],
                    "n_strata": result["n_strata"],
                    "n_boot": n_boot,
                    "confidence": confidence,
                    "inference_unit": (
                        "setting_behavior_group; resampled jointly across "
                        "models, contrasts, and mappings within depth"
                    ),
                }
            )

    contrast_rows: list[dict[str, Any]] = []
    raw = grouped.loc[
        grouped["mode"].eq("raw_canonical_direction")
        & grouped["mapping_name"].ne(canonical_mapping)
    ].copy()
    depth_pairs = ((0.75, 0.5), (0.875, 0.5), (0.875, 0.625))
    for metric, value in (
        ("extraction_identifier_effect", "original_letter_prob_gain"),
        (
            "extraction_identifier_advantage",
            "extraction_identifier_advantage",
        ),
    ):
        metric_frame = raw.loc[raw[value].notna()].copy()
        index = [
            "model_alias",
            "pair_type",
            "mapping_name",
            "pair_id",
            "group_key",
        ]
        pivot = metric_frame.pivot_table(
            index=index,
            columns="requested_layer_fraction",
            values=value,
            aggfunc="mean",
        ).reset_index()
        for late_depth, early_depth in depth_pairs:
            if late_depth not in pivot or early_depth not in pivot:
                continue
            difference = pivot[index].copy()
            difference["depth_difference"] = (
                pivot[late_depth] - pivot[early_depth]
            )
            difference = difference.dropna(subset=["depth_difference"])
            result = _cluster_bootstrap_balanced_means(
                difference,
                strata=("model_alias", "pair_type", "mapping_name"),
                values=("depth_difference",),
                n_boot=n_boot,
                confidence=confidence,
                rng=rng,
            )
            low, high = result["interval"]["depth_difference"]
            contrast_rows.append(
                {
                    "metric": metric,
                    "late_depth": late_depth,
                    "early_depth": early_depth,
                    "late_minus_early": result["point"][
                        "depth_difference"
                    ],
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "n_pairs": result["n_pairs"],
                    "n_groups": result["n_groups"],
                    "n_strata": result["n_strata"],
                    "n_boot": n_boot,
                    "confidence": confidence,
                    "inference_unit": (
                        "paired depth difference with setting_behavior_group "
                        "cluster resampling"
                    ),
                }
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(contrast_rows)
