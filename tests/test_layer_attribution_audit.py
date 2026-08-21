from __future__ import annotations

import numpy as np
import pandas as pd

from cross_encoding_steering.layer_attribution_audit import (
    LAYER_ATTRIBUTION_MODES,
    build_layer_direction_bank,
)
from cross_encoding_steering.layer_attribution_statistics import (
    build_layer_group_cluster_statistics,
)


def test_layer_direction_bank_builds_norm_matched_controls() -> None:
    inventory = pd.DataFrame(
        [
            {"left_identifier": "A", "right_identifier": "B"},
            {"left_identifier": "A", "right_identifier": "B"},
            {"left_identifier": "A", "right_identifier": "C"},
            {"left_identifier": "A", "right_identifier": "C"},
            {"left_identifier": "B", "right_identifier": "C"},
            {"left_identifier": "B", "right_identifier": "C"},
        ]
    )
    gradients = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    raw = np.asarray([1.0, 2.0, 1.0], dtype=np.float32)
    basis = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    directions, direction_inventory, arrays = build_layer_direction_bank(
        raw_directions={"taboo_vs_expected": raw},
        readout_basis=basis,
        train_gradients=gradients,
        train_gradient_inventory=inventory,
        canonical_labels=("taboo", "normal", "expected"),
        normalize_gradient_rows=True,
        enabled_modes=LAYER_ATTRIBUTION_MODES,
        extraction_mapping="canonical",
        layer_index=7,
        requested_layer_fraction=0.75,
        resolved_layer_fraction=0.7,
        depth_norm_reference_norms={
            "taboo_vs_expected": float(np.linalg.norm(raw))
        },
        depth_norm_reference_layer_index=7,
        depth_norm_reference_fraction=0.75,
    )
    assert len(directions) == 5
    raw_norm = np.linalg.norm(raw)
    for vector in directions.values():
        assert np.isclose(np.linalg.norm(vector), raw_norm)
    assert set(direction_inventory["mode"]) == set(
        LAYER_ATTRIBUTION_MODES
    )
    assert "readout_basis" in arrays


def test_depth_norm_matched_direction_uses_reference_layer_norm() -> None:
    raw = np.asarray([3.0, 4.0, 0.0], dtype=np.float32)
    target_norm = 2.5
    directions, inventory, arrays = build_layer_direction_bank(
        raw_directions={"taboo_vs_expected": raw},
        readout_basis=None,
        train_gradients=None,
        train_gradient_inventory=None,
        canonical_labels=("taboo", "normal", "expected"),
        normalize_gradient_rows=True,
        enabled_modes=(
            "raw_canonical_direction",
            "raw_canonical_direction_depth_norm_matched",
        ),
        extraction_mapping="canonical",
        layer_index=5,
        requested_layer_fraction=0.5,
        resolved_layer_fraction=0.5,
        depth_norm_reference_norms={"taboo_vs_expected": target_norm},
        depth_norm_reference_layer_index=7,
        depth_norm_reference_fraction=0.75,
    )
    matched = directions[
        (
            "taboo_vs_expected",
            "raw_canonical_direction_depth_norm_matched",
        )
    ]
    assert np.isclose(np.linalg.norm(matched), target_norm)
    matched_row = inventory.loc[inventory["depth_norm_matched"]].iloc[0]
    assert np.isclose(matched_row["depth_norm_scale"], 0.5)
    assert matched_row["depth_norm_reference_layer_index"] == 7
    assert "readout_basis" not in arrays


def test_layer_group_cluster_statistics_preserve_paired_depths() -> None:
    pairs = pd.DataFrame(
        [
            {
                "pair_id": f"p{group}_{contrast}",
                "pair_type": contrast,
                "group_key": f"g{group}",
                "split": "test",
            }
            for group in range(4)
            for contrast in ("low_vs_mid", "mid_vs_high")
        ]
    )
    rows = []
    for model_alias in ("model_a", "model_b"):
        for pair in pairs.itertuples(index=False):
            for depth, identifier_gain, current_gain in (
                (0.5, 0.1, 0.08),
                (0.75, 0.5, 0.1),
                (0.875, 0.7, 0.1),
            ):
                for mapping in ("canonical", "remap_1", "remap_2"):
                    canonical = mapping == "canonical"
                    rows.append(
                        {
                            "model_alias": model_alias,
                            "pair_id": pair.pair_id,
                            "pair_type": pair.pair_type,
                            "mapping_name": mapping,
                            "mode": "raw_canonical_direction",
                            "requested_layer_fraction": depth,
                            "semantic_prob_gain": (
                                identifier_gain if canonical else current_gain
                            ),
                            "original_letter_prob_gain": identifier_gain,
                            "semantic_minus_original_prob_gain_moved": (
                                np.nan
                                if canonical
                                else current_gain - identifier_gain
                            ),
                        }
                    )
    effects = pd.DataFrame(rows)
    matched = effects.copy()
    matched["mode"] = "raw_canonical_direction_depth_norm_matched"
    profile, contrasts = build_layer_group_cluster_statistics(
        pd.concat([effects, matched], ignore_index=True),
        pairs,
        canonical_mapping="canonical",
        n_boot=100,
        confidence=0.95,
        seed=13,
    )

    late_advantage = profile.loc[
        profile["mode"].eq("raw_canonical_direction")
        & profile["requested_layer_fraction"].eq(0.875)
        & profile["metric"].eq("extraction_identifier_advantage")
    ].iloc[0]
    assert np.isclose(late_advantage["mean"], 0.6)
    assert late_advantage["n_groups"] == 4
    late_vs_early = contrasts.loc[
        contrasts["mode"].eq("raw_canonical_direction")
        & contrasts["metric"].eq("extraction_identifier_effect")
        & contrasts["late_depth"].eq(0.875)
        & contrasts["early_depth"].eq(0.5)
    ].iloc[0]
    assert np.isclose(late_vs_early["late_minus_early"], 0.6)
    assert set(contrasts["mode"]) == {
        "raw_canonical_direction",
        "raw_canonical_direction_depth_norm_matched",
    }
