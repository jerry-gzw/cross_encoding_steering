from __future__ import annotations

import numpy as np
import pandas as pd

from cross_encoding_steering.readout_baseline_audit import (
    BASELINE_MODES,
    build_simple_readout_baseline_bank,
)


def test_simple_readout_baselines_follow_target_minus_source_orientation() -> None:
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
            [0.0, 2.0, 0.0],
            [0.0, 1.8, 0.1],
            [0.0, 0.0, 3.0],
            [0.1, 0.0, 2.9],
        ],
        dtype=np.float32,
    )
    raw = np.asarray([0.0, -2.0, 0.0], dtype=np.float32)
    cached = {
        ("taboo_vs_expected", "raw_canonical_direction"): raw,
        (
            "taboo_vs_expected",
            "readout_projection_norm_matched",
        ): raw,
    }
    directions, direction_inventory, cosines = (
        build_simple_readout_baseline_bank(
            cached_directions=cached,
            train_gradients=gradients,
            train_gradient_inventory=inventory,
            canonical_labels=("taboo", "normal", "expected"),
            identifier_token_vectors=np.eye(3, dtype=np.float32),
            normalize_gradient_rows=True,
            enabled_modes=BASELINE_MODES,
            extraction_mapping="canonical",
        )
    )
    mean_direction = directions[
        ("taboo_vs_expected", "mean_local_gradient_norm_matched")
    ]
    assert mean_direction[1] < 0
    assert np.isclose(np.linalg.norm(mean_direction), np.linalg.norm(raw))
    assert set(direction_inventory["mode"]) == set(BASELINE_MODES)
    assert set(cosines["reference_mode"]) == {
        "raw_canonical_direction",
        "readout_projection_norm_matched",
    }
