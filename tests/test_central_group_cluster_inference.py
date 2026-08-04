from __future__ import annotations

import numpy as np
import pandas as pd

from cross_interface_steering.central_group_cluster_inference import (
    _cluster_bootstrap_balanced_means,
    _pair_group_map,
)


def test_pair_group_map_rejects_conflicting_group_assignments() -> None:
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "pair_type": "a_vs_b",
                "group_key": "g1",
                "split": "test",
            },
            {
                "pair_id": "p1",
                "pair_type": "a_vs_b",
                "group_key": "g2",
                "split": "test",
            },
        ]
    )
    try:
        _pair_group_map(pairs)
    except ValueError as error:
        assert "one context group" in str(error)
    else:
        raise AssertionError("Conflicting pair-to-group assignments must fail")


def test_cluster_bootstrap_resamples_complete_groups_across_strata() -> None:
    rows = []
    for group_key, values in {
        "g1": {"c1": [1.0, 1.0], "c2": [2.0]},
        "g2": {"c1": [3.0], "c2": [4.0, 4.0]},
        "g3": {"c1": [5.0], "c2": [6.0]},
    }.items():
        for pair_type, group_values in values.items():
            for index, value in enumerate(group_values):
                rows.append(
                    {
                        "group_key": group_key,
                        "pair_type": pair_type,
                        "pair_id": f"{group_key}-{pair_type}-{index}",
                        "effect": value,
                    }
                )
    frame = pd.DataFrame(rows)
    result = _cluster_bootstrap_balanced_means(
        frame,
        strata=("pair_type",),
        values=("effect",),
        n_boot=250,
        confidence=0.95,
        rng=np.random.default_rng(7),
        standardized_value="effect",
        positive_value="effect",
    )
    expected = np.mean(
        [
            frame.loc[frame["pair_type"].eq("c1"), "effect"].mean(),
            frame.loc[frame["pair_type"].eq("c2"), "effect"].mean(),
        ]
    )
    assert np.isclose(result["point"]["effect"], expected)
    assert result["n_groups"] == 3
    assert result["n_strata"] == 2
    assert result["positive_point"] == 1.0
    assert result["interval"]["effect"][0] < result["point"]["effect"]
    assert result["interval"]["effect"][1] > result["point"]["effect"]
