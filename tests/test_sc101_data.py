from __future__ import annotations

from pathlib import Path

import pandas as pd

from cross_interface_steering.config import DatasetConfig
from cross_interface_steering.data import build_sc101_pairs


def test_sc101_action_only_adapter_collapses_labels(tmp_path: Path) -> None:
    rows = []
    judgments = [-2, 0, 2]
    for group_index in range(12):
        for label_index, judgment in enumerate(judgments):
            rows.append(
                {
                    "split": "train",
                    "rot-bad": 0,
                    "rot-categorization": "social-norms",
                    "action-moral-judgment": judgment,
                    "action": f"action {group_index} {label_index}",
                    "situation": f"situation {group_index}",
                    "area": "general",
                    "rot": "rule",
                    "rot-id": f"{group_index}-{label_index}",
                }
            )
    source = tmp_path / "social-chem-101.v1.0.tsv"
    pd.DataFrame(rows).to_csv(source, sep="\t", index=False)
    config = DatasetConfig(
        name="social_chemistry_101",
        adapter="social_chemistry_101",
        path=str(source),
        format="tsv",
        include_splits=["train"],
        max_pairs_per_type_split=100,
        generated_split={
            "enabled": True,
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
            "key": "pair_id",
        },
        metadata={"prompt_variant": "action_only", "social_norms_only": True},
        prompt_template="Action: {sc101_text}",
    )

    items, pairs, metadata = build_sc101_pairs(config, tmp_path, seed=13)

    assert set(items["label"]) == {"bad", "ok", "good"}
    assert set(pairs["pair_type"]) == {"bad_vs_ok", "bad_vs_good", "ok_vs_good"}
    assert set(pairs["split"]).issubset({"train", "validation", "test"})
    assert metadata.loc[0, "prompt_variant"] == "action_only"
