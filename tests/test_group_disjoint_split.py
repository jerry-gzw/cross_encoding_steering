from __future__ import annotations

import pandas as pd

from cross_encoding_steering.pairs import apply_generated_split, audit_split_isolation


def _synthetic_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    items = []
    for group_index in range(30):
        group_key = f"setting-{group_index} | behavior-{group_index}"
        endpoint_ids = [f"item-{group_index}-{label}" for label in ("low", "mid", "high")]
        for item_id in endpoint_ids:
            items.append({"item_id": item_id, "split": "source"})
        for pair_type, left, right in [
            ("low_vs_mid", 0, 1),
            ("low_vs_high", 0, 2),
            ("mid_vs_high", 1, 2),
        ]:
            rows.append(
                {
                    "pair_id": f"pair-{group_index}-{pair_type}",
                    "dataset": "normbank",
                    "pair_type": pair_type,
                    "split": "source",
                    "group_key": group_key,
                    "negative_item_id": endpoint_ids[left],
                    "positive_item_id": endpoint_ids[right],
                }
            )
    return pd.DataFrame(items), pd.DataFrame(rows)


def test_group_key_split_is_disjoint_for_groups_and_endpoints() -> None:
    items, pairs = _synthetic_pairs()
    split_spec = {
        "enabled": True,
        "train": 0.8,
        "validation": 0.1,
        "test": 0.1,
        "key": "group_key",
        "include_pair_type": False,
        "enforce_item_disjoint": True,
    }

    split_items, split_pairs, inventory = apply_generated_split(
        items,
        pairs,
        split_spec,
        seed=13,
    )
    audit = audit_split_isolation(split_pairs)
    summary = audit["split_isolation_summary"].iloc[0]

    assert split_pairs.groupby("group_key")["split"].nunique().max() == 1
    assert split_items.groupby("item_id")["split"].nunique().max() == 1
    assert bool(summary["strict_split_isolation_pass"])
    assert inventory["split_key"].eq("group_key").all()
    assert not inventory["include_pair_type_in_split_key"].any()


def test_pair_id_split_can_be_rejected_when_endpoints_cross_splits() -> None:
    items, pairs = _synthetic_pairs()
    split_spec = {
        "enabled": True,
        "train": 0.5,
        "validation": 0.25,
        "test": 0.25,
        "key": "pair_id",
        "include_pair_type": True,
        "enforce_item_disjoint": True,
    }

    try:
        apply_generated_split(items, pairs, split_spec, seed=13)
    except ValueError as error:
        assert "endpoint leakage" in str(error)
    else:
        raise AssertionError("Expected endpoint leakage to be rejected")
