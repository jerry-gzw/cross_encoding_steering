from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import write_tables


def build_ranked_endpoints_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Convert pair-audit rows into the endpoint table used by mapping audits.

    Pair builders produce one row per contrastive pair. The mapping-balanced
    steering code expects two rows per pair: one lower-ranked endpoint and one
    higher-ranked endpoint.
    """
    required = {
        "pair_id",
        "dataset",
        "pair_type",
        "split",
        "negative_item_id",
        "positive_item_id",
        "negative_label",
        "positive_label",
        "negative_text",
        "positive_text",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing required columns: {missing}")

    rows = []
    optional_columns = [
        "group_key",
        "moral_axis",
        "quality_score",
        "quality_bin",
        "token_jaccard",
        "length_ratio",
        "shared_context_overlap",
        "source_split",
    ]
    for _, pair in pairs.iterrows():
        common = {
            "pair_id": str(pair["pair_id"]),
            "dataset": str(pair["dataset"]),
            "pair_type": str(pair["pair_type"]),
            "direction_split": str(pair["split"]),
        }
        for column in optional_columns:
            if column in pairs.columns:
                common[column] = pair.get(column)
        endpoints = [
            ("negative", "negative_item_id", "negative_label", "negative_text"),
            ("positive", "positive_item_id", "positive_label", "positive_text"),
        ]
        for role, item_column, label_column, text_column in endpoints:
            text = "" if pd.isna(pair[text_column]) else str(pair[text_column]).strip()
            if not text:
                continue
            rows.append(
                {
                    **common,
                    "endpoint_role": role,
                    "item_id": str(pair[item_column]),
                    "label_name": str(pair[label_column]),
                    "text": text,
                }
            )
    endpoints = pd.DataFrame(rows)
    if endpoints.empty:
        raise ValueError("No ranked endpoints could be built from the pair table")
    endpoints = endpoints.drop_duplicates(["pair_id", "label_name", "direction_split"])
    invalid = []
    for pair_id, group in endpoints.groupby("pair_id", sort=True):
        if len(group) != 2 or group["label_name"].nunique() != 2:
            invalid.append(str(pair_id))
    if invalid:
        raise ValueError(f"Expected exactly two labels per pair; invalid examples: {invalid[:5]}")
    return endpoints.reset_index(drop=True)


def build_ranked_endpoints_from_csv(
    pairs_path: str | Path,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    pairs = pd.read_csv(Path(pairs_path).expanduser())
    endpoints = build_ranked_endpoints_from_pairs(pairs)
    summary = (
        endpoints.groupby(["dataset", "pair_type", "direction_split", "label_name"], as_index=False)
        .agg(n_items=("item_id", "nunique"), n_pairs=("pair_id", "nunique"))
        .sort_values(["dataset", "pair_type", "direction_split", "label_name"])
    )
    tables = {
        "ranked_endpoints": endpoints,
        "ranked_endpoint_inventory": summary,
    }
    write_tables(tables, output_dir)
    return tables
