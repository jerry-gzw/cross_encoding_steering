from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .text import clean_text, length_ratio, quality_bin, token_jaccard


def stable_id(*parts: Any) -> str:
    text = "||".join(clean_text(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def stable_float(*parts: Any, seed: int = 13) -> float:
    text = "||".join([str(seed), *[clean_text(part) for part in parts]])
    value = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)
    return value / float(16**12 - 1)


def render_prompt(template: str | None, row: pd.Series, text: str) -> str:
    if not template:
        return text
    values = {key: clean_text(value) for key, value in row.items()}
    values["text"] = clean_text(text)
    try:
        return template.format(**values)
    except KeyError:
        return text


def pair_quality(negative_text: str, positive_text: str, *, shared_context: str = "") -> dict[str, float | str]:
    jaccard = token_jaccard(negative_text, positive_text)
    ratio = length_ratio(negative_text, positive_text)
    context_bonus = token_jaccard(shared_context, negative_text + " " + positive_text) if shared_context else 0.0
    score = 0.55 * jaccard + 0.35 * ratio + 0.10 * context_bonus
    return {
        "token_jaccard": float(jaccard),
        "length_ratio": float(ratio),
        "shared_context_overlap": float(context_bonus),
        "quality_score": float(score),
        "quality_bin": quality_bin(float(score)),
    }


def summarize_pairs(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if pairs.empty:
        return {
            "pair_inventory": pd.DataFrame(),
            "quality_summary": pd.DataFrame(),
            "examples_good": pd.DataFrame(),
            "examples_bad": pd.DataFrame(),
        }
    group_cols = [col for col in ["dataset", "pair_type", "split", "quality_bin"] if col in pairs.columns]
    inventory = (
        pairs.groupby(group_cols, as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_quality_score=("quality_score", "mean"),
            mean_token_jaccard=("token_jaccard", "mean"),
            mean_length_ratio=("length_ratio", "mean"),
        )
        .sort_values(group_cols)
    )
    quality = (
        pairs.groupby(["dataset", "pair_type"], as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            high_quality_rate=("quality_bin", lambda s: float((s == "high").mean())),
            medium_or_high_rate=("quality_bin", lambda s: float(s.isin(["medium", "high"]).mean())),
            mean_quality_score=("quality_score", "mean"),
            mean_token_jaccard=("token_jaccard", "mean"),
            mean_length_ratio=("length_ratio", "mean"),
        )
        .sort_values(["dataset", "pair_type"])
    )
    examples_good = pairs.sort_values("quality_score", ascending=False).head(30).copy()
    examples_bad = pairs.sort_values("quality_score", ascending=True).head(30).copy()
    return {
        "pair_inventory": inventory,
        "quality_summary": quality,
        "examples_good": examples_good,
        "examples_bad": examples_bad,
    }


def apply_generated_split(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    split_spec: dict[str, Any] | None,
    *,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not split_spec or not split_spec.get("enabled", False) or pairs.empty:
        return items, pairs, pd.DataFrame()

    train_ratio = float(split_spec.get("train", 0.8))
    validation_ratio = float(split_spec.get("validation", 0.1))
    test_ratio = float(split_spec.get("test", 0.1))
    total = train_ratio + validation_ratio + test_ratio
    if total <= 0:
        raise ValueError("generated_split train/validation/test ratios must sum to a positive value")
    train_cut = train_ratio / total
    validation_cut = (train_ratio + validation_ratio) / total
    key_column = str(split_spec.get("key", "pair_id"))
    include_pair_type = bool(split_spec.get("include_pair_type", True))
    if key_column not in pairs.columns:
        raise KeyError(f"generated_split key column {key_column!r} not found in pairs")

    out_pairs = pairs.copy()
    old_split = out_pairs["split"].copy() if "split" in out_pairs.columns else pd.Series([""] * len(out_pairs))
    new_splits = []
    for _, row in out_pairs.iterrows():
        split_parts = [row[key_column], row.get("dataset", "")]
        if include_pair_type:
            split_parts.insert(1, row.get("pair_type", ""))
        value = stable_float(*split_parts, seed=seed)
        if value < train_cut:
            new_splits.append("train")
        elif value < validation_cut:
            new_splits.append("validation")
        else:
            new_splits.append("test")
    out_pairs["source_split"] = old_split.astype(str).to_numpy()
    out_pairs["split"] = new_splits

    out_items = items.copy()
    item_split: dict[str, str] = {}
    for _, row in out_pairs.iterrows():
        for column in ["negative_item_id", "positive_item_id"]:
            if column in row:
                item_split[str(row[column])] = str(row["split"])
    if "item_id" in out_items.columns and item_split:
        out_items["source_split"] = out_items["split"].astype(str) if "split" in out_items.columns else ""
        out_items["split"] = out_items["item_id"].astype(str).map(item_split).fillna(out_items.get("split", ""))

    inventory = (
        out_pairs.groupby(["dataset", "pair_type", "split"], as_index=False)
        .agg(n_pairs=("pair_id", "nunique"))
        .sort_values(["dataset", "pair_type", "split"])
    )
    inventory["split_source"] = "generated"
    inventory["split_key"] = key_column
    inventory["train_ratio"] = train_ratio
    inventory["validation_ratio"] = validation_ratio
    inventory["test_ratio"] = test_ratio
    return out_items, out_pairs, inventory


def build_grouped_label_pairs(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    text_fields: list[str],
    label_field: str,
    label_order: list[str],
    group_fields: list[str],
    split_field: str | None,
    id_field: str | None,
    prompt_template: str | None,
    max_pairs_per_type_split: int,
    include_splits: list[str] | None = None,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = frame.copy()
    if include_splits and split_field and split_field in rows.columns:
        rows = rows[rows[split_field].astype(str).isin(include_splits)].copy()
    for field in text_fields + [label_field] + group_fields:
        if field and field not in rows.columns:
            raise KeyError(f"Missing required column {field!r} for dataset {dataset_name}")

    rows["_item_id"] = rows[id_field].astype(str) if id_field and id_field in rows.columns else rows.index.astype(str)
    rows["_text"] = rows[text_fields].fillna("").astype(str).agg(" ".join, axis=1).map(clean_text)
    rows["_label"] = rows[label_field].astype(str)
    rows["_split"] = rows[split_field].astype(str) if split_field and split_field in rows.columns else "all"
    rows = rows[rows["_label"].isin(label_order) & rows["_text"].str.len().gt(0)].copy()

    label_pairs = list(combinations(label_order, 2))
    pair_rows: list[dict[str, Any]] = []
    groupby_cols = list(group_fields) + ["_split"] if group_fields else ["_split"]
    for group_key, group in rows.groupby(groupby_cols, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        split = str(group_key[-1])
        context_values = group_key[:-1]
        group_key_text = " | ".join(clean_text(value) for value in context_values)
        for negative_label, positive_label in label_pairs:
            left = group[group["_label"].eq(negative_label)]
            right = group[group["_label"].eq(positive_label)]
            if left.empty or right.empty:
                continue
            n = min(len(left), len(right), max_pairs_per_type_split)
            left_sample = left.sample(n=n, random_state=seed + len(pair_rows)).reset_index(drop=True)
            right_sample = right.sample(n=n, random_state=seed + len(pair_rows) + 97).reset_index(drop=True)
            if n > max_pairs_per_type_split:
                keep = rng.choice(n, size=max_pairs_per_type_split, replace=False)
            else:
                keep = range(n)
            pair_type = f"{negative_label}_vs_{positive_label}"
            for offset in keep:
                neg = left_sample.iloc[int(offset)]
                pos = right_sample.iloc[int(offset)]
                quality = pair_quality(neg["_text"], pos["_text"], shared_context=group_key_text)
                pair_id = stable_id(dataset_name, pair_type, split, neg["_item_id"], pos["_item_id"])
                pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "dataset": dataset_name,
                        "pair_type": pair_type,
                        "split": split,
                        "group_key": group_key_text,
                        "negative_item_id": neg["_item_id"],
                        "positive_item_id": pos["_item_id"],
                        "negative_label": negative_label,
                        "positive_label": positive_label,
                        "negative_text": neg["_text"],
                        "positive_text": pos["_text"],
                        "negative_prompt": render_prompt(prompt_template, neg, neg["_text"]),
                        "positive_prompt": render_prompt(prompt_template, pos, pos["_text"]),
                        **quality,
                    }
                )
    pair_frame = pd.DataFrame(pair_rows)
    item_rows = []
    for _, row in rows.iterrows():
        item_rows.append(
            {
                "item_id": row["_item_id"],
                "dataset": dataset_name,
                "split": row["_split"],
                "label": row["_label"],
                "text": row["_text"],
                "prompt": render_prompt(prompt_template, row, row["_text"]),
            }
        )
    return pd.DataFrame(item_rows).drop_duplicates("item_id"), pair_frame


def build_paired_field_pairs(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    negative_field: str,
    positive_field: str,
    context_fields: list[str],
    split_field: str | None,
    id_field: str | None,
    negative_label: str,
    positive_label: str,
    prompt_template: str | None,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del seed
    rows = frame.copy()
    for field in [negative_field, positive_field] + context_fields:
        if field and field not in rows.columns:
            raise KeyError(f"Missing required column {field!r} for dataset {dataset_name}")
    rows["_item_id"] = rows[id_field].astype(str) if id_field and id_field in rows.columns else rows.index.astype(str)
    rows["_split"] = rows[split_field].astype(str) if split_field and split_field in rows.columns else "all"
    pair_rows = []
    item_rows = []
    for _, row in rows.iterrows():
        context = " ".join(clean_text(row.get(field, "")) for field in context_fields)
        negative_text = clean_text(row.get(negative_field, ""))
        positive_text = clean_text(row.get(positive_field, ""))
        if not negative_text or not positive_text:
            continue
        base_id = row["_item_id"]
        neg_id = f"{base_id}:negative"
        pos_id = f"{base_id}:positive"
        quality = pair_quality(negative_text, positive_text, shared_context=context)
        pair_rows.append(
            {
                "pair_id": stable_id(dataset_name, base_id, negative_field, positive_field),
                "dataset": dataset_name,
                "pair_type": f"{negative_label}_vs_{positive_label}",
                "split": row["_split"],
                "group_key": context,
                "negative_item_id": neg_id,
                "positive_item_id": pos_id,
                "negative_label": negative_label,
                "positive_label": positive_label,
                "negative_text": negative_text,
                "positive_text": positive_text,
                "negative_prompt": render_prompt(prompt_template, row, negative_text),
                "positive_prompt": render_prompt(prompt_template, row, positive_text),
                **quality,
            }
        )
        item_rows.extend(
            [
                {
                    "item_id": neg_id,
                    "dataset": dataset_name,
                    "split": row["_split"],
                    "label": negative_label,
                    "text": negative_text,
                    "prompt": render_prompt(prompt_template, row, negative_text),
                },
                {
                    "item_id": pos_id,
                    "dataset": dataset_name,
                    "split": row["_split"],
                    "label": positive_label,
                    "text": positive_text,
                    "prompt": render_prompt(prompt_template, row, positive_text),
                },
            ]
        )
    return pd.DataFrame(item_rows), pd.DataFrame(pair_rows)
