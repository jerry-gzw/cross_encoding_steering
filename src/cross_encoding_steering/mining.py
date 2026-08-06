from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

from .pairs import pair_quality, render_prompt, stable_id
from .text import clean_text, length_ratio, normalize_text, token_jaccard, tokens


def sequence_ratio(left: str, right: str) -> float:
    return float(SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio())


def _candidate_indices(
    query_tokens: set[str],
    inverted_index: dict[str, set[int]],
    *,
    max_candidates: int,
) -> list[int]:
    counts: Counter[int] = Counter()
    for token in query_tokens:
        for idx in inverted_index.get(token, set()):
            counts[int(idx)] += 1
    return [idx for idx, _ in counts.most_common(max_candidates)]


def _build_inverted_index(frame: pd.DataFrame, text_column: str) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for idx, text in frame[text_column].items():
        for token in tokens(text):
            index[token].add(int(idx))
    return index


def mine_label_flip_pairs(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    text_column: str,
    label_column: str,
    negative_label: str,
    positive_label: str,
    split_column: str = "split",
    id_column: str | None = None,
    prompt_template: str | None = None,
    max_pairs_per_split: int = 1000,
    max_candidates_per_item: int = 200,
    max_source_items_per_split: int | None = 1000,
    max_sequence_candidates_per_item: int = 20,
    min_token_jaccard: float = 0.20,
    min_sequence_ratio: float = 0.42,
    min_length_ratio: float = 0.55,
    top_k_per_item: int = 1,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = frame.copy()
    if split_column not in rows.columns:
        rows[split_column] = "all"
    if id_column is None or id_column not in rows.columns:
        rows["_row_id"] = rows.index.astype(str)
        id_column = "_row_id"
    rows[text_column] = rows[text_column].map(clean_text)
    rows[label_column] = rows[label_column].astype(str)
    rows = rows[rows[text_column].str.len().gt(0)].copy()
    rows = rows[rows[label_column].isin([str(negative_label), str(positive_label)])].copy()

    rng = np.random.default_rng(seed)
    pair_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for split, split_rows in rows.groupby(split_column, dropna=False, sort=True):
        split = str(split)
        negative_rows = split_rows[split_rows[label_column].eq(str(negative_label))].copy()
        positive_rows = split_rows[split_rows[label_column].eq(str(positive_label))].copy()
        if negative_rows.empty or positive_rows.empty:
            continue
        if max_source_items_per_split and len(negative_rows) > max_source_items_per_split:
            negative_rows = negative_rows.sample(
                n=int(max_source_items_per_split),
                random_state=(int(stable_id(dataset_name, split), 16) + int(seed)) % (2**32 - 1),
            )
        positive_index = _build_inverted_index(positive_rows, text_column)
        scored_pairs = []
        for neg_idx, neg in negative_rows.iterrows():
            candidate_ids = _candidate_indices(
                tokens(neg[text_column]),
                positive_index,
                max_candidates=max_candidates_per_item,
            )
            preliminary_scores = []
            for pos_idx in candidate_ids:
                pos = positive_rows.loc[pos_idx]
                tj = token_jaccard(neg[text_column], pos[text_column])
                if tj < min_token_jaccard:
                    continue
                lr = length_ratio(neg[text_column], pos[text_column])
                if lr < min_length_ratio:
                    continue
                cheap_score = 0.65 * tj + 0.35 * lr
                preliminary_scores.append((float(cheap_score), float(tj), float(lr), int(pos_idx)))
            preliminary_scores.sort(key=lambda item: item[0], reverse=True)
            local_scores = []
            for _, tj, lr, pos_idx in preliminary_scores[: max(1, int(max_sequence_candidates_per_item))]:
                pos = positive_rows.loc[pos_idx]
                sr = sequence_ratio(neg[text_column], pos[text_column])
                if sr < min_sequence_ratio:
                    continue
                quality = pair_quality(neg[text_column], pos[text_column])
                mining_score = 0.45 * tj + 0.25 * lr + 0.30 * sr
                local_scores.append((float(mining_score), float(sr), int(neg_idx), int(pos_idx), quality))
            local_scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
            scored_pairs.extend(local_scores[: max(1, int(top_k_per_item))])
        scored_pairs.sort(key=lambda item: (item[0], item[1], rng.random()), reverse=True)
        used_neg: set[int] = set()
        used_pos: set[int] = set()
        kept = 0
        for mining_score, seq_ratio, neg_idx, pos_idx, quality in scored_pairs:
            if kept >= max_pairs_per_split:
                break
            if neg_idx in used_neg or pos_idx in used_pos:
                continue
            neg = rows.loc[neg_idx]
            pos = rows.loc[pos_idx]
            pair_type = f"{negative_label}_vs_{positive_label}"
            pair_id = stable_id(dataset_name, pair_type, split, neg[id_column], pos[id_column])
            negative_item_id = f"{neg[id_column]}:{negative_label}"
            positive_item_id = f"{pos[id_column]}:{positive_label}"
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "dataset": dataset_name,
                    "pair_type": pair_type,
                    "split": split,
                    "group_key": "",
                    "negative_item_id": negative_item_id,
                    "positive_item_id": positive_item_id,
                    "negative_source_id": str(neg[id_column]),
                    "positive_source_id": str(pos[id_column]),
                    "negative_label": str(negative_label),
                    "positive_label": str(positive_label),
                    "negative_text": neg[text_column],
                    "positive_text": pos[text_column],
                    "negative_prompt": render_prompt(prompt_template, neg, neg[text_column]),
                    "positive_prompt": render_prompt(prompt_template, pos, pos[text_column]),
                    "sequence_ratio": float(seq_ratio),
                    "mining_score": float(mining_score),
                    **quality,
                }
            )
            used_neg.add(neg_idx)
            used_pos.add(pos_idx)
            kept += 1

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pd.DataFrame(), pairs, pd.DataFrame()

    for side in ["negative", "positive"]:
        id_col = f"{side}_item_id"
        label_col = f"{side}_label"
        text_col = f"{side}_text"
        prompt_col = f"{side}_prompt"
        for _, pair in pairs.iterrows():
            item_rows.append(
                {
                    "item_id": pair[id_col],
                    "dataset": dataset_name,
                    "split": pair["split"],
                    "label": pair[label_col],
                    "text": pair[text_col],
                    "prompt": pair[prompt_col],
                }
            )
    items = pd.DataFrame(item_rows).drop_duplicates("item_id").reset_index(drop=True)
    inventory = (
        pairs.groupby(["dataset", "pair_type", "split", "quality_bin"], as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_quality_score=("quality_score", "mean"),
            mean_token_jaccard=("token_jaccard", "mean"),
            mean_sequence_ratio=("sequence_ratio", "mean"),
            mean_length_ratio=("length_ratio", "mean"),
        )
        .sort_values(["dataset", "pair_type", "split", "quality_bin"])
    )
    return items, pairs, inventory
