from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .io import read_table, write_tables
from .mining import sequence_ratio
from .pairs import pair_quality, stable_id
from .text import clean_text, normalize_text, token_jaccard


def endpoint_overlap_table(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["left_split", "right_split", "n_shared_source_endpoints"])
    endpoint_sets: dict[str, set[str]] = {}
    for split, group in pairs.groupby("split", sort=True):
        endpoints = set(group.get("negative_source_id", pd.Series(dtype=str)).astype(str))
        endpoints.update(group.get("positive_source_id", pd.Series(dtype=str)).astype(str))
        endpoint_sets[str(split)] = endpoints
    rows = []
    names = sorted(endpoint_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "n_shared_source_endpoints": len(endpoint_sets[left] & endpoint_sets[right]),
                }
            )
    return pd.DataFrame(rows)


def _moral_slug(value: Any) -> str:
    text = normalize_text(value) or "unknown"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "unknown"


def prepare_mic_rows(
    path: str | Path,
    *,
    min_rot_agreement: float = 3.0,
    exclude_cross_split_dialogues: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = read_table(path).copy()
    rows["source_row_id"] = rows.index.astype(str)
    rows["split"] = rows["split"].replace({"dev": "validation"})
    rows["A_agrees_num"] = pd.to_numeric(rows["A_agrees"], errors="coerce")
    rows["rot_agree_num"] = pd.to_numeric(rows["rot-agree"], errors="coerce")
    rows["violation_severity_num"] = pd.to_numeric(rows["violation-severity"], errors="coerce")
    for column in ["Q", "A", "QA", "rot", "moral", "chatbot"]:
        rows[column] = rows.get(column, "").map(clean_text)
    rows["QA"] = rows["QA"].where(rows["QA"].str.len().gt(0), "Q: " + rows["Q"] + " A: " + rows["A"])
    rows["qa_norm"] = rows["QA"].map(normalize_text)
    rows["rot_norm"] = rows["rot"].map(normalize_text)
    rows["moral_slug"] = rows["moral"].map(_moral_slug)
    valid = rows[
        rows["split"].isin(["train", "validation", "test"])
        & rows["A_agrees_num"].isin([0.0, 2.0])
        & rows["rot_agree_num"].ge(float(min_rot_agreement))
        & rows["qa_norm"].str.len().gt(0)
        & rows["rot_norm"].str.len().gt(0)
    ].copy()
    cross_split_dialogues = valid.groupby("qa_norm")["split"].nunique()
    cross_split_dialogues = set(cross_split_dialogues[cross_split_dialogues.gt(1)].index)
    valid["cross_split_dialogue"] = valid["qa_norm"].isin(cross_split_dialogues)
    if exclude_cross_split_dialogues:
        valid = valid[~valid["cross_split_dialogue"]].copy()
    inventory = (
        rows.assign(valid_for_pairing=rows.index.isin(valid.index))
        .groupby(["split", "A_agrees_num"], dropna=False, as_index=False)
        .agg(n_rows=("source_row_id", "size"), n_valid=("valid_for_pairing", "sum"))
        .sort_values(["split", "A_agrees_num"])
    )
    inventory["n_cross_split_dialogues_excluded"] = int(len(cross_split_dialogues)) if exclude_cross_split_dialogues else 0
    return valid, inventory


def _format_prompt(qa: str, rot: str) -> str:
    return f"Dialogue: {qa}\nRule of thumb: {rot}\nQuestion: Does the answer agree with the rule of thumb?"


def build_mic_rule_conditioned_pairs(
    rows: pd.DataFrame,
    *,
    max_pairs_per_dialogue_axis: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    conflict_rows = []
    for (split, qa_norm, moral_slug), group in rows.groupby(["split", "qa_norm", "moral_slug"], sort=True):
        negative = group[group["A_agrees_num"].eq(0.0)]
        positive = group[group["A_agrees_num"].eq(2.0)]
        if negative.empty or positive.empty:
            continue
        candidates = []
        for negative_index, negative_row in negative.iterrows():
            for positive_index, positive_row in positive.iterrows():
                same_rule = negative_row["rot_norm"] == positive_row["rot_norm"]
                rule_jaccard = token_jaccard(negative_row["rot"], positive_row["rot"])
                rule_sequence = sequence_ratio(negative_row["rot"], positive_row["rot"])
                if same_rule:
                    conflict_rows.append(
                        {
                            "split": split,
                            "moral_axis": moral_slug,
                            "qa": negative_row["QA"],
                            "rule": negative_row["rot"],
                            "negative_source_id": negative_row["source_row_id"],
                            "positive_source_id": positive_row["source_row_id"],
                        }
                    )
                    continue
                candidates.append(
                    (
                        0.65 * rule_jaccard + 0.35 * rule_sequence,
                        rule_jaccard,
                        rule_sequence,
                        int(negative_index),
                        int(positive_index),
                    )
                )
        candidates.sort(reverse=True)
        used_negative = set()
        used_positive = set()
        kept = 0
        for rule_match_score, rule_jaccard, rule_sequence, negative_index, positive_index in candidates:
            if kept >= int(max_pairs_per_dialogue_axis):
                break
            if negative_index in used_negative or positive_index in used_positive:
                continue
            negative_row = rows.loc[negative_index]
            positive_row = rows.loc[positive_index]
            negative_prompt = _format_prompt(negative_row["QA"], negative_row["rot"])
            positive_prompt = _format_prompt(positive_row["QA"], positive_row["rot"])
            quality = pair_quality(negative_prompt, positive_prompt, shared_context=negative_row["QA"])
            pair_type = f"{moral_slug}__disagree_vs_agree"
            pair_id = stable_id("mic_rule_conditioned", split, qa_norm, moral_slug, negative_index, positive_index)
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "dataset": "moral_integrity_corpus",
                    "pair_type": pair_type,
                    "moral_axis": moral_slug,
                    "split": split,
                    "group_key": qa_norm,
                    "negative_item_id": f"mic:{negative_row['source_row_id']}:disagrees",
                    "positive_item_id": f"mic:{positive_row['source_row_id']}:agrees",
                    "negative_source_id": negative_row["source_row_id"],
                    "positive_source_id": positive_row["source_row_id"],
                    "negative_label": "disagrees_with_rule",
                    "positive_label": "agrees_with_rule",
                    "negative_text": negative_prompt,
                    "positive_text": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "positive_prompt": positive_prompt,
                    "shared_dialogue": negative_row["QA"],
                    "negative_rule": negative_row["rot"],
                    "positive_rule": positive_row["rot"],
                    "negative_rot_agreement": float(negative_row["rot_agree_num"]),
                    "positive_rot_agreement": float(positive_row["rot_agree_num"]),
                    "negative_violation_severity": negative_row["violation_severity_num"],
                    "positive_violation_severity": positive_row["violation_severity_num"],
                    "rule_token_jaccard": float(rule_jaccard),
                    "rule_sequence_ratio": float(rule_sequence),
                    "rule_match_score": float(rule_match_score),
                    "exact_dialogue_match": True,
                    **quality,
                }
            )
            used_negative.add(negative_index)
            used_positive.add(positive_index)
            kept += 1
    pairs = pd.DataFrame(pair_rows)
    item_rows = []
    for _, pair in pairs.iterrows():
        for side in ["negative", "positive"]:
            item_rows.append(
                {
                    "item_id": pair[f"{side}_item_id"],
                    "dataset": pair["dataset"],
                    "pair_id": pair["pair_id"],
                    "pair_type": pair["pair_type"],
                    "moral_axis": pair["moral_axis"],
                    "split": pair["split"],
                    "side": side,
                    "label": pair[f"{side}_label"],
                    "text": pair[f"{side}_text"],
                    "prompt": pair[f"{side}_prompt"],
                }
            )
    return pd.DataFrame(item_rows), pairs, pd.DataFrame(conflict_rows).drop_duplicates()


def _dialogue_overlap_table(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["left_split", "right_split", "n_shared_dialogues"])
    groups = {str(split): set(group["group_key"].astype(str)) for split, group in pairs.groupby("split")}
    rows = []
    names = sorted(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "n_shared_dialogues": len(groups[left] & groups[right]),
                }
            )
    return pd.DataFrame(rows)


def run_mic_pair_audit(
    *,
    mic_path: str | Path,
    output_dir: str | Path,
    min_rot_agreement: float = 3.0,
    exclude_cross_split_dialogues: bool = True,
    max_pairs_per_dialogue_axis: int = 1,
    min_axis_train_pairs: int = 128,
    min_axis_eval_pairs: int = 20,
) -> dict[str, pd.DataFrame]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, source_inventory = prepare_mic_rows(
        mic_path,
        min_rot_agreement=min_rot_agreement,
        exclude_cross_split_dialogues=exclude_cross_split_dialogues,
    )
    items, pairs, conflicts = build_mic_rule_conditioned_pairs(
        rows,
        max_pairs_per_dialogue_axis=max_pairs_per_dialogue_axis,
    )
    pair_inventory = (
        pairs.groupby(["moral_axis", "pair_type", "split", "quality_bin"], as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_quality_score=("quality_score", "mean"),
            mean_rule_jaccard=("rule_token_jaccard", "mean"),
            mean_rule_sequence_ratio=("rule_sequence_ratio", "mean"),
        )
        .sort_values(["moral_axis", "split", "quality_bin"])
    )
    axis_coverage = pairs.pivot_table(
        index="moral_axis",
        columns="split",
        values="pair_id",
        aggfunc="nunique",
        fill_value=0,
    ).reset_index()
    axis_coverage.columns.name = None
    for split in ["train", "validation", "test"]:
        if split not in axis_coverage:
            axis_coverage[split] = 0
    axis_coverage["decomposition_ready"] = (
        axis_coverage["train"].ge(int(min_axis_train_pairs))
        & axis_coverage["validation"].ge(int(min_axis_eval_pairs))
        & axis_coverage["test"].ge(int(min_axis_eval_pairs))
    )
    axis_coverage = axis_coverage.sort_values(["decomposition_ready", "test", "train"], ascending=False)
    ready_axes = int(axis_coverage["decomposition_ready"].sum())
    decision = (
        "candidate_for_small_rule_conditioned_decomposition"
        if ready_axes >= 3
        else "candidate_for_rule_conditioned_raw_steering_only"
    )
    recommendation = pd.DataFrame(
        [
            {
                "dataset": "moral_integrity_corpus",
                "construction": "same_dialogue_same_moral_different_rule",
                "n_pairs": int(len(pairs)),
                "n_ready_moral_axes": ready_axes,
                "decision": decision,
                "claim_boundary": "rule-conditioned norm adherence, not action-level moral acceptability",
            }
        ]
    )
    quality_summary = pd.DataFrame(
        [
            {
                "n_pairs": int(len(pairs)),
                "high_quality_rate": float(pairs["quality_bin"].eq("high").mean()),
                "medium_or_high_rate": float(pairs["quality_bin"].isin(["medium", "high"]).mean()),
                "mean_quality_score": float(pairs["quality_score"].mean()),
                "mean_rule_jaccard": float(pairs["rule_token_jaccard"].mean()),
                "exact_rule_label_conflicts": int(len(conflicts)),
            }
        ]
    )
    examples_good = pairs.sort_values(["quality_score", "rule_match_score"], ascending=False).head(40)
    examples_bad = pairs.sort_values(["quality_score", "rule_match_score"], ascending=True).head(40)
    tables = {
        "source_inventory": source_inventory,
        "items": items,
        "pairs": pairs,
        "pair_inventory": pair_inventory,
        "axis_coverage": axis_coverage,
        "quality_summary": quality_summary,
        "exact_rule_label_conflicts": conflicts,
        "endpoint_overlap": endpoint_overlap_table(pairs),
        "dialogue_overlap": _dialogue_overlap_table(pairs),
        "examples_good": examples_good,
        "examples_bad": examples_bad,
        "recommendation": recommendation,
    }
    write_tables(tables, out)
    report = [
        "# MIC Rule-Conditioned Pair Audit",
        "",
        f"- MIC path: `{Path(mic_path).expanduser().resolve()}`",
        f"- Constructed pairs: `{len(pairs)}`",
        f"- Decomposition-ready moral axes: `{ready_axes}`",
        f"- Decision: `{decision}`",
        "",
        "Pairs keep the dialogue and moral axis fixed while changing the rule of thumb.",
        "The resulting claim concerns rule-conditioned norm adherence rather than action-level moral acceptability.",
    ]
    (out / "mic_pair_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "recommendation": recommendation,
        "quality_summary": quality_summary,
        "axis_coverage": axis_coverage,
        "pair_inventory": pair_inventory,
        "endpoint_overlap": tables["endpoint_overlap"],
        "dialogue_overlap": tables["dialogue_overlap"],
    }
