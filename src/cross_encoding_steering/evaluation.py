from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .decomposition import contrast_directions, decompose_directions
from .io import write_tables
from .metrics import js_divergence
from .pairs import stable_id
from .steering import (
    ChoiceSpec,
    build_binary_items_from_pairs,
    collect_choice_probs,
    collect_choice_probs_and_activations,
    collect_choice_probs_and_position_activations,
    indexed_pairs,
    patched_choice_probs,
    resolve_layer_index,
    run_binary_direction_validation,
)


PAPER_MODES = [
    "raw_caa_direction",
    "shared_only_direction",
    "residual_only_direction",
    "mixed_shared_residual",
    "loco_shared_direction",
    "loco_residual_direction",
]
CONTROL_MODES = ["random_direction_control", "wrong_direction_control", "zero_direction_control"]


@dataclass(frozen=True)
class EvaluationSpec:
    dataset_name: str
    group_column: str
    group_values: tuple[str, ...]
    negative_label: str
    positive_label: str
    scope: str = "cross_interface_steering"
    report_title: str = "Norm Direction Decomposition Evaluation"


def select_balanced_pairs(
    pairs: pd.DataFrame,
    *,
    group_column: str,
    group_values: list[str] | tuple[str, ...],
    max_train_pairs_per_group: int,
    max_validation_pairs_per_group: int,
    max_test_pairs_per_group: int,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_column not in pairs.columns:
        raise ValueError(f"Pair table is missing configured group column {group_column!r}")
    limits = {
        "train": int(max_train_pairs_per_group),
        "validation": int(max_validation_pairs_per_group),
        "test": int(max_test_pairs_per_group),
    }
    configured_groups = [str(value) for value in group_values]
    rows = pairs.copy()
    if configured_groups:
        rows = rows[rows[group_column].astype(str).isin(configured_groups)].copy()
    selected = []
    inventory = []
    for (group_value, split), group in rows.groupby([group_column, "split"], sort=True):
        limit = limits.get(str(split), len(group))
        if len(group) > limit:
            random_state = (int(stable_id(group_value, split, seed), 16) + int(seed)) % (2**32 - 1)
            group = group.sample(n=limit, random_state=random_state)
        selected.append(group)
        inventory.append(
            {
                "group_column": group_column,
                "group_value": group_value,
                "split": split,
                "n_selected_pairs": int(len(group)),
                "mean_quality_score": float(group["quality_score"].mean())
                if "quality_score" in group.columns
                else np.nan,
            }
        )
    if not selected:
        raise ValueError("No pairs remain after configured group/split selection")
    return pd.concat(selected, ignore_index=True), pd.DataFrame(inventory)


def _aggregate_settings(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["mode", "layer_index", "alpha", "subspace_dim", "mix_lambda"]
    out = (
        summary.groupby(group_cols, as_index=False)
        .agg(
            n_conditions=("pair_type", "size"),
            mean_delta_intended_acc=("delta_intended_acc", "mean"),
            mean_delta_target_prob=("mean_delta_target_prob", "mean"),
            mean_js_shift=("mean_js_shift", "mean"),
            mean_prediction_changed_rate=("prediction_changed_rate", "mean"),
        )
        .sort_values(group_cols)
    )
    out["selection_score"] = out["mean_delta_intended_acc"] - 0.25 * out["mean_js_shift"]
    return out


def select_validation_settings(
    validation_summary: pd.DataFrame,
    *,
    default_subspace_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    raw_rows = validation_summary[
        validation_summary["mode"].eq("raw_caa_direction")
        & validation_summary["subspace_dim"].eq(int(default_subspace_dim))
    ].copy()
    raw_aggregate = _aggregate_settings(raw_rows)
    if raw_aggregate.empty:
        raise ValueError("No raw validation rows were produced")
    best_raw = raw_aggregate.sort_values(
        ["selection_score", "mean_delta_intended_acc", "mean_js_shift"],
        ascending=[False, False, True],
    ).iloc[0]
    selected_layer = int(best_raw["layer_index"])

    layer_rows = validation_summary[validation_summary["layer_index"].eq(selected_layer)].copy()
    candidates = []
    aggregate_parts = []
    for mode in PAPER_MODES:
        mode_rows = layer_rows[layer_rows["mode"].eq(mode)].copy()
        if mode == "raw_caa_direction":
            mode_rows = mode_rows[mode_rows["subspace_dim"].eq(int(default_subspace_dim))]
        aggregate = _aggregate_settings(mode_rows)
        aggregate_parts.append(aggregate)
        if aggregate.empty:
            continue
        best = aggregate.sort_values(
            ["selection_score", "mean_delta_intended_acc", "mean_js_shift"],
            ascending=[False, False, True],
        ).iloc[0]
        candidates.append(best.to_dict())
    selected = pd.DataFrame(candidates)
    raw_setting = selected[selected["mode"].eq("raw_caa_direction")].iloc[0]
    for mode in CONTROL_MODES:
        candidates.append(
            {
                "mode": mode,
                "layer_index": selected_layer,
                "alpha": float(raw_setting["alpha"]),
                "subspace_dim": 0 if mode == "zero_direction_control" else int(default_subspace_dim),
                "mix_lambda": 1.0,
                "selection_score": np.nan,
                "selection_source": "raw_caa_setting",
            }
        )
    selected = pd.DataFrame(candidates)
    selected["selection_source"] = selected.get("selection_source", pd.Series(index=selected.index, dtype=object)).fillna(
        "validation_objective"
    )
    all_aggregates = pd.concat(aggregate_parts, ignore_index=True, sort=False)
    return selected, all_aggregates, selected_layer


def filter_locked_test_rows(test_summary: pd.DataFrame, selected_settings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, setting in selected_settings.iterrows():
        mask = (
            test_summary["mode"].eq(setting["mode"])
            & test_summary["layer_index"].eq(int(setting["layer_index"]))
            & np.isclose(test_summary["alpha"], float(setting["alpha"]))
            & test_summary["subspace_dim"].eq(int(setting["subspace_dim"]))
            & np.isclose(test_summary["mix_lambda"], float(setting["mix_lambda"]))
        )
        part = test_summary[mask].copy()
        part["selection_source"] = setting["selection_source"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def filter_locked_test_pair_rows(test_rows: pd.DataFrame, selected_settings: pd.DataFrame) -> pd.DataFrame:
    """Keep row-level test effects for settings selected only on validation data."""
    return filter_locked_test_rows(test_rows, selected_settings)


def summarize_locked_test(locked_rows: pd.DataFrame) -> pd.DataFrame:
    if locked_rows.empty:
        return pd.DataFrame()
    return (
        locked_rows.groupby("mode", as_index=False)
        .agg(
            n_conditions=("pair_type", "size"),
            mean_delta_intended_acc=("delta_intended_acc", "mean"),
            mean_delta_target_prob=("mean_delta_target_prob", "mean"),
            mean_js_shift=("mean_js_shift", "mean"),
            mean_prediction_changed_rate=("prediction_changed_rate", "mean"),
        )
        .sort_values("mean_delta_intended_acc", ascending=False)
    )


def run_locked_test_row_inference(
    model: Any,
    tokenizer: Any,
    selected_pairs: pd.DataFrame,
    selected_settings: pd.DataFrame,
    *,
    output_dir: str | Path,
    negative_label: str,
    positive_label: str,
    prompt_variants: list[str],
    eval_directions: list[str],
    batch_size: int = 2,
    max_length: int = 512,
    seed: int = 13,
) -> dict[str, pd.DataFrame]:
    """Recompute only row-level locked-test outputs for an existing experiment."""
    if selected_pairs.empty or selected_settings.empty:
        raise ValueError("selected_pairs and selected_settings must be non-empty")
    layers = sorted(selected_settings["layer_index"].dropna().astype(int).unique())
    if len(layers) != 1:
        raise ValueError(f"Expected one locked layer, found {layers}")
    paper_settings = selected_settings[selected_settings["mode"].isin(PAPER_MODES)].copy()
    if paper_settings.empty:
        raise ValueError("No paper modes found in selected_settings")
    alpha_grid = sorted(selected_settings["alpha"].dropna().astype(float).unique())
    subspace_dims = sorted(
        value for value in paper_settings["subspace_dim"].dropna().astype(int).unique() if value > 0
    )
    mixing_grid = sorted(paper_settings["mix_lambda"].dropna().astype(float).unique())
    if not subspace_dims:
        raise ValueError("No positive subspace dimension found in selected_settings")
    if not mixing_grid:
        raise ValueError("No mixing coefficient found in selected_settings")
    tables = run_binary_direction_validation(
        model,
        tokenizer,
        selected_pairs,
        layer_index=layers[0],
        alpha_grid=alpha_grid,
        subspace_dims=subspace_dims,
        mixing_grid=mixing_grid,
        train_split="train",
        eval_split="test",
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
        prompt_variants=prompt_variants,
        eval_directions=eval_directions,
        include_decomposition_modes=True,
        negative_label=negative_label,
        positive_label=positive_label,
    )
    locked_pair_rows = filter_locked_test_pair_rows(tables["eval_rows"], selected_settings)
    if locked_pair_rows.empty:
        raise ValueError("Locked settings did not match any recomputed test rows")
    outputs = {
        "test_eval_rows": tables["eval_rows"],
        "locked_test_pair_rows": locked_pair_rows,
    }
    write_tables(outputs, Path(output_dir).expanduser().resolve())
    return outputs


def _choice_spec(prompt_variant: str, *, negative_label: str, positive_label: str) -> ChoiceSpec:
    if prompt_variant == "canonical":
        return ChoiceSpec(
            negative_label=negative_label,
            positive_label=positive_label,
            negative_letter="A",
            positive_letter="B",
        )
    if prompt_variant == "label_order_flip":
        return ChoiceSpec(
            negative_label=negative_label,
            positive_label=positive_label,
            negative_letter="B",
            positive_letter="A",
        )
    raise ValueError(f"Unknown prompt variant: {prompt_variant}")


def evaluate_cross_group_transfer(
    model: Any,
    tokenizer: Any,
    pairs: pd.DataFrame,
    *,
    selected_layer: int,
    selected_settings: pd.DataFrame,
    dataset_name: str,
    negative_label: str,
    positive_label: str,
    prompt_variants: list[str],
    eval_directions: list[str],
    batch_size: int,
    max_length: int,
    seed: int,
    include_loco: bool = False,
    loco_subspace_dim: int = 1,
    extraction_position: str = "pre_answer",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_paper = selected_settings[selected_settings["mode"].isin(PAPER_MODES)].copy()
    needed_dims = sorted({int(value) for value in selected_paper["subspace_dim"]})
    needed_mixes = sorted({float(value) for value in selected_paper["mix_lambda"]})
    transfer_rows = []
    direction_inventories = []
    resolved_layer = int(selected_layer)

    for prompt_variant in prompt_variants:
        choice_spec = _choice_spec(
            prompt_variant,
            negative_label=negative_label,
            positive_label=positive_label,
        )
        items = build_binary_items_from_pairs(pairs, choice_spec=choice_spec, prompt_variant=prompt_variant)
        items["dataset"] = dataset_name
        indexed = indexed_pairs(pairs, items)
        offsets = (
            items["scenario_char_end"].astype(int).tolist()
            if extraction_position == "scenario_end"
            else [None] * len(items)
        )
        _, position_activations = collect_choice_probs_and_position_activations(
            model,
            tokenizer,
            items["prompt"].tolist(),
            position_character_offsets={extraction_position: offsets},
            layer_indices=[resolved_layer],
            batch_size=batch_size,
            max_length=max_length,
            choice_letters=["A", "B"],
        )
        activations = position_activations[extraction_position]
        raw, raw_inventory = contrast_directions(
            indexed,
            activations[resolved_layer],
            negative_column="negative_item_index",
            positive_column="positive_item_index",
            split_column="split",
            train_split="train",
        )
        bank = decompose_directions(
            raw,
            subspace_dims=needed_dims,
            mixing_grid=needed_mixes,
            seed=seed,
            include_loco=include_loco,
            loco_subspace_dim=loco_subspace_dim,
        )
        inventory = bank.inventory.copy()
        inventory["prompt_variant"] = prompt_variant
        inventory["layer_index"] = resolved_layer
        inventory["extraction_position"] = extraction_position
        direction_inventories.append(inventory)

        target_pair_types = sorted(pairs["pair_type"].astype(str).unique())
        baseline_cache = {}
        for target_pair_type in target_pair_types:
            for eval_direction in eval_directions:
                side = "negative" if eval_direction == "negative_to_positive" else "positive"
                target_items = items[
                    items["split"].astype(str).eq("test")
                    & items["side"].eq(side)
                    & items["pair_type"].astype(str).eq(target_pair_type)
                ].copy()
                prompts = target_items["prompt"].tolist()
                base = collect_choice_probs(
                    model,
                    tokenizer,
                    prompts,
                    batch_size=batch_size,
                    max_length=max_length,
                    choice_letters=["A", "B"],
                )
                target_choice = (
                    choice_spec.positive_choice_index
                    if eval_direction == "negative_to_positive"
                    else choice_spec.negative_choice_index
                )
                baseline_cache[(target_pair_type, eval_direction)] = (target_items, prompts, base, int(target_choice))

        for _, setting in selected_paper.iterrows():
            mode = str(setting["mode"])
            key_dim = int(setting["subspace_dim"])
            key_mix = float(setting["mix_lambda"])
            alpha = float(setting["alpha"])
            for source_pair_type in target_pair_types:
                key = (source_pair_type, mode, key_dim, key_mix)
                vector = bank.vectors.get(key)
                if vector is None:
                    continue
                for target_pair_type in target_pair_types:
                    for eval_direction in eval_directions:
                        target_items, prompts, base, target_choice = baseline_cache[(target_pair_type, eval_direction)]
                        applied_vector = vector if eval_direction == "negative_to_positive" else -vector
                        patched = patched_choice_probs(
                            model,
                            tokenizer,
                            prompts,
                            layer_index=resolved_layer,
                            direction=applied_vector,
                            alpha=alpha,
                            batch_size=batch_size,
                            max_length=max_length,
                            choice_letters=["A", "B"],
                        )
                        base_pred = base.argmax(axis=1)
                        patched_pred = patched.argmax(axis=1)
                        target = np.full(len(base), target_choice, dtype=int)
                        transfer_rows.append(
                            {
                                "prompt_variant": prompt_variant,
                                "eval_direction": eval_direction,
                                "mode": mode,
                                "layer_index": resolved_layer,
                                "alpha": alpha,
                                "subspace_dim": key_dim,
                                "mix_lambda": key_mix,
                                "source_pair_type": source_pair_type,
                                "target_pair_type": target_pair_type,
                                "is_self_group": source_pair_type == target_pair_type,
                                "n_items": int(len(target_items)),
                                "base_intended_acc": float((base_pred == target).mean()),
                                "patched_intended_acc": float((patched_pred == target).mean()),
                                "delta_intended_acc": float((patched_pred == target).mean() - (base_pred == target).mean()),
                                "mean_delta_target_prob": float((patched[:, target_choice] - base[:, target_choice]).mean()),
                                "mean_js_shift": float(np.mean([js_divergence(base[i], patched[i]) for i in range(len(base))])),
                                "prediction_changed_rate": float((patched_pred != base_pred).mean()),
                            }
                        )
    transfer = pd.DataFrame(transfer_rows)
    specificity = (
        transfer.groupby(["mode", "source_pair_type", "is_self_group"], as_index=False)
        .agg(
            mean_delta_intended_acc=("delta_intended_acc", "mean"),
            mean_delta_target_prob=("mean_delta_target_prob", "mean"),
            mean_js_shift=("mean_js_shift", "mean"),
            n_conditions=("target_pair_type", "size"),
        )
    )
    pivot = specificity.pivot_table(
        index=["mode", "source_pair_type"],
        columns="is_self_group",
        values=["mean_delta_intended_acc", "mean_delta_target_prob", "mean_js_shift"],
    )
    pivot.columns = [f"{metric}_{'self' if is_self else 'cross'}" for metric, is_self in pivot.columns]
    pivot = pivot.reset_index()
    pivot["accuracy_specificity_gap"] = pivot.get("mean_delta_intended_acc_self", np.nan) - pivot.get(
        "mean_delta_intended_acc_cross", np.nan
    )
    denominator = pivot.get("mean_delta_target_prob_self", pd.Series(np.nan, index=pivot.index)).abs().clip(lower=1e-8)
    pivot["target_prob_transfer_ratio"] = pivot.get("mean_delta_target_prob_cross", np.nan).abs() / denominator
    inventory_all = pd.concat(direction_inventories, ignore_index=True, sort=False) if direction_inventories else pd.DataFrame()
    return transfer, pivot, inventory_all


def run_decomposition_evaluation(
    model: Any,
    tokenizer: Any,
    pairs: pd.DataFrame,
    *,
    spec: EvaluationSpec,
    output_dir: str | Path,
    layer_indices: list[int],
    alpha_grid: list[float],
    subspace_dims: list[int],
    mixing_grid: list[float],
    prompt_variants: list[str],
    eval_directions: list[str],
    max_train_pairs_per_group: int = 256,
    max_validation_pairs_per_group: int = 128,
    max_test_pairs_per_group: int = 128,
    batch_size: int = 4,
    max_length: int = 512,
    seed: int = 13,
    include_loco: bool = False,
    loco_subspace_dim: int = 1,
    extraction_position: str = "pre_answer",
) -> dict[str, pd.DataFrame]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"
    selected_pairs, pair_inventory = select_balanced_pairs(
        pairs,
        group_column=spec.group_column,
        group_values=spec.group_values,
        max_train_pairs_per_group=max_train_pairs_per_group,
        max_validation_pairs_per_group=max_validation_pairs_per_group,
        max_test_pairs_per_group=max_test_pairs_per_group,
        seed=seed,
    )
    n_layers = len(getattr(getattr(model, "model", model), "layers", []))
    if not n_layers:
        from .steering import decoder_layers

        n_layers = len(decoder_layers(model))
    resolved_layers = sorted({resolve_layer_index(layer, n_layers) for layer in layer_indices})

    validation_parts = []
    for layer in resolved_layers:
        tables = run_binary_direction_validation(
            model,
            tokenizer,
            selected_pairs,
            layer_index=layer,
            alpha_grid=alpha_grid,
            subspace_dims=subspace_dims,
            mixing_grid=mixing_grid,
            train_split="train",
            eval_split="validation",
            batch_size=batch_size,
            max_length=max_length,
            seed=seed,
            prompt_variants=prompt_variants,
            eval_directions=eval_directions,
            include_decomposition_modes=True,
            include_loco=include_loco,
            loco_subspace_dim=loco_subspace_dim,
            extraction_position=extraction_position,
            negative_label=spec.negative_label,
            positive_label=spec.positive_label,
        )
        validation_parts.append(tables["eval_summary"])
    validation_summary = pd.concat(validation_parts, ignore_index=True, sort=False)
    default_dim = min(int(value) for value in subspace_dims)
    selected_settings, validation_setting_summary, selected_layer = select_validation_settings(
        validation_summary,
        default_subspace_dim=default_dim,
    )
    selected_settings["extraction_position"] = extraction_position
    validation_setting_summary["extraction_position"] = extraction_position

    test_tables = run_binary_direction_validation(
        model,
        tokenizer,
        selected_pairs,
        layer_index=selected_layer,
        alpha_grid=alpha_grid,
        subspace_dims=subspace_dims,
        mixing_grid=mixing_grid,
        train_split="train",
        eval_split="test",
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
        prompt_variants=prompt_variants,
        eval_directions=eval_directions,
        include_decomposition_modes=True,
        include_loco=include_loco,
        loco_subspace_dim=loco_subspace_dim,
        extraction_position=extraction_position,
        negative_label=spec.negative_label,
        positive_label=spec.positive_label,
    )
    locked_rows = filter_locked_test_rows(test_tables["eval_summary"], selected_settings)
    locked_pair_rows = filter_locked_test_pair_rows(test_tables["eval_rows"], selected_settings)
    locked_summary = summarize_locked_test(locked_rows)
    transfer, specificity, selected_direction_inventory = evaluate_cross_group_transfer(
        model,
        tokenizer,
        selected_pairs,
        selected_layer=selected_layer,
        selected_settings=selected_settings,
        dataset_name=spec.dataset_name,
        negative_label=spec.negative_label,
        positive_label=spec.positive_label,
        prompt_variants=prompt_variants,
        eval_directions=eval_directions,
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
        include_loco=include_loco,
        loco_subspace_dim=loco_subspace_dim,
        extraction_position=extraction_position,
    )
    specificity_by_mode = (
        specificity.groupby("mode", as_index=False)
        .agg(
            mean_accuracy_specificity_gap=("accuracy_specificity_gap", "mean"),
            mean_target_prob_transfer_ratio=("target_prob_transfer_ratio", "mean"),
            mean_self_target_prob=("mean_delta_target_prob_self", "mean"),
            mean_cross_target_prob=("mean_delta_target_prob_cross", "mean"),
        )
        .sort_values("mean_target_prob_transfer_ratio")
    )
    mode_values = locked_summary.set_index("mode")["mean_delta_intended_acc"].to_dict() if not locked_summary.empty else {}
    raw_margin = float(mode_values.get("raw_caa_direction", np.nan) - mode_values.get("random_direction_control", np.nan))
    decision = pd.DataFrame(
        [
            {
                "selected_layer": selected_layer,
                "n_selected_pairs": int(len(selected_pairs)),
                "raw_minus_random_accuracy": raw_margin,
                "raw_signal_supported": bool(np.isfinite(raw_margin) and raw_margin >= 0.03),
                "scope": spec.scope,
                "extraction_position": extraction_position,
                "loco_enabled": bool(include_loco),
            }
        ]
    )
    outputs = {
        "selected_pairs": selected_pairs,
        "pair_selection_inventory": pair_inventory,
        "validation_eval_summary": validation_summary,
        "validation_setting_summary": validation_setting_summary,
        "selected_settings": selected_settings,
        "test_eval_rows": test_tables["eval_rows"],
        "test_eval_summary": test_tables["eval_summary"],
        "locked_test_pair_rows": locked_pair_rows,
        "locked_test_rows": locked_rows,
        "locked_test_summary": locked_summary,
        "cross_group_transfer": transfer,
        "specificity_by_group": specificity,
        "specificity_by_mode": specificity_by_mode,
        "selected_direction_inventory": selected_direction_inventory,
        "decision": decision,
    }
    write_tables(outputs, out)
    report = [
        f"# {spec.report_title}",
        "",
        f"- Selected layer: `{selected_layer}`",
        f"- Selected pairs: `{len(selected_pairs)}`",
        f"- Raw minus random accuracy: `{raw_margin:.4f}`",
        f"- Extraction position: `{extraction_position}`",
        f"- LOCO decomposition: `{include_loco}`",
        "",
        "Hyperparameters are selected on validation and evaluated once on the locked test split.",
        "Cross-group transfer is reported separately from harmful leakage; interpretation depends on the configured groups.",
    ]
    (out / "decomposition_evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    return outputs
