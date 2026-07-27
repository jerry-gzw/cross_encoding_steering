"""Baseline competence and identifier-position-semantics factorial audit.

The direction is extracted once from the canonical NormBank interface. During
evaluation, semantic assignment, answer identifier vocabulary, and displayed
row order are varied independently.
"""
from __future__ import annotations

import itertools
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cross_interface_audit import PromptTemplate, _build_eval_items
from .io import write_tables
from .mapping_audit import (
    FixedDirectionMappingAuditConfig,
    MappingDefinition,
    _load_mapping_audit_config,
    build_canonical_direction_bank,
    build_mapping_items,
    build_scenario_context,
    load_base_items,
)
from .steering import (
    cleanup_model,
    completion_sequence_logprobs,
    load_tokenizer_and_model,
)
from .statistics import holm_adjust, paired_sign_flip_pvalue


@dataclass(frozen=True)
class IdentifierSet:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class RowOrder:
    name: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class FactorialCondition:
    mapping: MappingDefinition
    identifier_set: IdentifierSet
    row_order: RowOrder

    @property
    def name(self) -> str:
        return (
            f"{self.mapping.name}__{self.identifier_set.name}__"
            f"{self.row_order.name}"
        )


@dataclass(frozen=True)
class InterfaceFactorialConfig:
    audit: FixedDirectionMappingAuditConfig
    templates: tuple[PromptTemplate, ...]
    mapping_names: tuple[str, ...]
    identifier_sets: tuple[IdentifierSet, ...]
    row_orders: tuple[RowOrder, ...]
    key_templates: tuple[str, ...]
    competence_threshold: float

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "InterfaceFactorialConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        audit = FixedDirectionMappingAuditConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        spec = dict(data.get("interface_factorial", {}))
        templates = tuple(
            PromptTemplate(
                name=str(item["name"]),
                question=str(item["question"]),
                answer_instruction=str(item["answer_instruction"]),
            )
            for item in spec.get(
                "templates",
                [
                    {
                        "name": "primary",
                        "question": audit.dataset.question,
                        "answer_instruction": audit.dataset.answer_instruction,
                    }
                ],
            )
        )
        mapping_names = tuple(
            str(value)
            for value in spec.get(
                "mapping_names",
                [
                    mapping.name
                    for mapping in audit.mappings
                    if mapping.family == "semantic_words"
                ],
            )
        )
        mapping_lookup = {mapping.name: mapping for mapping in audit.mappings}
        unknown_mappings = sorted(set(mapping_names) - set(mapping_lookup))
        if unknown_mappings:
            raise ValueError(f"Unknown factorial mappings: {unknown_mappings}")
        n_labels = len(audit.dataset.label_ranks)
        identifier_sets = tuple(
            IdentifierSet(
                name=str(item["name"]),
                values=tuple(str(value) for value in item["values"]),
            )
            for item in spec.get("identifier_sets", [])
        )
        if not identifier_sets:
            raise ValueError("interface_factorial.identifier_sets cannot be empty")
        for identifiers in identifier_sets:
            if len(identifiers.values) != n_labels:
                raise ValueError(
                    f"Identifier set {identifiers.name!r} must have {n_labels} values"
                )
            if len(set(identifiers.values)) != n_labels:
                raise ValueError(
                    f"Identifier set {identifiers.name!r} contains duplicates"
                )
        configured_orders = spec.get("row_orders")
        if configured_orders:
            row_orders = tuple(
                RowOrder(
                    name=str(item["name"]),
                    indices=tuple(int(value) for value in item["indices"]),
                )
                for item in configured_orders
            )
        else:
            row_orders = tuple(
                RowOrder(
                    name="order_" + "".join(str(index + 1) for index in order),
                    indices=tuple(order),
                )
                for order in itertools.permutations(range(n_labels))
            )
        expected_indices = tuple(range(n_labels))
        for order in row_orders:
            if tuple(sorted(order.indices)) != expected_indices:
                raise ValueError(
                    f"Row order {order.name!r} must be a permutation of "
                    f"{expected_indices}"
                )
        key_templates = tuple(
            str(value)
            for value in spec.get(
                "key_templates",
                [
                    "Which identifier means {label}?",
                    "Return the identifier assigned to {label}.",
                    "If the correct category is {label}, what should be returned?",
                ],
            )
        )
        threshold = float(spec.get("competence_threshold", 0.8))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("interface_factorial.competence_threshold must be in [0, 1]")
        return cls(
            audit=audit,
            templates=templates,
            mapping_names=mapping_names,
            identifier_sets=identifier_sets,
            row_orders=row_orders,
            key_templates=key_templates,
            competence_threshold=threshold,
        )


def _canonical_mapping(config: InterfaceFactorialConfig) -> MappingDefinition:
    for mapping in config.audit.mappings:
        if mapping.name == config.audit.dataset.canonical_mapping:
            return mapping
    raise KeyError("Canonical mapping is missing")


def _factorial_conditions(
    config: InterfaceFactorialConfig,
) -> tuple[FactorialCondition, ...]:
    mappings = {mapping.name: mapping for mapping in config.audit.mappings}
    return tuple(
        FactorialCondition(mappings[mapping_name], identifiers, row_order)
        for mapping_name in config.mapping_names
        for identifiers in config.identifier_sets
        for row_order in config.row_orders
    )


def _identifier_index_by_label(
    condition: FactorialCondition,
) -> dict[str, int]:
    return {
        label: index
        for index, label in enumerate(condition.mapping.option_order)
    }


def _label_by_identifier_index(
    condition: FactorialCondition,
) -> dict[int, str]:
    return {
        index: label
        for index, label in enumerate(condition.mapping.option_order)
    }


def _prompt_and_candidates(
    row: pd.Series,
    config: InterfaceFactorialConfig,
    template: PromptTemplate,
    condition: FactorialCondition,
) -> tuple[str, dict[str, str]]:
    label_to_index = _identifier_index_by_label(condition)
    index_to_label = _label_by_identifier_index(condition)
    lines = [build_scenario_context(row, config.audit.dataset), template.question]
    for identifier_index in condition.row_order.indices:
        label = index_to_label[identifier_index]
        identifier = condition.identifier_set.values[identifier_index]
        lines.append(f"{identifier}. {condition.mapping.option_texts[label]}")
    valid = ", ".join(condition.identifier_set.values)
    lines.extend(
        [
            template.answer_instruction,
            f"Return exactly one identifier from: {valid}.",
            "Answer:",
        ]
    )
    candidates = {
        label: condition.identifier_set.values[identifier_index]
        for label, identifier_index in label_to_index.items()
    }
    return "\n".join(lines), candidates


def _attribution_labels(
    condition: FactorialCondition,
    canonical: MappingDefinition,
    *,
    target_label: str,
    source_label: str,
) -> dict[str, tuple[str, str]]:
    """Labels for semantic, identifier-index, and row-position attribution."""
    current_by_identifier = _label_by_identifier_index(condition)
    canonical_target_index = canonical.option_order.index(target_label)
    canonical_source_index = canonical.option_order.index(source_label)

    identifier_target = current_by_identifier[canonical_target_index]
    identifier_source = current_by_identifier[canonical_source_index]

    identifier_at_target_row = condition.row_order.indices[canonical_target_index]
    identifier_at_source_row = condition.row_order.indices[canonical_source_index]
    row_target = current_by_identifier[identifier_at_target_row]
    row_source = current_by_identifier[identifier_at_source_row]
    return {
        "semantic": (target_label, source_label),
        "extraction_identifier": (identifier_target, identifier_source),
        "extraction_row": (row_target, row_source),
    }


def _macro_f1(y_true: Iterable[str], y_pred: Iterable[str], labels: Iterable[str]) -> float:
    truth = np.asarray(list(y_true), dtype=object)
    pred = np.asarray(list(y_pred), dtype=object)
    scores: list[float] = []
    for label in labels:
        tp = int(np.sum((truth == label) & (pred == label)))
        fp = int(np.sum((truth != label) & (pred == label)))
        fn = int(np.sum((truth == label) & (pred != label)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return float(np.mean(scores))


def _key_competence_items(
    config: InterfaceFactorialConfig,
    condition: FactorialCondition,
) -> pd.DataFrame:
    labels = list(config.audit.dataset.label_ranks)
    index_by_label = _identifier_index_by_label(condition)
    key = "; ".join(
        f"{condition.identifier_set.values[index_by_label[label]]} means {label}"
        for label in labels
    )
    rows: list[dict[str, Any]] = []
    for template_index, question in enumerate(config.key_templates):
        for target_label in labels:
            rows.append(
                {
                    "semantic_mapping": condition.mapping.name,
                    "identifier_set": condition.identifier_set.name,
                    "template": f"key_{template_index + 1}",
                    "target_label": target_label,
                    "target_identifier": condition.identifier_set.values[
                        index_by_label[target_label]
                    ],
                    "prompt": "\n".join(
                        [
                            f"Temporary answer key: {key}.",
                            question.format(label=target_label),
                            "Return exactly one identifier.",
                            "Answer:",
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_key_competence(
    model: Any,
    tokenizer: Any,
    config: InterfaceFactorialConfig,
) -> pd.DataFrame:
    labels = list(config.audit.dataset.label_ranks)
    rows: list[dict[str, Any]] = []
    # Row order cannot affect a scenario-free key, so evaluate each mapping x
    # identifier vocabulary once.
    seen: set[tuple[str, str]] = set()
    for condition in _factorial_conditions(config):
        key = (condition.mapping.name, condition.identifier_set.name)
        if key in seen:
            continue
        seen.add(key)
        items = _key_competence_items(config, condition)
        candidates = [
            [" " + value for value in condition.identifier_set.values]
            for _ in range(len(items))
        ]
        _, mean_scores, _ = completion_sequence_logprobs(
            model,
            tokenizer,
            items["prompt"].tolist(),
            candidates,
            batch_size=config.audit.runtime.batch_size,
            max_length=config.audit.runtime.max_length,
        )
        index_by_label = _identifier_index_by_label(condition)
        for row_index, item in items.iterrows():
            target_index = index_by_label[str(item["target_label"])]
            prediction_index = int(np.argmax(mean_scores[row_index]))
            other_scores = np.delete(mean_scores[row_index], target_index)
            rows.append(
                {
                    **item.to_dict(),
                    "prediction_identifier": condition.identifier_set.values[
                        prediction_index
                    ],
                    "correct": prediction_index == target_index,
                    "target_margin": float(
                        mean_scores[row_index, target_index]
                        - np.max(other_scores)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_condition(
    model: Any,
    tokenizer: Any,
    base_items: pd.DataFrame,
    eval_items: pd.DataFrame,
    directions: dict[tuple[str, str], np.ndarray],
    config: InterfaceFactorialConfig,
    model_config: Any,
    template: PromptTemplate,
    condition: FactorialCondition,
) -> pd.DataFrame:
    labels = list(config.audit.dataset.label_ranks)
    canonical = _canonical_mapping(config)
    prompts: list[str] = []
    candidate_rows: list[list[str]] = []
    for item in eval_items.itertuples():
        prompt, candidates = _prompt_and_candidates(
            base_items.loc[item.row_index],
            config,
            template,
            condition,
        )
        prompts.append(prompt)
        candidate_rows.append([" " + candidates[label] for label in labels])
    base_sum, base_mean, _ = completion_sequence_logprobs(
        model,
        tokenizer,
        prompts,
        candidate_rows,
        batch_size=config.audit.runtime.batch_size,
        max_length=config.audit.runtime.max_length,
    )
    base_prediction = np.argmax(base_mean, axis=1)
    rows: list[dict[str, Any]] = []
    for item_index, item in eval_items.reset_index(drop=True).iterrows():
        gold_index = labels.index(str(item["base_label"]))
        rows.append(
            {
                "record_type": "baseline",
                "pair_id": item["pair_id"],
                "pair_type": item["pair_type"],
                "endpoint": item["endpoint"],
                "template": template.name,
                "condition": condition.name,
                "semantic_mapping": condition.mapping.name,
                "identifier_set": condition.identifier_set.name,
                "row_order": condition.row_order.name,
                "gold_label": item["base_label"],
                "prediction_label": labels[int(base_prediction[item_index])],
                "correct": int(base_prediction[item_index]) == gold_index,
                "base_gold_mean_logprob": float(base_mean[item_index, gold_index]),
            }
        )

    for (pair_type, mode), direction in sorted(directions.items()):
        pair_mask = eval_items["pair_type"].astype(str).eq(pair_type).to_numpy()
        for sign in (1.0, -1.0):
            mask = pair_mask & eval_items["direction_sign"].eq(sign).to_numpy()
            if not mask.any():
                continue
            indices = np.flatnonzero(mask)
            subset = eval_items.loc[mask].reset_index(drop=True)
            _, patched_mean, _ = completion_sequence_logprobs(
                model,
                tokenizer,
                [prompts[index] for index in indices],
                [candidate_rows[index] for index in indices],
                batch_size=config.audit.runtime.batch_size,
                max_length=config.audit.runtime.max_length,
                layer_index=model_config.locked_layer,
                direction=np.asarray(direction, dtype=np.float32) * sign,
                alpha=model_config.locked_alpha,
            )
            base_scores = base_mean[mask]
            for local_index, item in subset.iterrows():
                attribution = _attribution_labels(
                    condition,
                    canonical,
                    target_label=str(item["target_label"]),
                    source_label=str(item["base_label"]),
                )
                effects: dict[str, float] = {}
                for attribution_name, (target_label, source_label) in attribution.items():
                    target_index = labels.index(target_label)
                    source_index = labels.index(source_label)
                    effects[f"{attribution_name}_margin_gain"] = float(
                        patched_mean[local_index, target_index]
                        - patched_mean[local_index, source_index]
                        - base_scores[local_index, target_index]
                        + base_scores[local_index, source_index]
                    )
                rows.append(
                    {
                        "record_type": "steering",
                        "pair_id": item["pair_id"],
                        "pair_type": item["pair_type"],
                        "endpoint": item["endpoint"],
                        "direction_sign": sign,
                        "mode": mode,
                        "template": template.name,
                        "condition": condition.name,
                        "semantic_mapping": condition.mapping.name,
                        "identifier_set": condition.identifier_set.name,
                        "row_order": condition.row_order.name,
                        "base_label": item["base_label"],
                        "target_label": item["target_label"],
                        **effects,
                    }
                )
    return pd.DataFrame(rows)


def _summarize_competence(rows: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    keys = [
        "model_alias",
        "model_name",
        "template",
        "semantic_mapping",
        "identifier_set",
        "row_order",
    ]
    for values, group in rows.groupby(keys, sort=True):
        records.append(
            {
                **dict(zip(keys, values)),
                "n_items": len(group),
                "accuracy": float(group["correct"].mean()),
                "macro_f1": _macro_f1(
                    group["gold_label"],
                    group["prediction_label"],
                    labels,
                ),
            }
        )
    return pd.DataFrame(records)


def _summarize_factorial(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if rows.empty:
        return {
            "factorial_pair_effects": pd.DataFrame(),
            "factorial_condition_summary": pd.DataFrame(),
            "factorial_factor_summary": pd.DataFrame(),
            "factorial_attribution_profile": pd.DataFrame(),
        }
    metrics = [
        "semantic_margin_gain",
        "extraction_identifier_margin_gain",
        "extraction_row_margin_gain",
    ]
    condition_keys = [
        "model_alias",
        "model_name",
        "mode",
        "template",
        "pair_type",
        "semantic_mapping",
        "identifier_set",
        "row_order",
    ]
    pair = (
        rows.groupby([*condition_keys, "condition", "pair_id"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in metrics})
    )
    condition = (
        pair.groupby(condition_keys, as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            **{metric: (metric, "mean") for metric in metrics},
        )
    )
    factor_rows: list[pd.DataFrame] = []
    base_keys = ["model_alias", "model_name", "mode", "template", "pair_type"]
    for factor in ("semantic_mapping", "identifier_set", "row_order"):
        frame = (
            pair.groupby([*base_keys, factor], as_index=False)
            .agg(
                n_pairs=("pair_id", "nunique"),
                **{metric: (metric, "mean") for metric in metrics},
            )
            .rename(columns={factor: "factor_level"})
        )
        frame.insert(len(base_keys), "factor", factor)
        factor_rows.append(frame)
    factor = pd.concat(factor_rows, ignore_index=True, sort=False)
    profile = (
        pair.groupby(["model_alias", "model_name", "mode"], as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            n_conditions=("condition", "nunique"),
            **{metric: (metric, "mean") for metric in metrics},
        )
    )
    return {
        "factorial_pair_effects": pair,
        "factorial_condition_summary": condition,
        "factorial_factor_summary": factor,
        "factorial_attribution_profile": profile,
    }


def aggregate_interface_factorial_outputs(
    config: InterfaceFactorialConfig,
    *,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    baseline_paths = sorted(
        config.audit.output_dir.glob("*/baseline_competence_rows.csv")
    )
    factorial_paths = sorted(
        config.audit.output_dir.glob("*/factorial_steering_rows.csv")
    )
    key_paths = sorted(
        config.audit.output_dir.glob("*/mapping_key_competence_rows.csv")
    )
    inventory_paths = sorted(
        config.audit.output_dir.glob("*/factorial_direction_inventory.csv")
    )
    baseline = (
        pd.concat([pd.read_csv(path) for path in baseline_paths], ignore_index=True)
        if baseline_paths
        else pd.DataFrame()
    )
    factorial = (
        pd.concat([pd.read_csv(path) for path in factorial_paths], ignore_index=True)
        if factorial_paths
        else pd.DataFrame()
    )
    key_rows = (
        pd.concat([pd.read_csv(path) for path in key_paths], ignore_index=True)
        if key_paths
        else pd.DataFrame()
    )
    inventory = (
        pd.concat([pd.read_csv(path) for path in inventory_paths], ignore_index=True)
        if inventory_paths
        else pd.DataFrame()
    )
    labels = list(config.audit.dataset.label_ranks)
    competence_summary = _summarize_competence(baseline, labels)
    primary_baseline = pd.DataFrame()
    if not competence_summary.empty:
        primary_baseline = competence_summary[
            competence_summary["identifier_set"].eq("letters_abc")
            & competence_summary["row_order"].eq("order_123")
        ].copy()
    key_summary = pd.DataFrame()
    if not key_rows.empty:
        key_summary = (
            key_rows.groupby(
                ["model_alias", "model_name", "semantic_mapping", "identifier_set"],
                as_index=False,
            )
            .agg(
                n_items=("correct", "size"),
                accuracy=("correct", "mean"),
                min_target_margin=("target_margin", "min"),
            )
        )
        key_summary["threshold"] = config.competence_threshold
        key_summary["competence_passed"] = (
            key_summary["accuracy"].ge(config.competence_threshold)
            & key_summary["min_target_margin"].gt(0.0)
        )
    tables = {
        "baseline_competence_rows": baseline,
        "baseline_competence_summary": competence_summary,
        "six_mapping_baseline_competence": primary_baseline,
        "mapping_key_competence_rows": key_rows,
        "mapping_key_competence_summary": key_summary,
        "factorial_steering_rows": factorial,
        "factorial_direction_inventory": inventory,
        "interface_factorial_errors": pd.DataFrame(errors or []),
    }
    tables.update(_summarize_factorial(factorial))
    condition_summary = tables["factorial_condition_summary"]
    association = pd.DataFrame()
    if not primary_baseline.empty and not condition_summary.empty:
        primary_effect = condition_summary[
            condition_summary["identifier_set"].eq("letters_abc")
            & condition_summary["row_order"].eq("order_123")
            & condition_summary["mode"].eq("raw_canonical_direction")
        ].copy()
        association = primary_effect.merge(
            primary_baseline,
            on=[
                "model_alias",
                "model_name",
                "template",
                "semantic_mapping",
                "identifier_set",
                "row_order",
            ],
            how="left",
            validate="many_to_one",
        )
    tables["competence_effect_association"] = association
    write_tables(tables, config.audit.output_dir)
    return tables


FACTORIAL_METRICS = (
    "semantic_margin_gain",
    "extraction_identifier_margin_gain",
    "extraction_row_margin_gain",
)


def _equal_contrast_bootstrap(
    frame: pd.DataFrame,
    metrics: tuple[str, ...],
    *,
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap pairs within contrast and average contrasts equally."""
    parts = [part.reset_index(drop=True) for _, part in frame.groupby("pair_type", sort=True)]
    if not parts:
        return {metric: (np.nan, np.nan, np.nan) for metric in metrics}
    boot = {metric: np.zeros(n_boot, dtype=float) for metric in metrics}
    point = {metric: 0.0 for metric in metrics}
    for part in parts:
        indices = rng.integers(0, len(part), size=(n_boot, len(part)))
        for metric in metrics:
            values = part[metric].astype(float).to_numpy()
            point[metric] += float(values.mean()) / len(parts)
            boot[metric] += values[indices].mean(axis=1) / len(parts)
    tail = (1.0 - confidence) / 2.0
    return {
        metric: (
            point[metric],
            float(np.quantile(boot[metric], tail)),
            float(np.quantile(boot[metric], 1.0 - tail)),
        )
        for metric in metrics
    }


def _pair_level_fixed_design_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Average the fixed 108-condition design before pair-level inference."""
    metrics = [*FACTORIAL_METRICS]
    return (
        frame.groupby(["model_alias", "pair_type", "pair_id"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in metrics})
        .assign(
            identifier_minus_semantics=lambda rows: (
                rows["extraction_identifier_margin_gain"]
                - rows["semantic_margin_gain"]
            ),
            identifier_minus_row=lambda rows: (
                rows["extraction_identifier_margin_gain"]
                - rows["extraction_row_margin_gain"]
            ),
        )
    )


def summarize_interface_factorial_statistics(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build CPU-only, competence-gated statistics for the factorial audit."""
    config = InterfaceFactorialConfig.from_json(
        config_path,
        project_root=project_root,
    )
    source = config.audit.output_dir
    output_dir = source / "statistics"
    pair_path = source / "factorial_pair_effects.csv"
    key_path = source / "mapping_key_competence_summary.csv"
    baseline_path = source / "six_mapping_baseline_competence.csv"
    condition_path = source / "factorial_condition_summary.csv"
    required = [pair_path, key_path, baseline_path, condition_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing factorial audit outputs: " + ", ".join(missing)
        )

    pair_columns = [
        "model_alias",
        "model_name",
        "pair_type",
        "pair_id",
        "semantic_mapping",
        "identifier_set",
        "row_order",
        "condition",
        *FACTORIAL_METRICS,
    ]
    pair = pd.read_csv(pair_path, usecols=pair_columns)
    key = pd.read_csv(key_path)
    baseline = pd.read_csv(baseline_path)
    condition = pd.read_csv(condition_path)
    gate = key[
        [
            "model_alias",
            "semantic_mapping",
            "identifier_set",
            "competence_passed",
        ]
    ].copy()
    pair = pair.merge(
        gate,
        on=["model_alias", "semantic_mapping", "identifier_set"],
        how="left",
        validate="many_to_one",
    )
    condition = condition.merge(
        gate,
        on=["model_alias", "semantic_mapping", "identifier_set"],
        how="left",
        validate="many_to_one",
    )
    if pair["competence_passed"].isna().any():
        raise ValueError("Some factorial cells have no mapping-competence record")

    n_boot = int(config.audit.statistics.n_boot)
    confidence = float(config.audit.statistics.confidence)
    n_permutations = int(config.audit.statistics.n_permutations)
    rng = np.random.default_rng(config.audit.runtime.seed)
    scopes = {
        "all_cells": pair,
        "competence_gated": pair[pair["competence_passed"].astype(bool)].copy(),
    }
    bootstrap_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    inferential_metrics = (
        *FACTORIAL_METRICS,
        "identifier_minus_semantics",
        "identifier_minus_row",
    )
    for scope_name, scope_frame in scopes.items():
        fixed = _pair_level_fixed_design_means(scope_frame)
        for model_alias, group in fixed.groupby("model_alias", sort=True):
            intervals = _equal_contrast_bootstrap(
                group,
                inferential_metrics,
                n_boot=n_boot,
                confidence=confidence,
                rng=rng,
            )
            for metric, (mean, low, high) in intervals.items():
                record = {
                    "scope": scope_name,
                    "model_alias": model_alias,
                    "metric": metric,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "n_contrasts": int(group["pair_type"].nunique()),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "confidence": confidence,
                    "estimand": "equal_contrast_mean_over_fixed_interface_design",
                }
                bootstrap_rows.append(record)
                if metric in {"identifier_minus_semantics", "identifier_minus_row"}:
                    values = group[metric].astype(float).to_numpy()
                    contrast_rows.append(
                        {
                            **record,
                            "p_sign_flip": paired_sign_flip_pvalue(
                                values,
                                n_permutations=n_permutations,
                                rng=rng,
                            ),
                        }
                    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    paired = pd.DataFrame(contrast_rows)
    if not paired.empty:
        paired["p_holm"] = np.nan
        for scope_name, indices in paired.groupby("scope").groups.items():
            paired.loc[indices, "p_holm"] = holm_adjust(
                paired.loc[indices, "p_sign_flip"].to_numpy()
            )

    key_summary = (
        key.groupby(["model_alias", "identifier_set"], as_index=False)
        .agg(
            n_mapping_cells=("competence_passed", "size"),
            n_passed=("competence_passed", "sum"),
            mean_key_accuracy=("accuracy", "mean"),
            min_key_accuracy=("accuracy", "min"),
            min_target_margin=("min_target_margin", "min"),
        )
    )
    key_summary["pass_rate"] = (
        key_summary["n_passed"] / key_summary["n_mapping_cells"]
    )

    baseline_summary = (
        baseline.groupby(["model_alias", "model_name"], as_index=False)
        .agg(
            n_mappings=("semantic_mapping", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            mean_macro_f1=("macro_f1", "mean"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
        )
    )

    non_source = condition[
        ~condition["semantic_mapping"].eq(
            config.audit.dataset.canonical_mapping
        )
    ].copy()
    consistency_rows: list[dict[str, Any]] = []
    for scope_name, scope_frame in {
        "all_cells": non_source,
        "competence_gated": non_source[
            non_source["competence_passed"].astype(bool)
        ],
    }.items():
        for model_alias, group in scope_frame.groupby("model_alias", sort=True):
            consistency_rows.append(
                {
                    "scope": scope_name,
                    "model_alias": model_alias,
                    "n_condition_contrast_cells": len(group),
                    "n_interface_cells": int(
                        group[
                            ["semantic_mapping", "identifier_set", "row_order"]
                        ].drop_duplicates().shape[0]
                    ),
                    "semantic_positive_rate": float(
                        (group["semantic_margin_gain"] > 0).mean()
                    ),
                    "identifier_positive_rate": float(
                        (group["extraction_identifier_margin_gain"] > 0).mean()
                    ),
                    "row_positive_rate": float(
                        (group["extraction_row_margin_gain"] > 0).mean()
                    ),
                    "identifier_gt_semantics_rate": float(
                        (
                            group["extraction_identifier_margin_gain"]
                            > group["semantic_margin_gain"]
                        ).mean()
                    ),
                    "identifier_gt_row_rate": float(
                        (
                            group["extraction_identifier_margin_gain"]
                            > group["extraction_row_margin_gain"]
                        ).mean()
                    ),
                    "mean_semantic_gain": float(
                        group["semantic_margin_gain"].mean()
                    ),
                    "mean_identifier_gain": float(
                        group["extraction_identifier_margin_gain"].mean()
                    ),
                    "mean_row_gain": float(
                        group["extraction_row_margin_gain"].mean()
                    ),
                }
            )
    consistency = pd.DataFrame(consistency_rows)

    gated_pair = pair[pair["competence_passed"].astype(bool)].copy()
    vocabulary_pair = (
        gated_pair.groupby(
            ["model_alias", "model_name", "identifier_set", "pair_type", "pair_id"],
            as_index=False,
        )
        .agg(**{metric: (metric, "mean") for metric in FACTORIAL_METRICS})
    )
    vocabulary_rows: list[dict[str, Any]] = []
    for (model_alias, model_name, identifier_set), group in vocabulary_pair.groupby(
        ["model_alias", "model_name", "identifier_set"],
        sort=True,
    ):
        intervals = _equal_contrast_bootstrap(
            group,
            FACTORIAL_METRICS,
            n_boot=n_boot,
            confidence=confidence,
            rng=rng,
        )
        for metric, (mean, low, high) in intervals.items():
            vocabulary_rows.append(
                {
                    "model_alias": model_alias,
                    "model_name": model_name,
                    "identifier_set": identifier_set,
                    "metric": metric,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "confidence": confidence,
                    "scope": "competence_gated",
                }
            )
    vocabulary = pd.DataFrame(vocabulary_rows)

    paper = bootstrap[
        bootstrap["scope"].eq("competence_gated")
    ].pivot(
        index="model_alias",
        columns="metric",
        values=["mean", "ci_low", "ci_high"],
    )
    paper.columns = [
        f"{metric}__{statistic}" for statistic, metric in paper.columns
    ]
    paper = paper.reset_index().merge(
        baseline_summary,
        on="model_alias",
        how="left",
        validate="one_to_one",
    )
    gate_counts = (
        key.groupby("model_alias", as_index=False)
        .agg(
            n_key_cells=("competence_passed", "size"),
            n_key_cells_passed=("competence_passed", "sum"),
        )
    )
    paper = paper.merge(
        gate_counts,
        on="model_alias",
        how="left",
        validate="one_to_one",
    )

    quality = pd.DataFrame(
        [
            {
                "status": "complete",
                "n_models": int(pair["model_alias"].nunique()),
                "n_pairs": int(pair["pair_id"].nunique()),
                "n_contrasts": int(pair["pair_type"].nunique()),
                "n_interface_conditions": int(pair["condition"].nunique()),
                "n_pair_condition_rows": len(pair),
                "n_key_cells": len(key),
                "n_key_cells_passed": int(key["competence_passed"].sum()),
                "key_cell_pass_rate": float(key["competence_passed"].mean()),
                "n_boot": n_boot,
                "n_permutations": n_permutations,
                "confidence": confidence,
                "primary_scope": "competence_gated",
            }
        ]
    )
    tables = {
        "factorial_statistics_manifest": quality,
        "factorial_pair_cluster_bootstrap_ci": bootstrap,
        "factorial_paired_contrast_tests": paired,
        "factorial_non_source_mapping_consistency": consistency,
        "factorial_identifier_vocabulary_bootstrap_ci": vocabulary,
        "factorial_six_mapping_baseline_competence": baseline,
        "factorial_baseline_competence_by_model": baseline_summary,
        "factorial_mapping_key_competence": key,
        "factorial_mapping_key_competence_by_model_identifier": key_summary,
        "factorial_paper_model_summary": paper,
    }
    write_tables(tables, output_dir)
    return tables


def run_interface_factorial_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    config = InterfaceFactorialConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    if phase not in {"run", "aggregate", "all"}:
        raise ValueError("phase must be run, aggregate, or all")
    errors: list[dict[str, Any]] = []
    if phase in {"run", "all"}:
        requested = set(
            model_aliases or [model.model.alias for model in config.audit.models]
        )
        base_items = load_base_items(
            config.audit.dataset,
            seed=config.audit.runtime.seed,
        )
        eval_items = _build_eval_items(base_items, _cross_interface_view(config))
        canonical = _canonical_mapping(config)
        canonical_items = build_mapping_items(
            base_items,
            config.audit.dataset,
            canonical,
            canonical,
        )
        conditions = _factorial_conditions(config)
        for model_config in config.audit.models:
            if model_config.model.alias not in requested:
                continue
            model_dir = config.audit.output_dir / model_config.model.alias
            complete_path = model_dir / "interface_factorial_run_complete.csv"
            if complete_path.exists() and not config.audit.runtime.force_rerun:
                continue
            tokenizer = model = None
            try:
                tokenizer, model = load_tokenizer_and_model(
                    model_config.model.load_source,
                    device_map=model_config.model.device_map,
                    torch_dtype=model_config.model.torch_dtype,
                )
                directions, inventory = build_canonical_direction_bank(
                    model,
                    tokenizer,
                    canonical_items,
                    config.audit.dataset,
                    layer_index=model_config.locked_layer,
                    subspace_dim=config.audit.subspace_dim,
                    batch_size=config.audit.runtime.batch_size,
                    max_length=config.audit.runtime.max_length,
                    seed=config.audit.runtime.seed,
                    extraction_position="pre_answer",
                    enabled_modes=("raw_canonical_direction",),
                )
                result_frames: list[pd.DataFrame] = []
                for condition_index, condition in enumerate(conditions, start=1):
                    print(
                        f"[{model_config.model.alias}] condition "
                        f"{condition_index}/{len(conditions)}: {condition.name}"
                    )
                    for template in config.templates:
                        condition_dir = (
                            model_dir / "conditions" / condition.name / template.name
                        )
                        complete = condition_dir / "condition_complete.csv"
                        baseline_path = condition_dir / "baseline_rows.csv"
                        steering_path = condition_dir / "steering_rows.csv"
                        if (
                            complete.exists()
                            and baseline_path.exists()
                            and steering_path.exists()
                            and not config.audit.runtime.force_rerun
                        ):
                            result_frames.extend(
                                [
                                    pd.read_csv(baseline_path),
                                    pd.read_csv(steering_path),
                                ]
                            )
                            continue
                        condition_rows = _evaluate_condition(
                            model,
                            tokenizer,
                            base_items,
                            eval_items,
                            directions,
                            config,
                            model_config,
                            template,
                            condition,
                        )
                        condition_baseline = condition_rows[
                            condition_rows["record_type"].eq("baseline")
                        ].dropna(axis=1, how="all")
                        condition_steering = condition_rows[
                            condition_rows["record_type"].eq("steering")
                        ].dropna(axis=1, how="all")
                        write_tables(
                            {
                                "baseline_rows": condition_baseline,
                                "steering_rows": condition_steering,
                                "condition_complete": pd.DataFrame(
                                    [
                                        {
                                            "condition": condition.name,
                                            "template": template.name,
                                            "status": "complete",
                                        }
                                    ]
                                ),
                            },
                            condition_dir,
                        )
                        result_frames.extend(
                            [condition_baseline, condition_steering]
                        )
                combined = pd.concat(result_frames, ignore_index=True, sort=False)
                baseline = combined[
                    combined["record_type"].eq("baseline")
                ].dropna(axis=1, how="all")
                steering = combined[
                    combined["record_type"].eq("steering")
                ].dropna(axis=1, how="all")
                key_rows = _evaluate_key_competence(model, tokenizer, config)
                for frame in (baseline, steering, key_rows, inventory):
                    frame.insert(0, "model_alias", model_config.model.alias)
                    frame.insert(1, "model_name", model_config.model.name)
                write_tables(
                    {
                        "baseline_competence_rows": baseline,
                        "factorial_steering_rows": steering,
                        "mapping_key_competence_rows": key_rows,
                        "factorial_direction_inventory": inventory,
                        "interface_factorial_run_complete": pd.DataFrame(
                            [
                                {
                                    "model_alias": model_config.model.alias,
                                    "status": "complete",
                                    "n_conditions": len(conditions),
                                    "locked_layer": model_config.locked_layer,
                                    "locked_alpha": model_config.locked_alpha,
                                }
                            ]
                        ),
                    },
                    model_dir,
                )
            except Exception as exc:
                errors.append(
                    {
                        "model_alias": model_config.model.alias,
                        "model_name": model_config.model.name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if not config.audit.runtime.continue_on_error:
                    raise
            finally:
                del model, tokenizer
                cleanup_model()
    return aggregate_interface_factorial_outputs(config, errors=errors)


def _cross_interface_view(config: InterfaceFactorialConfig) -> Any:
    """Minimal adapter required by the shared strict-pair item builder."""
    return SimpleNamespace(audit=config.audit)
