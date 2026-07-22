"""Cross-interface evaluation for frozen contrastive steering directions.

The audit deliberately keeps direction extraction fixed while changing only
the answer interface.  It reports sequence-likelihood margins rather than
approximating multi-token verbalizers with a single next-token probability.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .decomposition import contrast_directions
from .io import write_tables
from .mapping_audit import (
    FixedDirectionMappingAuditConfig,
    MappingDefinition,
    _load_mapping_audit_config,
    build_canonical_train_index,
    build_canonical_direction_bank,
    build_mapping_items,
    build_scenario_context,
    load_base_items,
)
from .steering import (
    cleanup_model,
    collect_choice_probs_and_activations,
    completion_sequence_logprobs,
    decoder_layers,
    load_tokenizer_and_model,
    resolve_layer_index,
)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    question: str
    answer_instruction: str


@dataclass(frozen=True)
class InterfaceDefinition:
    name: str
    kind: str
    label_values: dict[str, str]
    mapping_name: str | None = None
    key_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class CrossInterfaceConfig:
    audit: FixedDirectionMappingAuditConfig
    templates: tuple[PromptTemplate, ...]
    interfaces: tuple[InterfaceDefinition, ...]
    reference_interface: str
    primary_score: str
    minimum_reference_effect: float
    direction_sources: tuple[str, ...]
    mapping_balance_mappings: tuple[str, ...]
    random_control_seeds: tuple[int, ...]
    key_competence_threshold: float

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "CrossInterfaceConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        audit = FixedDirectionMappingAuditConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        spec = dict(data.get("cross_interface", {}))
        templates = tuple(
            PromptTemplate(
                name=str(item["name"]),
                question=str(item["question"]),
                answer_instruction=str(item["answer_instruction"]),
            )
            for item in spec.get("templates", [])
        )
        if not templates:
            raise ValueError("cross_interface.templates must contain at least one template")
        interfaces = tuple(
            InterfaceDefinition(
                name=str(item["name"]),
                kind=str(item["kind"]),
                label_values={str(key): str(value) for key, value in item["label_values"].items()},
                mapping_name=(str(item["mapping_name"]) if item.get("mapping_name") else None),
                key_lines=tuple(str(value) for value in item.get("key_lines", [])),
            )
            for item in spec.get("interfaces", [])
        )
        labels = set(audit.dataset.label_ranks)
        if not interfaces:
            raise ValueError("cross_interface.interfaces must contain at least one interface")
        for interface in interfaces:
            if interface.kind not in {"letter_mcq", "completion"}:
                raise ValueError(f"Unsupported interface kind: {interface.kind}")
            if set(interface.label_values) != labels:
                raise ValueError(f"Interface {interface.name!r} must map every semantic label")
        reference = str(spec.get("reference_interface", interfaces[0].name))
        if reference not in {item.name for item in interfaces}:
            raise ValueError("reference_interface must name a configured interface")
        score = str(spec.get("primary_score", "mean_logprob"))
        if score not in {"sum_logprob", "mean_logprob"}:
            raise ValueError("primary_score must be sum_logprob or mean_logprob")
        sources = tuple(str(value) for value in spec.get("direction_sources", ["raw_pre_answer", "raw_scenario_end", "random", "wrong"]))
        allowed = {
            "raw_pre_answer",
            "raw_scenario_end",
            "random",
            "wrong",
            "mapping_balanced",
            "mapping_sensitive_residual",
            "label_only",
            "slot_layout",
            "random_ensemble",
        }
        unknown = sorted(set(sources) - allowed)
        if unknown:
            raise ValueError(f"Unsupported direction_sources: {unknown}")
        mapping_names = {mapping.name for mapping in audit.mappings}
        default_balance = [mapping.name for mapping in audit.mappings if mapping.family == "semantic_words"]
        balance_mappings = tuple(str(value) for value in spec.get("mapping_balance_mappings", default_balance))
        if any(source in {"mapping_balanced", "mapping_sensitive_residual"} for source in sources):
            if len(balance_mappings) < 2:
                raise ValueError("Mapping-balanced directions require at least two extraction mappings")
            if audit.dataset.canonical_mapping not in balance_mappings:
                raise ValueError("mapping_balance_mappings must include the canonical mapping")
            unknown_mappings = sorted(set(balance_mappings) - mapping_names)
            if unknown_mappings:
                raise ValueError(f"Unknown mapping_balance_mappings: {unknown_mappings}")
        random_control_seeds = tuple(
            dict.fromkeys(int(value) for value in spec.get("random_control_seeds", []))
        )
        if "random_ensemble" in sources and len(random_control_seeds) < 2:
            raise ValueError(
                "random_ensemble requires at least two cross_interface.random_control_seeds"
            )
        key_competence_threshold = float(spec.get("key_competence_threshold", 0.9))
        if not 0.0 <= key_competence_threshold <= 1.0:
            raise ValueError("cross_interface.key_competence_threshold must be in [0, 1]")
        return cls(
            audit=audit,
            templates=templates,
            interfaces=interfaces,
            reference_interface=reference,
            primary_score=score,
            minimum_reference_effect=float(spec.get("minimum_reference_effect", 0.01)),
            direction_sources=sources,
            mapping_balance_mappings=balance_mappings,
            random_control_seeds=random_control_seeds,
            key_competence_threshold=key_competence_threshold,
        )


def _canonical_mapping(config: CrossInterfaceConfig) -> MappingDefinition:
    for mapping in config.audit.mappings:
        if mapping.name == config.audit.dataset.canonical_mapping:
            return mapping
    raise KeyError("Configured canonical mapping was not found")


def _interface_prefix(
    row: pd.Series,
    config: CrossInterfaceConfig,
    template: PromptTemplate,
    interface: InterfaceDefinition,
) -> str:
    dataset = config.audit.dataset
    lines = [build_scenario_context(row, dataset)]
    lines.extend(interface.key_lines)
    lines.append(template.question)
    if interface.kind == "letter_mcq":
        mapping_name = interface.mapping_name or dataset.canonical_mapping
        mapping = next(item for item in config.audit.mappings if item.name == mapping_name)
        for index, label in enumerate(mapping.option_order):
            lines.append(f"{chr(ord('A') + index)}. {interface.label_values[label]}")
    else:
        values = ", ".join(interface.label_values[label] for label in sorted(interface.label_values))
        lines.append(f"Valid completions: {values}.")
    lines.extend([template.answer_instruction, "Answer:"])
    return "\n".join(lines)


def _candidate_values(
    config: CrossInterfaceConfig,
    interface: InterfaceDefinition,
) -> dict[str, str]:
    if interface.kind != "letter_mcq":
        return interface.label_values
    mapping_name = interface.mapping_name or config.audit.dataset.canonical_mapping
    mapping = next(item for item in config.audit.mappings if item.name == mapping_name)
    return {label: chr(ord("A") + mapping.option_order.index(label)) for label in mapping.option_order}


def _original_slot_labels(
    config: CrossInterfaceConfig,
    interface: InterfaceDefinition,
    *,
    target_label: str,
    source_label: str,
) -> tuple[str | None, str | None]:
    """Return labels occupying the original target/source letter slots.

    The returned labels are interpreted under the current mapping.  Their
    scores therefore track the original answer tokens while the prompt itself
    uses the remapped semantic assignment.
    """
    if interface.kind != "letter_mcq":
        return None, None
    canonical = _canonical_mapping(config)
    mapping_name = interface.mapping_name or config.audit.dataset.canonical_mapping
    current = next(item for item in config.audit.mappings if item.name == mapping_name)
    target_slot = canonical.option_order.index(target_label)
    source_slot = canonical.option_order.index(source_label)
    return current.option_order[target_slot], current.option_order[source_slot]


def _build_eval_items(base_items: pd.DataFrame, config: CrossInterfaceConfig) -> pd.DataFrame:
    dataset = config.audit.dataset
    rows = []
    test = base_items[base_items[dataset.split_column].astype(str).eq(dataset.test_split)].copy()
    for (pair_type, pair_id), group in test.groupby([dataset.contrast_column, dataset.pair_id_column], sort=True):
        ordered = group.sort_values("semantic_rank")
        low, high = ordered.iloc[0], ordered.iloc[-1]
        for endpoint_name, endpoint, target, sign in [
            ("low", low, high, 1.0),
            ("high", high, low, -1.0),
        ]:
            rows.append(
                {
                    "pair_id": str(pair_id),
                    "pair_type": str(pair_type),
                    "endpoint": endpoint_name,
                    "direction_sign": sign,
                    "base_label": str(endpoint[dataset.label_column]),
                    "target_label": str(target[dataset.label_column]),
                    "low_label": str(low[dataset.label_column]),
                    "high_label": str(high[dataset.label_column]),
                    "row_index": int(endpoint.name),
                }
            )
    return pd.DataFrame(rows)


def _unit_scaled(vector: np.ndarray, target_norm: float) -> np.ndarray:
    """Match a nuisance direction's norm to its raw CAA counterpart."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8 or target_norm <= 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm * float(target_norm)).astype(np.float32)


def _pair_labels(base_items: pd.DataFrame, config: CrossInterfaceConfig) -> dict[str, tuple[str, str]]:
    dataset = config.audit.dataset
    output: dict[str, tuple[str, str]] = {}
    for pair_type, group in base_items.groupby(dataset.contrast_column, sort=True):
        labels = group[[dataset.label_column, "semantic_rank"]].drop_duplicates().sort_values("semantic_rank")
        if len(labels) != 2:
            raise ValueError(f"Expected exactly two labels for contrast {pair_type!r}")
        output[str(pair_type)] = (
            str(labels[dataset.label_column].iloc[0]),
            str(labels[dataset.label_column].iloc[-1]),
        )
    return output


def _layout_only_prompt(dataset: Any, mapping: MappingDefinition) -> str:
    """A fixed output-layout prompt with no scenario-specific content."""
    lines = [
        "Interface-only classification task. No behavior scenario is provided.",
        dataset.question,
    ]
    for index, label in enumerate(mapping.option_order):
        lines.append(f"{chr(ord('A') + index)}. {mapping.option_texts[label]}")
    lines.extend([dataset.answer_instruction, "Answer:"])
    return "\n".join(lines)


def _label_only_prompt(label: str) -> str:
    """A label lexical baseline intentionally stripped of scenario and options."""
    return f"Reference classification label: {label}\nAnswer:"


def _single_layer_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    layer_index: int,
    batch_size: int,
    max_length: int,
    n_labels: int,
) -> tuple[int, np.ndarray]:
    resolved = resolve_layer_index(layer_index, len(decoder_layers(model)))
    _, activations = collect_choice_probs_and_activations(
        model,
        tokenizer,
        prompts,
        layer_indices=[resolved],
        batch_size=batch_size,
        max_length=max_length,
        choice_letters=[chr(ord("A") + index) for index in range(n_labels)],
    )
    return resolved, activations[resolved]


def _mapping_norm_diagnostics(
    vectors: list[np.ndarray],
    mean_vector: np.ndarray,
    target_norm: float,
) -> dict[str, float]:
    component_norms = np.asarray(
        [float(np.linalg.norm(vector)) for vector in vectors],
        dtype=np.float64,
    )
    mean_component_norm = float(component_norms.mean())
    mean_vector_norm = float(np.linalg.norm(mean_vector))
    return {
        "mean_mapping_direction_l2": mean_component_norm,
        "min_mapping_direction_l2": float(component_norms.min()),
        "max_mapping_direction_l2": float(component_norms.max()),
        "mapping_mean_norm_retention_ratio": (
            mean_vector_norm / mean_component_norm
            if mean_component_norm > 0.0
            else np.nan
        ),
        "mapping_balance_rescale_factor": (
            target_norm / mean_vector_norm
            if mean_vector_norm > 0.0
            else np.nan
        ),
    }


def _mapping_balanced_directions(
    model: Any,
    tokenizer: Any,
    base_items: pd.DataFrame,
    config: CrossInterfaceConfig,
    model_cfg: Any,
    canonical_raw: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    """Average CAA directions extracted under counterbalanced answer mappings."""
    dataset = config.audit.dataset
    canonical = _canonical_mapping(config)
    mapping_by_name = {mapping.name: mapping for mapping in config.audit.mappings}
    raw_by_mapping: dict[str, dict[str, np.ndarray]] = {canonical.name: canonical_raw}
    resolved_layer = resolve_layer_index(model_cfg.locked_layer, len(decoder_layers(model)))
    for mapping_name in config.mapping_balance_mappings:
        if mapping_name == canonical.name:
            continue
        mapping = mapping_by_name[mapping_name]
        items = build_mapping_items(base_items, dataset, mapping, canonical)
        train_items, train_pairs = build_canonical_train_index(items, dataset)
        _, activations = collect_choice_probs_and_activations(
            model,
            tokenizer,
            train_items["prompt"].tolist(),
            layer_indices=[resolved_layer],
            batch_size=config.audit.runtime.batch_size,
            max_length=config.audit.runtime.max_length,
            choice_letters=[chr(ord("A") + index) for index in range(len(dataset.label_ranks))],
        )
        directions, _ = contrast_directions(train_pairs, activations[resolved_layer], train_split=dataset.train_split)
        raw_by_mapping[mapping_name] = directions

    balanced: dict[str, np.ndarray] = {}
    residual: dict[str, np.ndarray] = {}
    rows = []
    for pair_type, raw_vector in canonical_raw.items():
        vectors = [raw_by_mapping[name][pair_type] for name in config.mapping_balance_mappings]
        mean_vector = np.mean(np.stack(vectors), axis=0).astype(np.float32)
        target_norm = float(np.linalg.norm(raw_vector))
        norm_diagnostics = _mapping_norm_diagnostics(vectors, mean_vector, target_norm)
        balanced[pair_type] = _unit_scaled(mean_vector, target_norm)
        residual[pair_type] = _unit_scaled(raw_vector - mean_vector, target_norm)
        for mode, vector, unscaled in [
            ("mapping_balanced_direction", balanced[pair_type], mean_vector),
            ("mapping_sensitive_residual", residual[pair_type], raw_vector - mean_vector),
        ]:
            rows.append(
                {
                    "pair_type": pair_type,
                    "mode": mode,
                    "source_protocol": "counterbalanced_answer_mapping",
                    "extraction_position": "pre_answer",
                    "layer_index": resolved_layer,
                    "mapping_balance_mappings": ",".join(config.mapping_balance_mappings),
                    "n_mapping_extractions": len(config.mapping_balance_mappings),
                    "direction_l2": float(np.linalg.norm(vector)),
                    "unscaled_direction_l2": float(np.linalg.norm(unscaled)),
                    **norm_diagnostics,
                    "matched_raw_l2": target_norm,
                }
            )
    return balanced, residual, pd.DataFrame(rows)


def _label_and_layout_directions(
    model: Any,
    tokenizer: Any,
    base_items: pd.DataFrame,
    config: CrossInterfaceConfig,
    model_cfg: Any,
    canonical_raw: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    """Construct lexical-label and answer-layout nuisance baselines.

    ``label_only`` varies the label word with no scenario or options.
    ``slot_layout`` moves the same higher label between answer slots in an
    otherwise fixed, scenario-free interface prompt.
    """
    dataset = config.audit.dataset
    canonical = _canonical_mapping(config)
    labels = _pair_labels(base_items, config)
    label_prompts, layout_prompts, order = [], [], []
    for pair_type, (low_label, high_label) in labels.items():
        label_prompts.extend([_label_only_prompt(low_label), _label_only_prompt(high_label)])
        swapped = list(canonical.option_order)
        low_index, high_index = swapped.index(low_label), swapped.index(high_label)
        swapped[low_index], swapped[high_index] = swapped[high_index], swapped[low_index]
        swapped_mapping = MappingDefinition(
            name=f"layout_swap_{pair_type}",
            option_order=tuple(swapped),
            option_texts=canonical.option_texts,
            header_lines=canonical.header_lines,
            family=canonical.family,
        )
        layout_prompts.extend([_layout_only_prompt(dataset, canonical), _layout_only_prompt(dataset, swapped_mapping)])
        order.append(pair_type)
    resolved, label_acts = _single_layer_activations(
        model, tokenizer, label_prompts, layer_index=model_cfg.locked_layer,
        batch_size=config.audit.runtime.batch_size, max_length=config.audit.runtime.max_length,
        n_labels=len(dataset.label_ranks),
    )
    _, layout_acts = _single_layer_activations(
        model, tokenizer, layout_prompts, layer_index=model_cfg.locked_layer,
        batch_size=config.audit.runtime.batch_size, max_length=config.audit.runtime.max_length,
        n_labels=len(dataset.label_ranks),
    )
    label_only: dict[str, np.ndarray] = {}
    slot_layout: dict[str, np.ndarray] = {}
    rows = []
    for index, pair_type in enumerate(order):
        target_norm = float(np.linalg.norm(canonical_raw[pair_type]))
        label_vector = label_acts[2 * index + 1] - label_acts[2 * index]
        layout_vector = layout_acts[2 * index] - layout_acts[2 * index + 1]
        label_only[pair_type] = _unit_scaled(label_vector, target_norm)
        slot_layout[pair_type] = _unit_scaled(layout_vector, target_norm)
        for mode, vector, unscaled in [
            ("label_only_direction", label_only[pair_type], label_vector),
            ("slot_layout_direction", slot_layout[pair_type], layout_vector),
        ]:
            rows.append(
                {
                    "pair_type": pair_type,
                    "mode": mode,
                    "source_protocol": "scenario_free_nuisance_baseline",
                    "extraction_position": "pre_answer",
                    "layer_index": resolved,
                    "direction_l2": float(np.linalg.norm(vector)),
                    "unscaled_direction_l2": float(np.linalg.norm(unscaled)),
                    "matched_raw_l2": target_norm,
                }
            )
    return label_only, slot_layout, pd.DataFrame(rows)


def _random_control_ensemble(
    canonical_raw: dict[str, np.ndarray],
    *,
    seeds: tuple[int, ...],
    layer_index: int,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    """Build reproducible norm-matched Gaussian controls without re-extracting CAA."""
    directions: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for pair_type, raw_vector in sorted(canonical_raw.items()):
            raw = np.asarray(raw_vector, dtype=np.float32)
            raw_norm = float(np.linalg.norm(raw))
            sampled = rng.normal(size=raw.shape).astype(np.float32)
            vector = _unit_scaled(sampled, raw_norm)
            mode = f"random_direction_control_seed_{int(seed)}"
            directions[(pair_type, mode)] = vector
            rows.append(
                {
                    "pair_type": pair_type,
                    "mode": mode,
                    "source_protocol": "seeded_norm_matched_gaussian",
                    "random_seed": int(seed),
                    "extraction_position": "none",
                    "layer_index": int(layer_index),
                    "direction_l2": float(np.linalg.norm(vector)),
                    "matched_raw_l2": raw_norm,
                    "norm_match_abs_error": abs(float(np.linalg.norm(vector)) - raw_norm),
                }
            )
    return directions, pd.DataFrame(rows)


def _build_key_competence_items(config: CrossInterfaceConfig) -> pd.DataFrame:
    """Build scenario-free completion checks for every temporary opaque key."""
    labels = list(config.audit.dataset.label_ranks)
    question_templates = (
        "Which completion means {label}?",
        "Under the temporary key, write the codeword for {label}.",
        "If the classification is {label}, which completion should be returned?",
    )
    rows: list[dict[str, Any]] = []
    for interface in config.interfaces:
        if interface.kind != "completion" or not interface.key_lines:
            continue
        values = [interface.label_values[label] for label in labels]
        for target_label in labels:
            for template_index, question_template in enumerate(question_templates):
                lines = list(interface.key_lines)
                lines.append(question_template.format(label=target_label.replace("_", " ")))
                lines.append(f"Valid completions: {', '.join(values)}.")
                lines.extend(["Return exactly one completion.", "Answer:"])
                rows.append(
                    {
                        "interface": interface.name,
                        "interface_kind": interface.kind,
                        "template": f"key_check_{template_index + 1}",
                        "target_label": target_label,
                        "target_completion": interface.label_values[target_label],
                        "prompt": "\n".join(lines),
                    }
                )
    return pd.DataFrame(rows)


def _evaluate_key_competence(
    model: Any,
    tokenizer: Any,
    config: CrossInterfaceConfig,
) -> pd.DataFrame:
    items = _build_key_competence_items(config)
    if items.empty:
        return pd.DataFrame(
            columns=[
                "interface", "template", "target_label", "target_completion",
                "prediction_label", "correct", "target_mean_logprob",
                "best_other_mean_logprob", "target_margin",
            ]
        )
    labels = list(config.audit.dataset.label_ranks)
    interface_by_name = {interface.name: interface for interface in config.interfaces}
    result_rows: list[dict[str, Any]] = []
    for interface_name, group in items.groupby("interface", sort=True):
        interface = interface_by_name[str(interface_name)]
        candidates = [" " + interface.label_values[label] for label in labels]
        completions = [candidates for _ in range(len(group))]
        _, mean_scores, token_counts = completion_sequence_logprobs(
            model,
            tokenizer,
            group["prompt"].tolist(),
            completions,
            batch_size=config.audit.runtime.batch_size,
            max_length=config.audit.runtime.max_length,
        )
        for row_index, (_, item) in enumerate(group.reset_index(drop=True).iterrows()):
            target_index = labels.index(str(item["target_label"]))
            prediction_index = int(np.argmax(mean_scores[row_index]))
            other_scores = np.delete(mean_scores[row_index], target_index)
            result_rows.append(
                {
                    **item.to_dict(),
                    "prediction_label": labels[prediction_index],
                    "correct": prediction_index == target_index,
                    "target_mean_logprob": float(mean_scores[row_index, target_index]),
                    "best_other_mean_logprob": float(np.max(other_scores)),
                    "target_margin": float(
                        mean_scores[row_index, target_index] - np.max(other_scores)
                    ),
                    "target_n_tokens": int(token_counts[row_index, target_index]),
                }
            )
    return pd.DataFrame(result_rows)


def _direction_sources(model: Any, tokenizer: Any, base_items: pd.DataFrame, config: CrossInterfaceConfig, model_cfg: Any) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    canonical = _canonical_mapping(config)
    canonical_items = build_mapping_items(base_items, config.audit.dataset, canonical, canonical)
    requested = ("raw_canonical_direction", "random_direction_control", "wrong_direction_control")
    pre_bank, pre_inventory = build_canonical_direction_bank(
        model, tokenizer, canonical_items, config.audit.dataset,
        layer_index=model_cfg.locked_layer, subspace_dim=config.audit.subspace_dim,
        batch_size=config.audit.runtime.batch_size, max_length=config.audit.runtime.max_length,
        seed=config.audit.runtime.seed, extraction_position="pre_answer", enabled_modes=requested,
    )
    scenario_bank: dict[tuple[str, str], np.ndarray] = {}
    scenario_inventory = pd.DataFrame()
    if "raw_scenario_end" in config.direction_sources:
        scenario_bank, scenario_inventory = build_canonical_direction_bank(
            model, tokenizer, canonical_items, config.audit.dataset,
            layer_index=model_cfg.locked_layer, subspace_dim=config.audit.subspace_dim,
            batch_size=config.audit.runtime.batch_size, max_length=config.audit.runtime.max_length,
            seed=config.audit.runtime.seed, extraction_position="scenario_end", enabled_modes=("raw_canonical_direction",),
        )
    rename = {
        "raw_pre_answer": (pre_bank, "raw_canonical_direction", "raw_pre_answer"),
        "raw_scenario_end": (scenario_bank, "raw_canonical_direction", "raw_scenario_end"),
        "random": (pre_bank, "random_direction_control", "random_direction_control"),
        "wrong": (pre_bank, "wrong_direction_control", "wrong_direction_control"),
    }
    directions: dict[tuple[str, str], np.ndarray] = {}
    inventory = []
    inventories = [(pre_inventory, "pre_answer")]
    if not scenario_inventory.empty:
        inventories.append((scenario_inventory, "scenario_end"))
    for source in config.direction_sources:
        if source not in rename:
            continue
        bank, bank_mode, out_mode = rename[source]
        for (pair_type, mode), vector in bank.items():
            if mode != bank_mode:
                continue
            directions[(pair_type, out_mode)] = vector
            source_inventory = next(frame for frame, position in inventories if position == ("scenario_end" if source == "raw_scenario_end" else "pre_answer"))
            meta = source_inventory[(source_inventory["pair_type"].astype(str) == pair_type) & (source_inventory["mode"].astype(str) == bank_mode)].iloc[0].to_dict()
            meta["mode"] = out_mode
            meta["source_protocol"] = source
            inventory.append(meta)

    canonical_raw = {
        pair_type: vector
        for (pair_type, mode), vector in pre_bank.items()
        if mode == "raw_canonical_direction"
    }
    requested_sources = set(config.direction_sources)
    if "random_ensemble" in requested_sources:
        random_directions, random_inventory = _random_control_ensemble(
            canonical_raw,
            seeds=config.random_control_seeds,
            layer_index=resolve_layer_index(model_cfg.locked_layer, len(decoder_layers(model))),
        )
        directions.update(random_directions)
        inventory.extend(random_inventory.to_dict("records"))
    if requested_sources & {"mapping_balanced", "mapping_sensitive_residual"}:
        balanced, residual, nuisance_inventory = _mapping_balanced_directions(
            model, tokenizer, base_items, config, model_cfg, canonical_raw
        )
        if "mapping_balanced" in requested_sources:
            directions.update(
                {(pair_type, "mapping_balanced_direction"): vector for pair_type, vector in balanced.items()}
            )
        if "mapping_sensitive_residual" in requested_sources:
            directions.update(
                {(pair_type, "mapping_sensitive_residual"): vector for pair_type, vector in residual.items()}
            )
        inventory.extend(nuisance_inventory.to_dict("records"))
    if requested_sources & {"label_only", "slot_layout"}:
        label_only, slot_layout, nuisance_inventory = _label_and_layout_directions(
            model, tokenizer, base_items, config, model_cfg, canonical_raw
        )
        if "label_only" in requested_sources:
            directions.update(
                {(pair_type, "label_only_direction"): vector for pair_type, vector in label_only.items()}
            )
        if "slot_layout" in requested_sources:
            directions.update(
                {(pair_type, "slot_layout_direction"): vector for pair_type, vector in slot_layout.items()}
            )
        inventory.extend(nuisance_inventory.to_dict("records"))
    return directions, pd.DataFrame(inventory)


def _evaluate_interface(model: Any, tokenizer: Any, base_items: pd.DataFrame, eval_items: pd.DataFrame, directions: dict[tuple[str, str], np.ndarray], config: CrossInterfaceConfig, model_cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = config.audit.dataset
    rows, tokens = [], []
    for interface in config.interfaces:
        for template in config.templates:
            prefixes = [_interface_prefix(base_items.loc[item.row_index], config, template, interface) for item in eval_items.itertuples()]
            labels = list(dataset.label_ranks)
            candidate_values = _candidate_values(config, interface)
            completions = [[" " + candidate_values[label] for label in labels] for _ in prefixes]
            base_sum, base_mean, counts = completion_sequence_logprobs(model, tokenizer, prefixes, completions, batch_size=config.audit.runtime.batch_size, max_length=config.audit.runtime.max_length)
            for label_index, label in enumerate(labels):
                tokens.extend({"interface": interface.name, "template": template.name, "label": label, "completion": candidate_values[label], "n_tokens": int(count)} for count in np.unique(counts[:, label_index]))
            for (pair_type, mode), direction in directions.items():
                pair_mask = eval_items["pair_type"].astype(str).eq(pair_type).to_numpy()
                if not pair_mask.any():
                    continue
                for policy in ("counterfactual", "context_plus"):
                    sign_groups = [1.0, -1.0] if policy == "counterfactual" else [1.0]
                    for sign in sign_groups:
                        mask = pair_mask & (
                            eval_items["direction_sign"].eq(sign).to_numpy()
                            if policy == "counterfactual"
                            else np.ones(len(eval_items), dtype=bool)
                        )
                        if not mask.any():
                            continue
                        indices = np.flatnonzero(mask)
                        subset = eval_items.loc[mask].reset_index(drop=True)
                        patched_sum, patched_mean, _ = completion_sequence_logprobs(
                            model, tokenizer, [prefixes[index] for index in indices], [completions[index] for index in indices],
                            batch_size=config.audit.runtime.batch_size, max_length=config.audit.runtime.max_length,
                            layer_index=model_cfg.locked_layer, direction=np.asarray(direction) * sign, alpha=model_cfg.locked_alpha,
                        )
                        base_scores = base_mean[mask] if config.primary_score == "mean_logprob" else base_sum[mask]
                        patched_scores = patched_mean if config.primary_score == "mean_logprob" else patched_sum
                        base_sum_scores, base_mean_scores = base_sum[mask], base_mean[mask]
                        for index, item in subset.iterrows():
                            target = labels.index(item.target_label)
                            source = labels.index(item.base_label)
                            high = labels.index(item.high_label)
                            low = labels.index(item.low_label)
                            target_gain = (patched_scores[index, target] - patched_scores[index, source]) - (base_scores[index, target] - base_scores[index, source])
                            high_low_delta = (patched_scores[index, high] - patched_scores[index, low]) - (base_scores[index, high] - base_scores[index, low])
                            target_gain_sum = (patched_sum[index, target] - patched_sum[index, source]) - (base_sum_scores[index, target] - base_sum_scores[index, source])
                            target_gain_mean = (patched_mean[index, target] - patched_mean[index, source]) - (base_mean_scores[index, target] - base_mean_scores[index, source])
                            original_target_label, original_source_label = _original_slot_labels(
                                config,
                                interface,
                                target_label=str(item.target_label),
                                source_label=str(item.base_label),
                            )
                            if original_target_label is not None and original_source_label is not None:
                                original_target = labels.index(original_target_label)
                                original_source = labels.index(original_source_label)
                                original_slot_margin_gain = (
                                    patched_scores[index, original_target]
                                    - patched_scores[index, original_source]
                                    - base_scores[index, original_target]
                                    + base_scores[index, original_source]
                                )
                            else:
                                original_slot_margin_gain = np.nan
                            rows.append({
                                "pair_id": item.pair_id, "pair_type": item.pair_type, "endpoint": item.endpoint,
                                "direction_sign": sign, "evaluation_policy": policy, "mode": mode, "interface": interface.name,
                                "interface_kind": interface.kind, "template": template.name,
                                "base_label": item.base_label, "target_label": item.target_label,
                                "target_margin_gain": float(target_gain), "target_margin_gain_sum": float(target_gain_sum),
                                "target_margin_gain_mean": float(target_gain_mean), "high_low_margin_delta": float(high_low_delta),
                                "original_target_slot_label": original_target_label,
                                "original_source_slot_label": original_source_label,
                                "original_slot_margin_gain": float(original_slot_margin_gain),
                                "base_high_low_margin": float(base_scores[index, high] - base_scores[index, low]),
                                "patched_high_low_margin": float(patched_scores[index, high] - patched_scores[index, low]),
                                "score_normalization": config.primary_score,
                            })
    return pd.DataFrame(rows), pd.DataFrame(tokens).drop_duplicates()


def _summaries(rows: pd.DataFrame, config: CrossInterfaceConfig) -> dict[str, pd.DataFrame]:
    if rows.empty:
        return {"cross_interface_pair_effects": rows, "cross_interface_summary": pd.DataFrame(), "context_interactions": pd.DataFrame(), "interface_retention": pd.DataFrame()}
    counter = rows[rows["evaluation_policy"].eq("counterfactual")].copy()
    group_prefix = ["model_alias", "model_name"] if "model_alias" in rows.columns else []
    pair = counter.groupby([*group_prefix, "pair_id", "pair_type", "mode", "interface", "interface_kind", "template"], as_index=False).agg(target_margin_gain=("target_margin_gain", "mean"))
    summary = pair.groupby([*group_prefix, "mode", "interface", "interface_kind", "template", "pair_type"], as_index=False).agg(n_pairs=("pair_id", "nunique"), target_margin_gain=("target_margin_gain", "mean"))
    probe = rows[rows["evaluation_policy"].eq("context_plus")].copy()
    interaction = probe.pivot_table(index=[*group_prefix, "pair_id", "pair_type", "mode", "interface", "template"], columns="endpoint", values="high_low_margin_delta", aggfunc="mean").reset_index()
    if {"low", "high"}.issubset(interaction.columns):
        interaction["context_interaction_low_minus_high"] = interaction["low"] - interaction["high"]
    context = interaction.groupby([*group_prefix, "mode", "interface", "template", "pair_type"], as_index=False).agg(n_pairs=("pair_id", "nunique"), context_interaction=("context_interaction_low_minus_high", "mean")) if "context_interaction_low_minus_high" in interaction else pd.DataFrame()
    reference = pair[pair["interface"].eq(config.reference_interface)].rename(columns={"target_margin_gain": "reference_gain"})
    merged = pair.merge(reference[[*group_prefix, "pair_id", "pair_type", "mode", "template", "reference_gain"]], on=[*group_prefix, "pair_id", "pair_type", "mode", "template"], how="left")
    merged["retention_ratio"] = np.where(merged["reference_gain"].abs().ge(config.minimum_reference_effect), merged["target_margin_gain"] / merged["reference_gain"], np.nan)
    merged["sign_consistent"] = np.where(merged["reference_gain"].abs().ge(config.minimum_reference_effect), np.sign(merged["target_margin_gain"]) == np.sign(merged["reference_gain"]), np.nan)
    return {"cross_interface_pair_effects": pair, "cross_interface_summary": summary, "context_interactions": context, "interface_retention": merged}


def aggregate_cross_interface_outputs(
    config: CrossInterfaceConfig,
    *,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Aggregate completed model directories without loading any model."""
    all_rows = [
        pd.read_csv(path)
        for path in config.audit.output_dir.glob("*/cross_interface_eval_rows.csv")
    ]
    all_tokens = [
        pd.read_csv(path)
        for path in config.audit.output_dir.glob("*/cross_interface_tokenization.csv")
    ]
    all_inventory = [
        pd.read_csv(path)
        for path in config.audit.output_dir.glob("*/cross_interface_direction_inventory.csv")
    ]
    all_competence = [
        pd.read_csv(path)
        for path in config.audit.output_dir.glob("*/cross_interface_key_competence.csv")
    ]
    rows = pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    tables = _summaries(rows, config)
    competence = (
        pd.concat(all_competence, ignore_index=True, sort=False)
        if all_competence
        else pd.DataFrame()
    )
    competence_summary = pd.DataFrame()
    if not competence.empty:
        competence_summary = (
            competence.groupby(["model_alias", "model_name", "interface"], as_index=False)
            .agg(
                n_items=("correct", "size"),
                accuracy=("correct", "mean"),
                mean_target_margin=("target_margin", "mean"),
                min_target_margin=("target_margin", "min"),
                n_target_labels=("target_label", "nunique"),
            )
        )
        competence_summary["threshold"] = config.key_competence_threshold
        competence_summary["competence_passed"] = (
            competence_summary["accuracy"].ge(config.key_competence_threshold)
            & competence_summary["min_target_margin"].gt(0.0)
        )
    tables.update(
        {
            "cross_interface_eval_rows": rows,
            "cross_interface_tokenization": (
                pd.concat(all_tokens, ignore_index=True, sort=False)
                if all_tokens
                else pd.DataFrame()
            ),
            "cross_interface_direction_inventory": (
                pd.concat(all_inventory, ignore_index=True, sort=False)
                if all_inventory
                else pd.DataFrame()
            ),
            "cross_interface_key_competence": competence,
            "cross_interface_key_competence_summary": competence_summary,
            "cross_interface_errors": pd.DataFrame(errors or []),
            "cross_interface_config": pd.DataFrame(
                [
                    {
                        "reference_interface": config.reference_interface,
                        "primary_score": config.primary_score,
                        "minimum_reference_effect": config.minimum_reference_effect,
                        "direction_sources": ",".join(config.direction_sources),
                        "random_control_seeds": ",".join(
                            str(value) for value in config.random_control_seeds
                        ),
                        "key_competence_threshold": config.key_competence_threshold,
                    }
                ]
            ),
        }
    )
    write_tables(tables, config.audit.output_dir)
    return tables


def run_cross_interface_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    inventory_only: bool = False,
) -> dict[str, pd.DataFrame]:
    config = CrossInterfaceConfig.from_json(config_path, project_root=project_root, model_source_overrides=model_source_overrides)
    requested = set(model_aliases or [item.model.alias for item in config.audit.models])
    base_items = load_base_items(config.audit.dataset, seed=config.audit.runtime.seed)
    eval_items = _build_eval_items(base_items, config)
    errors = []
    for model_cfg in config.audit.models:
        if model_cfg.model.alias not in requested:
            continue
        model_dir = config.audit.output_dir / model_cfg.model.alias
        if (
            (model_dir / "cross_interface_run_complete.csv").exists()
            and not config.audit.runtime.force_rerun
            and not inventory_only
        ):
            continue
        tokenizer = model = None
        try:
            tokenizer, model = load_tokenizer_and_model(model_cfg.model.load_source, device_map=model_cfg.model.device_map, torch_dtype=model_cfg.model.torch_dtype)
            directions, inventory = _direction_sources(model, tokenizer, base_items, config, model_cfg)
            if inventory_only:
                inventory.insert(0, "model_alias", model_cfg.model.alias)
                inventory.insert(1, "model_name", model_cfg.model.name)
                write_tables(
                    {
                        "cross_interface_direction_inventory": inventory,
                        "cross_interface_inventory_complete": pd.DataFrame(
                            [
                                {
                                    "model_alias": model_cfg.model.alias,
                                    "status": "complete",
                                    "locked_layer": model_cfg.locked_layer,
                                    "locked_alpha": model_cfg.locked_alpha,
                                }
                            ]
                        ),
                    },
                    model_dir,
                )
                continue
            rows, token_audit = _evaluate_interface(model, tokenizer, base_items, eval_items, directions, config, model_cfg)
            key_competence = _evaluate_key_competence(model, tokenizer, config)
            for frame in (rows, token_audit, inventory, key_competence):
                frame.insert(0, "model_alias", model_cfg.model.alias)
                frame.insert(1, "model_name", model_cfg.model.name)
            write_tables({"cross_interface_eval_rows": rows, "cross_interface_tokenization": token_audit, "cross_interface_direction_inventory": inventory, "cross_interface_key_competence": key_competence, "cross_interface_run_complete": pd.DataFrame([{"model_alias": model_cfg.model.alias, "status": "complete", "locked_layer": model_cfg.locked_layer, "locked_alpha": model_cfg.locked_alpha}])}, model_dir)
        except Exception as exc:
            errors.append({"model_alias": model_cfg.model.alias, "model_name": model_cfg.model.name, "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
        finally:
            del model, tokenizer
            cleanup_model()
    # Re-read every completed model directory. This keeps resumed subset runs
    # complete and lets the CPU synthesis repair a parallel aggregation race.
    return aggregate_cross_interface_outputs(config, errors=errors)
