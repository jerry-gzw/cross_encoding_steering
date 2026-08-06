"""Matched three-way MNLI control for cross-interface steering effects."""
from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ModelConfig
from .io import write_tables
from .steering import (
    cleanup_model,
    collect_choice_probs_and_activations,
    completion_sequence_logprobs,
    decoder_layers,
    load_tokenizer_and_model,
    resolve_layer_index,
)


MNLI_LABELS = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True)
class MnliContrast:
    name: str
    source_label: str
    target_label: str


@dataclass(frozen=True)
class MnliInterface:
    name: str
    kind: str
    label_values: dict[str, str]
    option_order: tuple[str, ...] = ()
    key_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class MnliTemplate:
    name: str
    question: str
    answer_instruction: str


@dataclass(frozen=True)
class MnliModelRun:
    model: ModelConfig
    locked_layer: int
    locked_alpha: float


@dataclass(frozen=True)
class MnliControlConfig:
    project_root: Path
    input_path: Path
    output_dir: Path
    contrasts: tuple[MnliContrast, ...]
    interfaces: tuple[MnliInterface, ...]
    templates: tuple[MnliTemplate, ...]
    models: tuple[MnliModelRun, ...]
    reference_interface: str
    primary_score: str
    max_train_pairs_per_contrast: int
    max_test_pairs_per_contrast: int
    batch_size: int
    max_length: int
    seed: int
    force_rerun: bool
    n_boot: int
    confidence: float
    bootstrap_chunk_size: int

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "MnliControlConfig":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        contrasts = tuple(MnliContrast(**item) for item in data.get("contrasts", []))
        if not contrasts:
            raise ValueError("MNLI control requires at least one label contrast")
        for contrast in contrasts:
            if contrast.source_label not in MNLI_LABELS or contrast.target_label not in MNLI_LABELS:
                raise ValueError(f"Unknown MNLI contrast labels in {contrast.name!r}")
            if contrast.source_label == contrast.target_label:
                raise ValueError(f"MNLI contrast {contrast.name!r} uses the same source and target label")

        interfaces = tuple(
            MnliInterface(
                name=str(item["name"]),
                kind=str(item["kind"]),
                label_values={str(key): str(value) for key, value in item["label_values"].items()},
                option_order=tuple(str(value) for value in item.get("option_order", [])),
                key_lines=tuple(str(value) for value in item.get("key_lines", [])),
            )
            for item in data.get("interfaces", [])
        )
        if not interfaces:
            raise ValueError("MNLI control requires at least one answer interface")
        for interface in interfaces:
            if interface.kind not in {"letter_mcq", "completion"}:
                raise ValueError(f"Unsupported MNLI interface kind: {interface.kind}")
            if set(interface.label_values) != set(MNLI_LABELS):
                raise ValueError(f"Interface {interface.name!r} must map all MNLI labels")
            if interface.kind == "letter_mcq" and set(interface.option_order) != set(MNLI_LABELS):
                raise ValueError(f"Letter interface {interface.name!r} requires a complete option_order")

        templates = tuple(MnliTemplate(**item) for item in data.get("templates", []))
        if not templates:
            raise ValueError("MNLI control requires at least one prompt template")
        overrides = model_source_overrides or {}
        models = []
        for item in data.get("models", []):
            model_data = dict(item)
            locked_layer = int(model_data.pop("locked_layer"))
            locked_alpha = float(model_data.pop("locked_alpha", 0.8))
            alias = str(model_data["alias"])
            if alias in overrides:
                model_data["source"] = overrides[alias]
            models.append(MnliModelRun(ModelConfig.from_mapping(model_data), locked_layer, locked_alpha))
        if not models:
            raise ValueError("MNLI control requires at least one model")

        reference = str(data.get("reference_interface", interfaces[0].name))
        if reference not in {item.name for item in interfaces}:
            raise ValueError("reference_interface must name a configured interface")
        primary_score = str(data.get("primary_score", "mean_logprob"))
        if primary_score not in {"sum_logprob", "mean_logprob"}:
            raise ValueError("primary_score must be sum_logprob or mean_logprob")
        runtime = dict(data.get("runtime", {}))
        statistics = dict(data.get("statistics", {}))
        n_boot = int(statistics.get("n_boot", 10_000))
        confidence = float(statistics.get("confidence", 0.95))
        if n_boot < 100:
            raise ValueError("statistics.n_boot must be at least 100")
        if not 0.0 < confidence < 1.0:
            raise ValueError("statistics.confidence must be between zero and one")
        chunk_size = int(statistics.get("bootstrap_chunk_size", 500))
        if chunk_size < 1:
            raise ValueError("statistics.bootstrap_chunk_size must be positive")
        return cls(
            project_root=root,
            input_path=resolve(str(data.get("input_path", "datasets/multinli_1.0_train.jsonl"))),
            output_dir=resolve(str(data.get("output_dir", "outputs/mnli_non_norm_control"))),
            contrasts=contrasts,
            interfaces=interfaces,
            templates=templates,
            models=tuple(models),
            reference_interface=reference,
            primary_score=primary_score,
            max_train_pairs_per_contrast=int(data.get("max_train_pairs_per_contrast", 2048)),
            max_test_pairs_per_contrast=int(data.get("max_test_pairs_per_contrast", 512)),
            batch_size=int(runtime.get("batch_size", 2)),
            max_length=int(runtime.get("max_length", 512)),
            seed=int(runtime.get("seed", 13)),
            force_rerun=bool(runtime.get("force_rerun", False)),
            n_boot=n_boot,
            confidence=confidence,
            bootstrap_chunk_size=chunk_size,
        )


def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_mnli_items(config: MnliControlConfig) -> pd.DataFrame:
    if not config.input_path.exists():
        raise FileNotFoundError(f"MNLI input file does not exist: {config.input_path}")
    rows = pd.read_json(config.input_path, lines=True)
    required = {"pairID", "promptID", "genre", "gold_label", "sentence1", "sentence2"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"MNLI input is missing columns: {missing}")
    rows = rows[rows["gold_label"].astype(str).isin(MNLI_LABELS)].copy()
    rows["premise"] = rows["sentence1"].fillna("").astype(str).str.strip()
    rows["hypothesis"] = rows["sentence2"].fillna("").astype(str).str.strip()
    rows["label"] = rows["gold_label"].astype(str)
    rows["group_id"] = rows["premise"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    )
    rows = rows.drop_duplicates(["group_id", "hypothesis", "label"]).reset_index(drop=True)
    rows["item_id"] = [
        f"{group_id}:{pair_id}:{index}"
        for index, (group_id, pair_id) in enumerate(zip(rows["group_id"], rows["pairID"].astype(str)))
    ]
    rows["text_length"] = rows["premise"].str.split().str.len() + rows["hypothesis"].str.split().str.len()
    # Split at the normalized-premise group level. MultiNLI normally provides one
    # entailment, neutral, and contradiction hypothesis for the same premise;
    # splitting individual pair IDs would leak that premise across train/test.
    buckets = rows["group_id"].map(lambda value: _stable_hash(value, config.seed) % 100)
    rows["split"] = np.where(buckets.lt(80), "train", np.where(buckets.lt(90), "validation", "test"))
    rows["tie_break"] = rows["item_id"].map(lambda value: _stable_hash(value, config.seed + 1))
    return rows[["group_id", "item_id", "genre", "premise", "hypothesis", "label", "text_length", "split", "tie_break"]].reset_index(drop=True)


def _matched_label_pairs(
    items: pd.DataFrame,
    contrast: MnliContrast,
    *,
    split: str,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    selected = items[items["split"].eq(split)].copy()

    def side(label: str, prefix: str) -> pd.DataFrame:
        frame = (
            selected[selected["label"].eq(label)]
            .sort_values(["group_id", "text_length", "tie_break", "item_id"])
            .drop_duplicates("group_id")
            .copy()
        )
        return frame.rename(
            columns={column: f"{prefix}_{column}" for column in frame.columns if column != "group_id"}
        )

    source = side(contrast.source_label, "source")
    target = side(contrast.target_label, "target")
    pairs = source.merge(target, on="group_id", how="inner", validate="one_to_one")
    if not pairs["source_premise"].eq(pairs["target_premise"]).all():
        raise ValueError("Same-prompt MNLI pairs unexpectedly contain different premises")
    pairs["pair_type"] = contrast.name
    pairs["split"] = split
    pairs["length_difference"] = (pairs["source_text_length"] - pairs["target_text_length"]).abs()
    pairs["pair_hash"] = [
        _stable_hash(f"{contrast.name}:{left}:{right}", seed)
        for left, right in zip(pairs["source_item_id"], pairs["target_item_id"])
    ]
    # Round-robin over genres before truncation so one easy genre cannot
    # dominate the matched control merely because its examples are shorter.
    pairs = pairs.sort_values(["source_genre", "length_difference", "pair_hash"]).copy()
    pairs["genre_rank"] = pairs.groupby("source_genre").cumcount()
    pairs = pairs.sort_values(["genre_rank", "pair_hash"]).head(max_pairs).copy()
    pairs["pair_id"] = [
        f"mnli:{contrast.name}:{left}:{right}"
        for left, right in zip(pairs["source_item_id"], pairs["target_item_id"])
    ]
    return pairs.reset_index(drop=True)


def build_mnli_pairs(items: pd.DataFrame, config: MnliControlConfig) -> pd.DataFrame:
    frames = []
    for contrast in config.contrasts:
        frames.append(
            _matched_label_pairs(
                items,
                contrast,
                split="train",
                max_pairs=config.max_train_pairs_per_contrast,
                seed=config.seed,
            )
        )
        frames.append(
            _matched_label_pairs(
                items,
                contrast,
                split="test",
                max_pairs=config.max_test_pairs_per_contrast,
                seed=config.seed + 7,
            )
        )
    pairs = pd.concat(frames, ignore_index=True, sort=False)
    expected = {(contrast.name, split) for contrast in config.contrasts for split in ("train", "test")}
    observed = set(map(tuple, pairs[["pair_type", "split"]].drop_duplicates().to_numpy()))
    if observed != expected:
        raise ValueError(f"MNLI pairing did not produce every contrast/split: missing {sorted(expected - observed)}")
    return pairs


def _interface_prefix(row: pd.Series, template: MnliTemplate, interface: MnliInterface) -> str:
    lines = [f"Premise: {row['premise']}", f"Hypothesis: {row['hypothesis']}"]
    lines.extend(interface.key_lines)
    lines.append(template.question)
    if interface.kind == "letter_mcq":
        for index, label in enumerate(interface.option_order):
            lines.append(f"{chr(ord('A') + index)}. {interface.label_values[label]}")
    else:
        values = ", ".join(interface.label_values[label] for label in MNLI_LABELS)
        lines.append(f"Valid completions: {values}.")
    lines.extend([template.answer_instruction, "Answer:"])
    return "\n".join(lines)


def _candidate_values(interface: MnliInterface) -> dict[str, str]:
    if interface.kind == "completion":
        return interface.label_values
    return {label: chr(ord("A") + interface.option_order.index(label)) for label in MNLI_LABELS}


def _canonical_interface(config: MnliControlConfig) -> MnliInterface:
    return next(interface for interface in config.interfaces if interface.name == config.reference_interface)


def _endpoint_table(pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for pair in pairs[pairs["split"].eq(split)].itertuples():
        for endpoint, prefix, base_label, target_label, sign in [
            ("source", "source", pair.source_label, pair.target_label, 1.0),
            ("target", "target", pair.target_label, pair.source_label, -1.0),
        ]:
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "pair_type": pair.pair_type,
                    "endpoint": endpoint,
                    "direction_sign": sign,
                    "base_label": base_label,
                    "target_label": target_label,
                    "item_id": getattr(pair, f"{prefix}_item_id"),
                    "premise": getattr(pair, f"{prefix}_premise"),
                    "hypothesis": getattr(pair, f"{prefix}_hypothesis"),
                    "genre": getattr(pair, f"{prefix}_genre"),
                    "text_length": getattr(pair, f"{prefix}_text_length"),
                }
            )
    return pd.DataFrame(rows)


def _extract_directions(
    model: Any,
    tokenizer: Any,
    pairs: pd.DataFrame,
    config: MnliControlConfig,
    model_cfg: MnliModelRun,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    canonical = _canonical_interface(config)
    template = config.templates[0]
    endpoints = _endpoint_table(pairs, "train")
    unique = endpoints.drop_duplicates("item_id").reset_index(drop=True)
    prompts = [_interface_prefix(row, template, canonical) for _, row in unique.iterrows()]
    resolved = resolve_layer_index(model_cfg.locked_layer, len(decoder_layers(model)))
    _, activations = collect_choice_probs_and_activations(
        model,
        tokenizer,
        prompts,
        layer_indices=[resolved],
        batch_size=config.batch_size,
        max_length=config.max_length,
        choice_letters=["A", "B", "C"],
    )
    item_index = {item_id: index for index, item_id in enumerate(unique["item_id"].astype(str))}
    hidden = activations[resolved]
    rng = np.random.default_rng(config.seed + resolved)
    directions: dict[tuple[str, str], np.ndarray] = {}
    inventory = []
    for contrast in config.contrasts:
        group = pairs[(pairs["split"].eq("train")) & (pairs["pair_type"].eq(contrast.name))]
        source_indices = [item_index[str(value)] for value in group["source_item_id"]]
        target_indices = [item_index[str(value)] for value in group["target_item_id"]]
        raw = (hidden[target_indices] - hidden[source_indices]).mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        random = rng.normal(size=raw.shape).astype(np.float32)
        random = random / max(float(np.linalg.norm(random)), 1e-8) * norm
        modes = {
            "raw_mnli_direction": raw,
            "random_direction_control": random.astype(np.float32),
            "inverse_direction_control": -raw,
        }
        for mode, vector in modes.items():
            directions[(contrast.name, mode)] = vector
            inventory.append(
                {
                    "pair_type": contrast.name,
                    "source_label": contrast.source_label,
                    "target_label": contrast.target_label,
                    "mode": mode,
                    "extraction_interface": canonical.name,
                    "extraction_template": template.name,
                    "layer_index": resolved,
                    "direction_l2": float(np.linalg.norm(vector)),
                    "n_train_pairs": len(group),
                    "pair_protocol": "same_premise_different_hypothesis_group_split",
                }
            )
    return directions, pd.DataFrame(inventory)


def _evaluate(
    model: Any,
    tokenizer: Any,
    pairs: pd.DataFrame,
    directions: dict[tuple[str, str], np.ndarray],
    config: MnliControlConfig,
    model_cfg: MnliModelRun,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoints = _endpoint_table(pairs, "test")
    rows, tokens = [], []
    for interface in config.interfaces:
        candidates = _candidate_values(interface)
        for template in config.templates:
            prefixes = [_interface_prefix(row, template, interface) for _, row in endpoints.iterrows()]
            completions = [[" " + candidates[label] for label in MNLI_LABELS] for _ in prefixes]
            base_sum, base_mean, counts = completion_sequence_logprobs(
                model, tokenizer, prefixes, completions,
                batch_size=config.batch_size, max_length=config.max_length,
            )
            for label_index, label in enumerate(MNLI_LABELS):
                for count in np.unique(counts[:, label_index]):
                    tokens.append(
                        {
                            "interface": interface.name,
                            "template": template.name,
                            "label": label,
                            "completion": candidates[label],
                            "n_tokens": int(count),
                        }
                    )
            for (pair_type, mode), direction in directions.items():
                pair_mask = endpoints["pair_type"].astype(str).eq(pair_type).to_numpy()
                for sign in (1.0, -1.0):
                    mask = pair_mask & endpoints["direction_sign"].eq(sign).to_numpy()
                    if not mask.any():
                        continue
                    indices = np.flatnonzero(mask)
                    subset = endpoints.loc[mask].reset_index(drop=True)
                    patched_sum, patched_mean, _ = completion_sequence_logprobs(
                        model,
                        tokenizer,
                        [prefixes[index] for index in indices],
                        [completions[index] for index in indices],
                        batch_size=config.batch_size,
                        max_length=config.max_length,
                        layer_index=model_cfg.locked_layer,
                        direction=np.asarray(direction) * sign,
                        alpha=model_cfg.locked_alpha,
                    )
                    base_scores = base_mean[mask] if config.primary_score == "mean_logprob" else base_sum[mask]
                    patched_scores = patched_mean if config.primary_score == "mean_logprob" else patched_sum
                    for index, item in subset.iterrows():
                        target = MNLI_LABELS.index(item.target_label)
                        source = MNLI_LABELS.index(item.base_label)
                        base_margin = base_scores[index, target] - base_scores[index, source]
                        patched_margin = patched_scores[index, target] - patched_scores[index, source]
                        rows.append(
                            {
                                "pair_id": item.pair_id,
                                "pair_type": item.pair_type,
                                "endpoint": item.endpoint,
                                "direction_sign": sign,
                                "mode": mode,
                                "interface": interface.name,
                                "interface_kind": interface.kind,
                                "template": template.name,
                                "base_label": item.base_label,
                                "target_label": item.target_label,
                                "base_target_margin": float(base_margin),
                                "patched_target_margin": float(patched_margin),
                                "target_margin_gain": float(patched_margin - base_margin),
                                "score_normalization": config.primary_score,
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(tokens).drop_duplicates()


def _summaries(rows: pd.DataFrame, config: MnliControlConfig) -> dict[str, pd.DataFrame]:
    if rows.empty:
        return {
            "mnli_pair_effects": pd.DataFrame(),
            "mnli_interface_summary": pd.DataFrame(),
            "mnli_interface_retention": pd.DataFrame(),
        }
    group_prefix = ["model_alias", "model_name"] if "model_alias" in rows else []
    pair = (
        rows.groupby(
            [*group_prefix, "pair_id", "pair_type", "mode", "interface", "interface_kind", "template"],
            as_index=False,
        )
        .agg(target_margin_gain=("target_margin_gain", "mean"))
    )
    summary = (
        pair.groupby([*group_prefix, "mode", "interface", "interface_kind", "template", "pair_type"], as_index=False)
        .agg(n_pairs=("pair_id", "nunique"), target_margin_gain=("target_margin_gain", "mean"))
    )
    reference = pair[pair["interface"].eq(config.reference_interface)].rename(
        columns={"target_margin_gain": "reference_gain"}
    )
    merge_keys = [*group_prefix, "pair_id", "pair_type", "mode", "template"]
    retention = pair.merge(reference[[*merge_keys, "reference_gain"]], on=merge_keys, how="left")
    eligible = retention["reference_gain"].abs().ge(0.01)
    retention["retention_ratio"] = np.where(
        eligible,
        retention["target_margin_gain"] / retention["reference_gain"],
        np.nan,
    )
    retention["sign_consistent"] = np.where(
        eligible,
        np.sign(retention["target_margin_gain"]) == np.sign(retention["reference_gain"]),
        np.nan,
    )
    return {
        "mnli_pair_effects": pair,
        "mnli_interface_summary": summary,
        "mnli_interface_retention": retention,
    }


def _equal_strata_bootstrap(
    values_by_stratum: list[np.ndarray],
    *,
    n_boot: int,
    confidence: float,
    chunk_size: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values_by_stratum = [np.asarray(values, dtype=float) for values in values_by_stratum if len(values)]
    if not values_by_stratum:
        return np.nan, np.nan, np.nan
    estimate = float(np.mean([values.mean() for values in values_by_stratum]))
    boot = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        chunk = np.zeros(stop - start, dtype=float)
        for values in values_by_stratum:
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            chunk += values[indices].mean(axis=1) / len(values_by_stratum)
        boot[start:stop] = chunk
    tail = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail))


def _ci_record(
    frame: pd.DataFrame,
    *,
    strata_columns: list[str],
    config: MnliControlConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    strata = [
        group["effect"].astype(float).to_numpy()
        for _, group in frame.groupby(strata_columns, sort=True)
    ]
    estimate, low, high = _equal_strata_bootstrap(
        strata,
        n_boot=config.n_boot,
        confidence=config.confidence,
        chunk_size=config.bootstrap_chunk_size,
        rng=rng,
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "n_pairs": int(frame["pair_id"].nunique()),
        "n_strata": len(strata),
        "confidence": config.confidence,
        "estimand": "equal_strata_pair_bootstrap",
    }


def build_mnli_paired_statistics(
    pair_effects: pd.DataFrame,
    config: MnliControlConfig,
) -> dict[str, pd.DataFrame]:
    required = {"model_alias", "pair_id", "pair_type", "mode", "interface", "template", "target_margin_gain"}
    missing = sorted(required - set(pair_effects.columns))
    if missing:
        raise ValueError(f"MNLI pair effects are missing columns: {missing}")
    # Prompt templates are repeated measurements of the same independent pair.
    pair = (
        pair_effects.groupby(
            ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface"],
            as_index=False,
        )
        .agg(effect=("target_margin_gain", "mean"), n_templates=("template", "nunique"))
    )
    rng = np.random.default_rng(config.seed + 101)

    mode_rows = []
    for (model_alias, model_name, mode, interface), group in pair.groupby(
        ["model_alias", "model_name", "mode", "interface"], sort=True
    ):
        record = {
            "model_alias": model_alias,
            "model_name": model_name,
            "mode": mode,
            "interface": interface,
            "comparison": f"{mode}_vs_zero",
        }
        record.update(_ci_record(group, strata_columns=["pair_type"], config=config, rng=rng))
        mode_rows.append(record)

    def paired_mode(left_mode: str, right_mode: str, comparison: str) -> pd.DataFrame:
        keys = ["model_alias", "model_name", "pair_id", "pair_type", "interface"]
        left = pair[pair["mode"].eq(left_mode)][keys + ["effect"]].rename(columns={"effect": "left"})
        right = pair[pair["mode"].eq(right_mode)][keys + ["effect"]].rename(columns={"effect": "right"})
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
        merged["effect"] = merged["left"] - merged["right"]
        merged["comparison"] = comparison
        return merged

    comparisons = [
        paired_mode("raw_mnli_direction", "random_direction_control", "raw_minus_random"),
        paired_mode("raw_mnli_direction", "inverse_direction_control", "raw_minus_inverse"),
    ]
    raw = pair[pair["mode"].eq("raw_mnli_direction")].copy()
    interface_specs = [
        ("letter_canonical", "letter_reversed", "canonical_minus_reversed"),
        ("direct_label_completion", "letter_canonical", "direct_minus_canonical"),
        ("opaque_completion", "letter_canonical", "opaque_minus_canonical"),
    ]
    for left_interface, right_interface, comparison in interface_specs:
        keys = ["model_alias", "model_name", "pair_id", "pair_type"]
        left = raw[raw["interface"].eq(left_interface)][keys + ["effect"]].rename(columns={"effect": "left"})
        right = raw[raw["interface"].eq(right_interface)][keys + ["effect"]].rename(columns={"effect": "right"})
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
        merged["effect"] = merged["left"] - merged["right"]
        merged["comparison"] = comparison
        merged["interface"] = f"{left_interface}_vs_{right_interface}"
        comparisons.append(merged)
    comparison_pairs = pd.concat(comparisons, ignore_index=True, sort=False)

    comparison_rows = []
    for (model_alias, model_name, comparison, interface), group in comparison_pairs.groupby(
        ["model_alias", "model_name", "comparison", "interface"], sort=True
    ):
        record = {
            "model_alias": model_alias,
            "model_name": model_name,
            "comparison": comparison,
            "interface": interface,
        }
        record.update(_ci_record(group, strata_columns=["pair_type"], config=config, rng=rng))
        comparison_rows.append(record)

    global_rows = []
    global_inputs = []
    raw_global = pair[pair["mode"].eq("raw_mnli_direction")].copy()
    for interface, group in raw_global.groupby("interface", sort=True):
        keep = group.copy()
        keep["comparison"] = "raw_vs_zero"
        keep["comparison_interface"] = interface
        global_inputs.append(keep)
    for (comparison, interface), group in comparison_pairs.groupby(["comparison", "interface"], sort=True):
        keep = group.copy()
        keep["comparison_interface"] = str(interface)
        global_inputs.append(keep)
    for group in global_inputs:
        record = {
            "comparison": str(group["comparison"].iloc[0]),
            "interface": str(group["comparison_interface"].iloc[0]),
        }
        record.update(
            _ci_record(group, strata_columns=["model_alias", "pair_type"], config=config, rng=rng)
        )
        record["n_models"] = int(group["model_alias"].nunique())
        record["n_contrasts"] = int(group["pair_type"].nunique())
        global_rows.append(record)

    leave_one_out = []
    for held_out in sorted(pair["model_alias"].astype(str).unique()):
        for group in global_inputs:
            selected = group[group["model_alias"].astype(str).ne(held_out)]
            record = {
                "held_out_model": held_out,
                "comparison": str(group["comparison"].iloc[0]),
                "interface": str(group["comparison_interface"].iloc[0]),
                "estimate": float(
                    selected.groupby(["model_alias", "pair_type"])["effect"].mean().mean()
                ),
                "n_models": int(selected["model_alias"].nunique()),
                "n_contrasts": int(selected["pair_type"].nunique()),
            }
            leave_one_out.append(record)
    return {
        "mnli_mode_interface_bootstrap_ci": pd.DataFrame(mode_rows),
        "mnli_paired_comparison_bootstrap_ci": pd.DataFrame(comparison_rows),
        "mnli_global_bootstrap_ci": pd.DataFrame(global_rows),
        "mnli_leave_one_model_out": pd.DataFrame(leave_one_out),
        "mnli_statistics_pair_rows": pair,
        "mnli_statistics_comparison_pairs": comparison_pairs,
    }


def run_mnli_statistics_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = MnliControlConfig.from_json(config_path, project_root=project_root)
    pair_path = config.output_dir / "mnli_pair_effects.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"MNLI pair effects do not exist: {pair_path}")
    tables = build_mnli_paired_statistics(pd.read_csv(pair_path), config)
    tables["mnli_statistics_config"] = pd.DataFrame(
        [
            {
                "n_boot": config.n_boot,
                "confidence": config.confidence,
                "bootstrap_chunk_size": config.bootstrap_chunk_size,
                "seed": config.seed,
                "independent_unit": "matched_same_premise_pair",
                "stratification": "equal_model_equal_contrast",
            }
        ]
    )
    write_tables(tables, config.output_dir / "paired_statistics")
    return tables


def run_mnli_control_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    config = MnliControlConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    items = load_mnli_items(config)
    pairs = build_mnli_pairs(items, config)
    requested = set(model_aliases or [item.model.alias for item in config.models])
    errors = []
    for model_cfg in config.models:
        if model_cfg.model.alias not in requested:
            continue
        model_dir = config.output_dir / model_cfg.model.alias
        if (model_dir / "mnli_run_complete.csv").exists() and not config.force_rerun:
            continue
        tokenizer = model = None
        try:
            tokenizer, model = load_tokenizer_and_model(
                model_cfg.model.load_source,
                device_map=model_cfg.model.device_map,
                torch_dtype=model_cfg.model.torch_dtype,
            )
            directions, inventory = _extract_directions(model, tokenizer, pairs, config, model_cfg)
            rows, tokenization = _evaluate(model, tokenizer, pairs, directions, config, model_cfg)
            for frame in (rows, tokenization, inventory):
                frame.insert(0, "model_alias", model_cfg.model.alias)
                frame.insert(1, "model_name", model_cfg.model.name)
            write_tables(
                {
                    "mnli_eval_rows": rows,
                    "mnli_tokenization": tokenization,
                    "mnli_direction_inventory": inventory,
                    "mnli_run_complete": pd.DataFrame(
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
        except Exception as exc:
            errors.append(
                {
                    "model_alias": model_cfg.model.alias,
                    "model_name": model_cfg.model.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            del model, tokenizer
            cleanup_model()

    row_files = sorted(config.output_dir.glob("*/mnli_eval_rows.csv"))
    inventory_files = sorted(config.output_dir.glob("*/mnli_direction_inventory.csv"))
    token_files = sorted(config.output_dir.glob("*/mnli_tokenization.csv"))
    rows = pd.concat([pd.read_csv(path) for path in row_files], ignore_index=True, sort=False) if row_files else pd.DataFrame()
    tables = _summaries(rows, config)
    pair_inventory = (
        pairs.groupby(["split", "pair_type"], as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_length_difference=("length_difference", "mean"),
            median_length_difference=("length_difference", "median"),
            n_genres=("source_genre", "nunique"),
        )
    )
    tables.update(
        {
            "mnli_eval_rows": rows,
            "mnli_direction_inventory": pd.concat([pd.read_csv(path) for path in inventory_files], ignore_index=True, sort=False) if inventory_files else pd.DataFrame(),
            "mnli_tokenization": pd.concat([pd.read_csv(path) for path in token_files], ignore_index=True, sort=False) if token_files else pd.DataFrame(),
            "mnli_pair_inventory": pair_inventory,
            "mnli_pair_examples": pairs.head(30),
            "mnli_errors": pd.DataFrame(errors),
            "mnli_control_config": pd.DataFrame(
                [
                    {
                        "protocol_version": "v2_same_premise",
                        "split_unit": "normalized_premise_hash",
                        "pair_protocol": "same_premise_different_hypothesis",
                        "reference_interface": config.reference_interface,
                        "primary_score": config.primary_score,
                    }
                ]
            ),
        }
    )
    write_tables(tables, config.output_dir)
    return tables
