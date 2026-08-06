"""ITI-style head-probe intervention under exhaustive answer remapping.

The implementation follows the core ITI protocol shape: train linear probes
on individual attention-head outputs, select a small set of heads on held-out
validation data, and shift only those head outputs at inference time.  It is
an ITI-style adaptation rather than an exact TruthfulQA reproduction because
the probes and validation objective are defined by NormBank contrasts.
"""
from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cross_interface_audit import (
    CrossInterfaceConfig,
    InterfaceDefinition,
    PromptTemplate,
    _candidate_values,
    _canonical_mapping,
    _interface_prefix,
    _original_slot_labels,
)
from .io import write_tables
from .mapping_audit import (
    _load_mapping_audit_config,
    build_mapping_items,
    load_base_items,
)
from .steering import (
    _gather_last_token,
    _input_device,
    _last_token_indices,
    choice_token_ids,
    cleanup_model,
    decoder_layers,
    load_tokenizer_and_model,
)


@dataclass(frozen=True)
class HeadDirection:
    layer_index: int
    head_index: int
    vector: np.ndarray
    scale: float
    validation_accuracy: float
    train_accuracy: float


@dataclass(frozen=True)
class ITIProbeSpec:
    direction_method: str
    candidate_layer_fractions: tuple[float, ...]
    top_k_heads_grid: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    ridge: float
    random_control_seeds: tuple[int, ...]
    selection_template: str
    minimum_validation_gain: float
    n_boot: int
    confidence: float
    seed: int
    comparison_caa_bootstrap_path: Path | None


@dataclass(frozen=True)
class ITIProbeAuditConfig:
    cross_interface: CrossInterfaceConfig
    spec: ITIProbeSpec

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "ITIProbeAuditConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        cross_interface = CrossInterfaceConfig.from_json(
            config_path,
            project_root=project_root,
            model_source_overrides=model_source_overrides,
        )
        raw = dict(data.get("iti_probe", {}))
        comparison_path_value = raw.get("comparison_caa_bootstrap_path")
        comparison_path = None
        if comparison_path_value:
            comparison_path = Path(str(comparison_path_value)).expanduser()
            if not comparison_path.is_absolute():
                comparison_path = cross_interface.audit.project_root / comparison_path
            comparison_path = comparison_path.resolve()
        spec = ITIProbeSpec(
            direction_method=str(raw.get("direction_method", "mass_mean_shift")),
            candidate_layer_fractions=tuple(
                float(value)
                for value in raw.get("candidate_layer_fractions", [0.5, 0.625, 0.75, 0.875])
            ),
            top_k_heads_grid=tuple(
                int(value) for value in raw.get("top_k_heads_grid", [8, 16, 32])
            ),
            alpha_grid=tuple(
                float(value) for value in raw.get("alpha_grid", [0.25, 0.5, 1.0])
            ),
            ridge=float(raw.get("ridge", 0.01)),
            random_control_seeds=tuple(
                int(value) for value in raw.get("random_control_seeds", [13, 29, 47, 71, 101])
            ),
            selection_template=str(raw.get("selection_template", "classify")),
            minimum_validation_gain=float(raw.get("minimum_validation_gain", 0.0)),
            n_boot=int(raw.get("n_boot", 5_000)),
            confidence=float(raw.get("confidence", 0.95)),
            seed=int(raw.get("seed", 13)),
            comparison_caa_bootstrap_path=comparison_path,
        )
        config = cls(cross_interface=cross_interface, spec=spec)
        config.validate()
        return config

    def validate(self) -> None:
        spec = self.spec
        if spec.direction_method not in {"mass_mean_shift", "probe_weight"}:
            raise ValueError(
                "iti_probe.direction_method must be mass_mean_shift or probe_weight"
            )
        if not spec.candidate_layer_fractions:
            raise ValueError("iti_probe.candidate_layer_fractions cannot be empty")
        if any(not 0.0 <= value <= 1.0 for value in spec.candidate_layer_fractions):
            raise ValueError("ITI candidate layer fractions must lie in [0, 1]")
        if not spec.top_k_heads_grid or any(value <= 0 for value in spec.top_k_heads_grid):
            raise ValueError("iti_probe.top_k_heads_grid must contain positive integers")
        if not spec.alpha_grid or any(value <= 0 for value in spec.alpha_grid):
            raise ValueError("iti_probe.alpha_grid must contain positive values")
        if spec.ridge < 0:
            raise ValueError("iti_probe.ridge must be non-negative")
        if len(spec.random_control_seeds) < 1:
            raise ValueError("ITI audit requires at least one random control seed")
        if spec.n_boot <= 0 or not 0.0 < spec.confidence < 1.0:
            raise ValueError("ITI bootstrap settings are invalid")
        template_names = {template.name for template in self.cross_interface.templates}
        if spec.selection_template not in template_names:
            raise ValueError(
                f"Unknown ITI selection template {spec.selection_template!r}; "
                f"known templates: {sorted(template_names)}"
            )
        interfaces = self.cross_interface.interfaces
        if not interfaces or any(interface.kind != "letter_mcq" for interface in interfaces):
            raise ValueError("ITI exhaustive audit currently supports letter_mcq interfaces only")
        if len(interfaces) != 6:
            raise ValueError("ITI NormBank audit requires all six A/B/C mappings")


def resolve_candidate_layers(n_layers: int, fractions: Iterable[float]) -> tuple[int, ...]:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    return tuple(
        sorted(
            {
                min(n_layers - 1, max(0, int(round(float(fraction) * (n_layers - 1)))))
                for fraction in fractions
            }
        )
    )


def _attention_output_projection(layer: Any) -> Any:
    candidates = [
        ("self_attn", "o_proj"),
        ("self_attn", "out_proj"),
        ("attention", "o_proj"),
        ("attention", "wo"),
    ]
    for parent_name, projection_name in candidates:
        parent = getattr(layer, parent_name, None)
        projection = getattr(parent, projection_name, None) if parent is not None else None
        if projection is not None:
            return projection
    raise AttributeError("Could not locate the attention output projection for ITI")


def _head_geometry(model: Any, layer: Any) -> tuple[int, int]:
    projection = _attention_output_projection(layer)
    width = int(getattr(projection, "in_features", 0))
    n_heads = int(getattr(model.config, "num_attention_heads", 0))
    if width <= 0 or n_heads <= 0 or width % n_heads:
        raise ValueError(
            f"Invalid attention-head geometry: projection width={width}, heads={n_heads}"
        )
    return n_heads, width // n_heads


def collect_attention_head_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    layer_indices: Iterable[int],
    batch_size: int,
    max_length: int,
) -> dict[int, np.ndarray]:
    """Collect pre-o_proj head outputs at the final attended prompt token."""
    import torch

    layers = decoder_layers(model)
    selected_layers = tuple(int(value) for value in layer_indices)
    if any(value < 0 or value >= len(layers) for value in selected_layers):
        raise ValueError("ITI candidate layer index is outside the decoder")
    chunks: dict[int, list[np.ndarray]] = {value: [] for value in selected_layers}
    current_positions = None
    handles = []

    def make_hook(layer_index: int, n_heads: int, head_dim: int) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if current_positions is None:
                return None
            hidden = inputs[0]
            if hidden.shape[-1] != n_heads * head_dim:
                raise ValueError(
                    f"Unexpected attention width at layer {layer_index}: {hidden.shape[-1]}"
                )
            view = hidden.reshape(hidden.shape[0], hidden.shape[1], n_heads, head_dim)
            positions = current_positions.to(hidden.device)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            selected = view[batch_indices, positions, :, :]
            chunks[layer_index].append(selected.detach().float().cpu().numpy())
            return None

        return hook

    for layer_index in selected_layers:
        n_heads, head_dim = _head_geometry(model, layers[layer_index])
        projection = _attention_output_projection(layers[layer_index])
        handles.append(
            projection.register_forward_pre_hook(make_hook(layer_index, n_heads, head_dim))
        )

    device = _input_device(model)
    try:
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            current_positions = _last_token_indices(encoded["attention_mask"])
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                model(**encoded)
            current_positions = None
    finally:
        for handle in handles:
            handle.remove()
    return {
        layer_index: np.concatenate(layer_chunks, axis=0).astype(np.float32)
        for layer_index, layer_chunks in chunks.items()
    }


def fit_ridge_head_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    """Fit a standardized linear ridge probe and orient it toward label 1."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    y_validation = np.asarray(y_validation, dtype=int)
    if len(x_train) < 2 or len(np.unique(y_train)) != 2:
        raise ValueError("A head probe requires both labels in training data")
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_norm = (x_train - mean) / std
    signed = np.where(y_train > 0, 1.0, -1.0)
    lhs = x_norm.T @ x_norm + float(ridge) * np.eye(x_norm.shape[1])
    rhs = x_norm.T @ signed
    try:
        weight_scaled = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weight_scaled = np.linalg.pinv(lhs) @ rhs
    bias = float(signed.mean() - x_norm.mean(axis=0) @ weight_scaled)
    weight_raw = weight_scaled / std
    weight_norm = float(np.linalg.norm(weight_raw))
    if weight_norm <= 1e-12:
        direction = np.zeros_like(weight_raw)
    else:
        direction = weight_raw / weight_norm

    def scores(values: np.ndarray) -> np.ndarray:
        return ((values - mean) / std) @ weight_scaled + bias

    train_scores = scores(x_train)
    validation_scores = scores(x_validation)
    train_accuracy = float(((train_scores >= 0).astype(int) == y_train).mean())
    validation_accuracy = float(
        ((validation_scores >= 0).astype(int) == y_validation).mean()
    )
    positive_mean = x_train[y_train > 0].mean(axis=0)
    negative_mean = x_train[y_train <= 0].mean(axis=0)
    mass_mean_vector = positive_mean - negative_mean
    mass_mean_norm = float(np.linalg.norm(mass_mean_vector))
    mass_mean_direction = (
        mass_mean_vector / mass_mean_norm
        if mass_mean_norm > 1e-12
        else np.zeros_like(mass_mean_vector)
    )
    combined = np.concatenate([x_train, x_validation], axis=0)

    def projection_std(selected_direction: np.ndarray) -> float:
        value = float(np.std(combined @ selected_direction))
        return value if np.isfinite(value) and value > 1e-6 else 1.0

    return {
        "direction": direction.astype(np.float32),
        "probe_weight_direction": direction.astype(np.float32),
        "probe_weight_scale": projection_std(direction),
        "mass_mean_shift_direction": mass_mean_direction.astype(np.float32),
        "mass_mean_shift_scale": projection_std(mass_mean_direction),
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "weight_l2": weight_norm,
        "mass_mean_shift_l2": mass_mean_norm,
    }


def _probe_items(canonical_items: pd.DataFrame, config: ITIProbeAuditConfig) -> pd.DataFrame:
    dataset = config.cross_interface.audit.dataset
    selected = canonical_items[
        canonical_items[dataset.split_column]
        .astype(str)
        .isin([dataset.train_split, dataset.validation_split])
    ].copy()
    if selected.empty:
        raise ValueError("No train/validation endpoints are available for ITI probes")
    return selected.reset_index(drop=True)


def build_iti_input_inventory(
    base_items: pd.DataFrame,
    config: ITIProbeAuditConfig,
) -> pd.DataFrame:
    """Validate that every observed contrast has train, validation, and test pairs."""
    dataset = config.cross_interface.audit.dataset
    inventory = (
        base_items.groupby(
            [dataset.contrast_column, dataset.split_column],
            as_index=False,
        )
        .agg(
            n_pairs=(dataset.pair_id_column, "nunique"),
            n_endpoints=(dataset.label_column, "size"),
        )
        .rename(
            columns={
                dataset.contrast_column: "pair_type",
                dataset.split_column: "split",
            }
        )
        .sort_values(["pair_type", "split"])
        .reset_index(drop=True)
    )
    required_splits = (
        dataset.train_split,
        dataset.validation_split,
        dataset.test_split,
    )
    pair_types = sorted(inventory["pair_type"].astype(str).unique())
    observed = set(
        map(
            tuple,
            inventory[["pair_type", "split"]].astype(str).to_numpy(),
        )
    )
    missing = [
        (pair_type, split)
        for pair_type in pair_types
        for split in required_splits
        if (pair_type, split) not in observed
    ]
    if not pair_types:
        raise ValueError("The ITI input table contains no contrasts")
    if missing:
        available = {
            str(pair_type): sorted(group["split"].astype(str).unique())
            for pair_type, group in inventory.groupby("pair_type", sort=True)
        }
        raise ValueError(
            "ITI input split inventory is incomplete. "
            f"Missing pair_type/split cells: {missing}. "
            f"Expected split names: {required_splits}; observed: {available}"
        )
    return inventory


def build_head_probe_bank(
    probe_items: pd.DataFrame,
    activations: dict[int, np.ndarray],
    config: ITIProbeAuditConfig,
) -> tuple[dict[str, list[HeadDirection]], pd.DataFrame]:
    dataset = config.cross_interface.audit.dataset
    banks: dict[str, list[HeadDirection]] = {}
    inventory_rows = []
    for pair_type, group in probe_items.groupby(dataset.contrast_column, sort=True):
        group_indices = group.index.to_numpy(dtype=int)
        ranks = group["semantic_rank"].to_numpy(dtype=float)
        low_rank, high_rank = float(ranks.min()), float(ranks.max())
        labels = (ranks == high_rank).astype(int)
        train_local = group[dataset.split_column].astype(str).eq(dataset.train_split).to_numpy()
        validation_local = group[dataset.split_column].astype(str).eq(
            dataset.validation_split
        ).to_numpy()
        if not train_local.any() or not validation_local.any():
            raise ValueError(f"Contrast {pair_type!r} lacks train or validation ITI items")
        directions = []
        for layer_index, layer_values in activations.items():
            values = layer_values[group_indices]
            for head_index in range(values.shape[1]):
                state = fit_ridge_head_probe(
                    values[train_local, head_index, :],
                    labels[train_local],
                    values[validation_local, head_index, :],
                    labels[validation_local],
                    ridge=config.spec.ridge,
                )
                direction_key = f"{config.spec.direction_method}_direction"
                scale_key = f"{config.spec.direction_method}_scale"
                direction = HeadDirection(
                    layer_index=int(layer_index),
                    head_index=int(head_index),
                    vector=np.asarray(state[direction_key], dtype=np.float32),
                    scale=float(state[scale_key]),
                    validation_accuracy=float(state["validation_accuracy"]),
                    train_accuracy=float(state["train_accuracy"]),
                )
                directions.append(direction)
                inventory_rows.append(
                    {
                        "pair_type": str(pair_type),
                        "layer_index": int(layer_index),
                        "head_index": int(head_index),
                        "train_accuracy": direction.train_accuracy,
                        "validation_accuracy": direction.validation_accuracy,
                        "direction_l2": float(np.linalg.norm(direction.vector)),
                        "activation_projection_std": direction.scale,
                        "probe_weight_l2": float(state["weight_l2"]),
                        "mass_mean_shift_l2": float(state["mass_mean_shift_l2"]),
                        "direction_method": config.spec.direction_method,
                        "low_rank": low_rank,
                        "high_rank": high_rank,
                        "n_train_items": int(train_local.sum()),
                        "n_validation_items": int(validation_local.sum()),
                    }
                )
        banks[str(pair_type)] = sorted(
            directions,
            key=lambda item: (
                -item.validation_accuracy,
                -item.train_accuracy,
                item.layer_index,
                item.head_index,
            ),
        )
    inventory = pd.DataFrame(inventory_rows)
    if not inventory.empty:
        inventory["validation_rank"] = (
            inventory.groupby("pair_type")["validation_accuracy"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    return banks, inventory


def select_top_heads(
    ranked_banks: dict[str, list[HeadDirection]],
    top_k: int,
) -> dict[str, list[HeadDirection]]:
    return {
        pair_type: directions[: min(int(top_k), len(directions))]
        for pair_type, directions in ranked_banks.items()
    }


def randomize_head_bank(
    selected_banks: dict[str, list[HeadDirection]],
    *,
    seed: int,
) -> dict[str, list[HeadDirection]]:
    output = {}
    for pair_type, directions in selected_banks.items():
        digest = hashlib.sha256(f"{seed}:{pair_type}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        random_directions = []
        for direction in directions:
            vector = rng.standard_normal(len(direction.vector)).astype(np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            random_directions.append(
                HeadDirection(
                    layer_index=direction.layer_index,
                    head_index=direction.head_index,
                    vector=vector,
                    scale=direction.scale,
                    validation_accuracy=np.nan,
                    train_accuracy=np.nan,
                )
            )
        output[pair_type] = random_directions
    return output


def _build_eval_items(
    base_items: pd.DataFrame,
    config: ITIProbeAuditConfig,
    *,
    split: str,
) -> pd.DataFrame:
    dataset = config.cross_interface.audit.dataset
    rows = []
    selected = base_items[base_items[dataset.split_column].astype(str).eq(split)]
    for (pair_type, pair_id), group in selected.groupby(
        [dataset.contrast_column, dataset.pair_id_column], sort=True
    ):
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


def score_letter_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    candidate_letters: list[str],
    interventions: list[HeadDirection] | None,
    alpha: float,
    sign: float,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Score one-token letter candidates with optional per-head intervention."""
    import torch

    token_ids = choice_token_ids(tokenizer, candidate_letters)
    layers = decoder_layers(model)
    grouped: dict[int, list[HeadDirection]] = {}
    for direction in interventions or []:
        grouped.setdefault(direction.layer_index, []).append(direction)
    current_positions = None
    handles = []

    def make_hook(layer_index: int, head_directions: list[HeadDirection]) -> Any:
        n_heads, head_dim = _head_geometry(model, layers[layer_index])

        def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if current_positions is None:
                return None
            hidden = inputs[0].clone()
            if hidden.shape[-1] != n_heads * head_dim:
                raise ValueError(
                    f"Unexpected attention width at layer {layer_index}: {hidden.shape[-1]}"
                )
            view = hidden.reshape(hidden.shape[0], hidden.shape[1], n_heads, head_dim)
            positions = current_positions.to(hidden.device)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            for direction in head_directions:
                delta = torch.as_tensor(
                    direction.vector * direction.scale * float(alpha) * float(sign),
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                view[batch_indices, positions, direction.head_index, :] += delta
            return (view.reshape_as(hidden), *inputs[1:])

        return hook

    for layer_index, directions in grouped.items():
        handles.append(
            _attention_output_projection(layers[layer_index]).register_forward_pre_hook(
                make_hook(layer_index, directions)
            )
        )

    device = _input_device(model)
    batches = []
    try:
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            current_positions = _last_token_indices(encoded["attention_mask"])
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = _gather_last_token(model(**encoded).logits, encoded["attention_mask"])
                log_probs = torch.log_softmax(logits, dim=-1)[:, token_ids]
            batches.append(log_probs.detach().float().cpu().numpy())
            current_positions = None
    finally:
        for handle in handles:
            handle.remove()
    return (
        np.concatenate(batches, axis=0).astype(np.float32)
        if batches
        else np.zeros((0, len(token_ids)), dtype=np.float32)
    )


def _evaluate_banks(
    model: Any,
    tokenizer: Any,
    base_items: pd.DataFrame,
    eval_items: pd.DataFrame,
    config: ITIProbeAuditConfig,
    *,
    banks_by_mode: dict[str, dict[str, list[HeadDirection]]],
    mode_multipliers: dict[str, float],
    alpha: float,
    interfaces: Iterable[InterfaceDefinition],
    templates: Iterable[PromptTemplate],
) -> pd.DataFrame:
    cross = config.cross_interface
    dataset = cross.audit.dataset
    labels = list(dataset.label_ranks)
    rows = []
    for interface in interfaces:
        candidate_values = _candidate_values(cross, interface)
        candidate_letters = [candidate_values[label] for label in labels]
        for template in templates:
            prefixes = [
                _interface_prefix(base_items.loc[item.row_index], cross, template, interface)
                for item in eval_items.itertuples()
            ]
            base_scores = score_letter_prompts(
                model,
                tokenizer,
                prefixes,
                candidate_letters=candidate_letters,
                interventions=None,
                alpha=0.0,
                sign=1.0,
                batch_size=cross.audit.runtime.batch_size,
                max_length=cross.audit.runtime.max_length,
            )
            for mode, banks in banks_by_mode.items():
                for pair_type, interventions in banks.items():
                    pair_mask = eval_items["pair_type"].astype(str).eq(pair_type).to_numpy()
                    if not pair_mask.any():
                        continue
                    for direction_sign in (1.0, -1.0):
                        mask = pair_mask & eval_items["direction_sign"].eq(direction_sign).to_numpy()
                        if not mask.any():
                            continue
                        indices = np.flatnonzero(mask)
                        subset = eval_items.loc[mask].reset_index(drop=True)
                        patched = score_letter_prompts(
                            model,
                            tokenizer,
                            [prefixes[index] for index in indices],
                            candidate_letters=candidate_letters,
                            interventions=interventions,
                            alpha=alpha,
                            sign=direction_sign * float(mode_multipliers.get(mode, 1.0)),
                            batch_size=cross.audit.runtime.batch_size,
                            max_length=cross.audit.runtime.max_length,
                        )
                        subset_base = base_scores[mask]
                        for row_index, item in subset.iterrows():
                            target = labels.index(str(item.target_label))
                            source = labels.index(str(item.base_label))
                            current_gain = (
                                patched[row_index, target]
                                - patched[row_index, source]
                                - subset_base[row_index, target]
                                + subset_base[row_index, source]
                            )
                            original_target_label, original_source_label = _original_slot_labels(
                                cross,
                                interface,
                                target_label=str(item.target_label),
                                source_label=str(item.base_label),
                            )
                            original_gain = np.nan
                            if original_target_label is not None and original_source_label is not None:
                                original_target = labels.index(original_target_label)
                                original_source = labels.index(original_source_label)
                                original_gain = (
                                    patched[row_index, original_target]
                                    - patched[row_index, original_source]
                                    - subset_base[row_index, original_target]
                                    + subset_base[row_index, original_source]
                                )
                            rows.append(
                                {
                                    **item.to_dict(),
                                    "mode": mode,
                                    "interface": interface.name,
                                    "template": template.name,
                                    "alpha": float(alpha),
                                    "target_margin_gain": float(current_gain),
                                    "original_slot_margin_gain": float(original_gain),
                                    "n_intervened_heads": len(interventions),
                                }
                            )
    return pd.DataFrame(rows)


def _validation_grid(
    model: Any,
    tokenizer: Any,
    base_items: pd.DataFrame,
    ranked_banks: dict[str, list[HeadDirection]],
    config: ITIProbeAuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[HeadDirection]]]:
    dataset = config.cross_interface.audit.dataset
    validation_items = _build_eval_items(
        base_items,
        config,
        split=dataset.validation_split,
    )
    canonical_interface = next(
        interface
        for interface in config.cross_interface.interfaces
        if interface.name == config.cross_interface.reference_interface
    )
    selection_template = next(
        template
        for template in config.cross_interface.templates
        if template.name == config.spec.selection_template
    )
    grid_rows = []
    banks_by_k = {}
    for top_k in config.spec.top_k_heads_grid:
        selected = select_top_heads(ranked_banks, top_k)
        banks_by_k[int(top_k)] = selected
        for alpha in config.spec.alpha_grid:
            rows = _evaluate_banks(
                model,
                tokenizer,
                base_items,
                validation_items,
                config,
                banks_by_mode={"iti_probe_direction": selected},
                mode_multipliers={"iti_probe_direction": 1.0},
                alpha=float(alpha),
                interfaces=[canonical_interface],
                templates=[selection_template],
            )
            pair_means = (
                rows.groupby(["pair_id", "pair_type"], as_index=False)
                .agg(target_margin_gain=("target_margin_gain", "mean"))
            )
            contrast_means = pair_means.groupby("pair_type")["target_margin_gain"].mean()
            grid_rows.append(
                {
                    "top_k_heads": int(top_k),
                    "alpha": float(alpha),
                    "n_contrasts": int(len(contrast_means)),
                    "n_pairs": int(pair_means["pair_id"].nunique()),
                    "mean_validation_target_margin_gain": float(contrast_means.mean()),
                    "min_contrast_validation_gain": float(contrast_means.min()),
                    "n_positive_contrasts": int((contrast_means > 0).sum()),
                }
            )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["mean_validation_target_margin_gain", "top_k_heads", "alpha"],
        ascending=[False, True, True],
    )
    best = grid.iloc[0]
    selection = pd.DataFrame(
        [
            {
                **best.to_dict(),
                "selection_split": dataset.validation_split,
                "selection_interface": canonical_interface.name,
                "selection_template": selection_template.name,
                "validation_competence_pass": bool(
                    best["mean_validation_target_margin_gain"]
                    > config.spec.minimum_validation_gain
                ),
                "minimum_validation_gain": config.spec.minimum_validation_gain,
            }
        ]
    )
    return grid.reset_index(drop=True), selection, banks_by_k[int(best["top_k_heads"])]


def _pair_effects(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    keys = [
        "model_alias",
        "model_name",
        "pair_id",
        "pair_type",
        "mode",
        "interface",
        "template",
        "alpha",
    ]
    return (
        rows.groupby(keys, as_index=False)
        .agg(
            target_margin_gain=("target_margin_gain", "mean"),
            original_slot_margin_gain=("original_slot_margin_gain", "mean"),
            n_endpoints=("endpoint", "nunique"),
            n_intervened_heads=("n_intervened_heads", "max"),
        )
        .sort_values(keys)
    )


def _mode_summary(pair_effects: pd.DataFrame) -> pd.DataFrame:
    if pair_effects.empty:
        return pd.DataFrame()
    keys = ["model_alias", "model_name", "mode", "interface", "pair_type"]
    return (
        pair_effects.groupby(keys, as_index=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            target_margin_gain=("target_margin_gain", "mean"),
            original_slot_margin_gain=("original_slot_margin_gain", "mean"),
            n_intervened_heads=("n_intervened_heads", "max"),
        )
        .sort_values(keys)
    )


def adjusted_pair_effects(pair_effects: pd.DataFrame) -> pd.DataFrame:
    """Subtract the mean matched random-head-direction effect per pair."""
    if pair_effects.empty:
        return pd.DataFrame()
    averaged = (
        pair_effects.groupby(
            ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface"],
            as_index=False,
        )
        .agg(
            target_margin_gain=("target_margin_gain", "mean"),
            original_slot_margin_gain=("original_slot_margin_gain", "mean"),
        )
    )
    target = averaged[averaged["mode"].eq("iti_probe_direction")].copy()
    random = averaged[
        averaged["mode"].astype(str).str.startswith("iti_random_direction_seed_")
    ].copy()
    random = (
        random.groupby(
            ["model_alias", "model_name", "pair_id", "pair_type", "interface"],
            as_index=False,
        )
        .agg(
            random_target_gain=("target_margin_gain", "mean"),
            random_original_slot_gain=("original_slot_margin_gain", "mean"),
            n_random_controls=("mode", "nunique"),
        )
    )
    merged = target.merge(
        random,
        on=["model_alias", "model_name", "pair_id", "pair_type", "interface"],
        how="inner",
        validate="one_to_one",
    )
    frames = []
    for metric, target_column, random_column in [
        ("current_label_margin", "target_margin_gain", "random_target_gain"),
        (
            "extraction_slot_margin",
            "original_slot_margin_gain",
            "random_original_slot_gain",
        ),
    ]:
        frame = merged[
            [
                "model_alias",
                "model_name",
                "pair_id",
                "pair_type",
                "interface",
                "n_random_controls",
            ]
        ].copy()
        frame["metric"] = metric
        frame["target_gain"] = merged[target_column]
        frame["random_gain"] = merged[random_column]
        frame["adjusted_gain"] = frame["target_gain"] - frame["random_gain"]
        frames.append(frame)
    current, original = frames
    difference = current.copy()
    difference["metric"] = "current_label_minus_extraction_slot"
    difference["target_gain"] = current["target_gain"] - original["target_gain"]
    difference["random_gain"] = current["random_gain"] - original["random_gain"]
    difference["adjusted_gain"] = current["adjusted_gain"] - original["adjusted_gain"]
    return pd.concat([current, original, difference], ignore_index=True, sort=False)


def bootstrap_adjusted_effects(
    effects: pd.DataFrame,
    *,
    n_boot: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    output = []
    for (metric, interface), group in effects.groupby(["metric", "interface"], sort=True):
        strata = []
        for _, stratum in group.groupby(["model_alias", "pair_type"], sort=True):
            values = stratum.groupby("pair_id")["adjusted_gain"].mean().dropna().to_numpy()
            if len(values):
                strata.append(values)
        if not strata:
            continue
        estimate = float(np.mean([values.mean() for values in strata]))
        draws = np.empty(n_boot, dtype=np.float64)
        for index in range(n_boot):
            draws[index] = np.mean(
                [rng.choice(values, size=len(values), replace=True).mean() for values in strata]
            )
        stratum_means = group.groupby(["model_alias", "pair_type"])["adjusted_gain"].mean()
        output.append(
            {
                "metric": metric,
                "interface": interface,
                "n_models": int(group["model_alias"].nunique()),
                "n_contrasts": int(group["pair_type"].nunique()),
                "n_model_contrast_strata": int(len(strata)),
                "n_pairs": int(group["pair_id"].nunique()),
                "mean_adjusted_gain": estimate,
                "ci_low": float(np.quantile(draws, alpha)),
                "ci_high": float(np.quantile(draws, 1.0 - alpha)),
                "positive_strata": int((stratum_means > 0).sum()),
                "negative_strata": int((stratum_means < 0).sum()),
                "confidence": confidence,
                "n_boot": n_boot,
            }
        )
    return pd.DataFrame(output)


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def build_iti_competence_summary(
    selection: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the preregistered source-interface competence gate per model."""
    if selection.empty:
        return pd.DataFrame()
    duplicate_models = selection["model_alias"].duplicated(keep=False)
    if duplicate_models.any():
        aliases = sorted(selection.loc[duplicate_models, "model_alias"].astype(str).unique())
        raise ValueError(f"ITI selection contains duplicate model rows: {aliases}")
    columns = [
        "model_alias",
        "model_name",
        "top_k_heads",
        "alpha",
        "mean_validation_target_margin_gain",
        "min_contrast_validation_gain",
        "n_positive_contrasts",
        "n_contrasts",
        "validation_competence_pass",
        "minimum_validation_gain",
    ]
    summary = selection[[column for column in columns if column in selection.columns]].copy()
    summary["validation_competence_pass"] = _bool_series(
        summary["validation_competence_pass"]
    )
    if not inventory.empty:
        probe_rows = []
        for row in summary.itertuples(index=False):
            model_inventory = inventory[
                inventory["model_alias"].astype(str).eq(str(row.model_alias))
            ].copy()
            top_k = int(float(row.top_k_heads))
            selected_heads = model_inventory[
                pd.to_numeric(model_inventory["validation_rank"], errors="coerce") <= top_k
            ]
            contrast_probe = (
                selected_heads.groupby("pair_type", as_index=False)
                .agg(
                    mean_selected_head_validation_accuracy=(
                        "validation_accuracy",
                        "mean",
                    ),
                    max_head_validation_accuracy=("validation_accuracy", "max"),
                )
            )
            probe_rows.append(
                {
                    "model_alias": str(row.model_alias),
                    "mean_selected_head_validation_accuracy": float(
                        selected_heads["validation_accuracy"].mean()
                    ),
                    "min_contrast_selected_head_validation_accuracy": float(
                        contrast_probe["mean_selected_head_validation_accuracy"].min()
                    ),
                    "max_head_validation_accuracy": float(
                        contrast_probe["max_head_validation_accuracy"].max()
                    ),
                }
            )
        summary = summary.merge(
            pd.DataFrame(probe_rows),
            on="model_alias",
            how="left",
            validate="one_to_one",
        )
    summary["analysis_scope"] = np.where(
        summary["validation_competence_pass"],
        "primary_competent_model",
        "excluded_by_source_interface_gate",
    )
    return summary.sort_values("model_alias").reset_index(drop=True)


def filter_competent_models(
    frame: pd.DataFrame,
    competence: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty or competence.empty:
        return frame.iloc[0:0].copy()
    aliases = set(
        competence.loc[
            _bool_series(competence["validation_competence_pass"]), "model_alias"
        ].astype(str)
    )
    return frame[frame["model_alias"].astype(str).isin(aliases)].copy()


def build_model_interface_summary(
    effects: pd.DataFrame,
    competence: pd.DataFrame,
) -> pd.DataFrame:
    """Report equal-contrast adjusted effects without hiding model heterogeneity."""
    if effects.empty:
        return pd.DataFrame()
    contrast_means = (
        effects.groupby(
            ["model_alias", "model_name", "metric", "interface", "pair_type"],
            as_index=False,
        )
        .agg(
            mean_adjusted_gain=("adjusted_gain", "mean"),
            n_pairs=("pair_id", "nunique"),
        )
    )
    summary = (
        contrast_means.groupby(
            ["model_alias", "model_name", "metric", "interface"],
            as_index=False,
        )
        .agg(
            equal_contrast_mean_adjusted_gain=("mean_adjusted_gain", "mean"),
            min_contrast_adjusted_gain=("mean_adjusted_gain", "min"),
            max_contrast_adjusted_gain=("mean_adjusted_gain", "max"),
            positive_contrasts=("mean_adjusted_gain", lambda values: int((values > 0).sum())),
            negative_contrasts=("mean_adjusted_gain", lambda values: int((values < 0).sum())),
            n_contrasts=("pair_type", "nunique"),
            n_pairs=("n_pairs", "sum"),
        )
    )
    gate = competence[
        ["model_alias", "validation_competence_pass", "analysis_scope"]
    ].copy()
    return (
        summary.merge(gate, on="model_alias", how="left", validate="many_to_one")
        .sort_values(["metric", "interface", "model_alias"])
        .reset_index(drop=True)
    )


def selected_minus_wrong_pair_effects(pair_effects: pd.DataFrame) -> pd.DataFrame:
    """Build pair-level selected-minus-wrong effects after averaging templates."""
    if pair_effects.empty:
        return pd.DataFrame()
    averaged = (
        pair_effects.groupby(
            ["model_alias", "model_name", "pair_id", "pair_type", "mode", "interface"],
            as_index=False,
        )
        .agg(
            target_margin_gain=("target_margin_gain", "mean"),
            original_slot_margin_gain=("original_slot_margin_gain", "mean"),
        )
    )
    selected = averaged[averaged["mode"].eq("iti_probe_direction")].copy()
    wrong = averaged[averaged["mode"].eq("iti_wrong_direction")].copy()
    keys = ["model_alias", "model_name", "pair_id", "pair_type", "interface"]
    merged = selected.merge(
        wrong,
        on=keys,
        how="inner",
        suffixes=("_selected", "_wrong"),
        validate="one_to_one",
    )
    frames = []
    for metric, selected_column, wrong_column in [
        (
            "current_label_margin",
            "target_margin_gain_selected",
            "target_margin_gain_wrong",
        ),
        (
            "extraction_slot_margin",
            "original_slot_margin_gain_selected",
            "original_slot_margin_gain_wrong",
        ),
    ]:
        frame = merged[keys].copy()
        frame["metric"] = metric
        frame["target_gain"] = merged[selected_column]
        frame["wrong_gain"] = merged[wrong_column]
        frame["adjusted_gain"] = frame["target_gain"] - frame["wrong_gain"]
        frames.append(frame)
    current, extraction = frames
    difference = current.copy()
    difference["metric"] = "current_label_minus_extraction_slot"
    difference["target_gain"] = current["target_gain"] - extraction["target_gain"]
    difference["wrong_gain"] = current["wrong_gain"] - extraction["wrong_gain"]
    difference["adjusted_gain"] = (
        current["adjusted_gain"] - extraction["adjusted_gain"]
    )
    return pd.concat([current, extraction, difference], ignore_index=True, sort=False)


_CROSS_METHOD_METRICS = {
    "semantic_target_margin": "current_label_margin",
    "original_slot_margin": "extraction_slot_margin",
    "semantic_minus_original_slot": "current_label_minus_extraction_slot",
}


def build_cross_method_interface_summary(
    caa_bootstrap: pd.DataFrame,
    iti_all_models: pd.DataFrame,
    iti_competent_models: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize CAA and ITI bootstrap tables to one reviewer-facing schema."""
    frames = []
    if not caa_bootstrap.empty:
        caa = caa_bootstrap[
            caa_bootstrap["metric"].astype(str).isin(_CROSS_METHOD_METRICS)
        ].copy()
        caa["metric"] = caa["metric"].map(_CROSS_METHOD_METRICS)
        caa = caa.rename(
            columns={
                "mean_adjusted_ci_low": "ci_low",
                "mean_adjusted_ci_high": "ci_high",
            }
        )
        caa["method"] = "caa_residual_stream"
        caa["model_scope"] = "all_evaluated_models"
        caa["primary_scope"] = True
        frames.append(caa)
    for table, scope, primary in [
        (iti_all_models, "all_evaluated_models", False),
        (iti_competent_models, "source_interface_competent_models", True),
    ]:
        if table.empty:
            continue
        iti = table.copy()
        iti["method"] = "iti_head_probe"
        iti["model_scope"] = scope
        iti["primary_scope"] = primary
        frames.append(iti)
    if not frames:
        return pd.DataFrame()
    columns = [
        "method",
        "model_scope",
        "primary_scope",
        "metric",
        "interface",
        "n_models",
        "n_contrasts",
        "n_model_contrast_strata",
        "n_pairs",
        "mean_adjusted_gain",
        "ci_low",
        "ci_high",
        "positive_strata",
        "negative_strata",
        "confidence",
        "n_boot",
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in columns:
        if column not in combined.columns:
            combined[column] = np.nan
    return combined[columns].sort_values(
        ["method", "model_scope", "metric", "interface"]
    ).reset_index(drop=True)


def build_cross_method_decision(summary: pd.DataFrame) -> pd.DataFrame:
    """Condense each method's primary scope into transparent interface checks."""
    if summary.empty:
        return pd.DataFrame()
    primary = summary[summary["primary_scope"].fillna(False)].copy()
    rows = []
    for (method, scope), group in primary.groupby(["method", "model_scope"], sort=True):
        source = group[group["interface"].eq("letter_canonical")]
        non_source = group[~group["interface"].eq("letter_canonical")]
        current_source = source[source["metric"].eq("current_label_margin")]
        slot = non_source[non_source["metric"].eq("extraction_slot_margin")]
        gap = non_source[
            non_source["metric"].eq("current_label_minus_extraction_slot")
        ]
        rows.append(
            {
                "method": method,
                "model_scope": scope,
                "n_models": int(group["n_models"].max()),
                "n_non_source_interfaces": int(slot["interface"].nunique()),
                "source_interface_current_label_gain": (
                    float(current_source.iloc[0]["mean_adjusted_gain"])
                    if not current_source.empty
                    else np.nan
                ),
                "non_source_extraction_slot_mean_gain": float(
                    slot["mean_adjusted_gain"].mean()
                ),
                "non_source_extraction_slot_min_gain": float(
                    slot["mean_adjusted_gain"].min()
                ),
                "positive_extraction_slot_interfaces": int(
                    (slot["mean_adjusted_gain"] > 0).sum()
                ),
                "extraction_slot_dominant_interfaces": int(
                    (gap["mean_adjusted_gain"] < 0).sum()
                ),
                "all_non_source_extraction_slot_positive": bool(
                    len(slot) > 0 and (slot["mean_adjusted_gain"] > 0).all()
                ),
                "all_non_source_extraction_slot_dominant": bool(
                    len(gap) > 0 and (gap["mean_adjusted_gain"] < 0).all()
                ),
            }
        )
    return pd.DataFrame(rows)


def _read_completed_tables(output_dir: Path, filename: str) -> pd.DataFrame:
    parts = []
    for path in output_dir.glob(f"*/{filename}.csv"):
        marker = path.parent / "iti_probe_run_complete.csv"
        if marker.exists():
            parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def aggregate_iti_probe_audit(config: ITIProbeAuditConfig) -> dict[str, pd.DataFrame]:
    output_dir = config.cross_interface.audit.output_dir
    tables = {
        "iti_probe_multimodel_completion": _read_completed_tables(
            output_dir, "iti_probe_run_complete"
        ),
        "iti_probe_inventory": _read_completed_tables(output_dir, "iti_probe_inventory"),
        "iti_probe_validation_grid": _read_completed_tables(
            output_dir, "iti_probe_validation_grid"
        ),
        "iti_probe_selection": _read_completed_tables(output_dir, "iti_probe_selection"),
        "iti_probe_eval_rows": _read_completed_tables(output_dir, "iti_probe_eval_rows"),
    }
    rows = tables["iti_probe_eval_rows"]
    pair_effects = _pair_effects(rows)
    tables["iti_probe_pair_effects"] = pair_effects
    tables["iti_probe_mode_summary"] = _mode_summary(pair_effects)
    write_tables(tables, output_dir)
    return tables


def summarize_iti_probe_audit(config: ITIProbeAuditConfig) -> dict[str, pd.DataFrame]:
    tables = aggregate_iti_probe_audit(config)
    competence = build_iti_competence_summary(
        tables["iti_probe_selection"],
        tables["iti_probe_inventory"],
    )
    effects = adjusted_pair_effects(tables["iti_probe_pair_effects"])
    competent_effects = filter_competent_models(effects, competence)
    bootstrap_all = bootstrap_adjusted_effects(
        effects,
        n_boot=config.spec.n_boot,
        confidence=config.spec.confidence,
        seed=config.spec.seed + 901,
    )
    bootstrap_competent = bootstrap_adjusted_effects(
        competent_effects,
        n_boot=config.spec.n_boot,
        confidence=config.spec.confidence,
        seed=config.spec.seed + 902,
    )
    wrong_effects = selected_minus_wrong_pair_effects(tables["iti_probe_pair_effects"])
    competent_wrong_effects = filter_competent_models(wrong_effects, competence)
    wrong_bootstrap_all = bootstrap_adjusted_effects(
        wrong_effects,
        n_boot=config.spec.n_boot,
        confidence=config.spec.confidence,
        seed=config.spec.seed + 903,
    )
    wrong_bootstrap_competent = bootstrap_adjusted_effects(
        competent_wrong_effects,
        n_boot=config.spec.n_boot,
        confidence=config.spec.confidence,
        seed=config.spec.seed + 904,
    )
    caa_bootstrap = pd.DataFrame()
    if config.spec.comparison_caa_bootstrap_path is not None:
        if not config.spec.comparison_caa_bootstrap_path.exists():
            raise FileNotFoundError(
                "Configured CAA comparison bootstrap does not exist: "
                f"{config.spec.comparison_caa_bootstrap_path}"
            )
        caa_bootstrap = pd.read_csv(config.spec.comparison_caa_bootstrap_path)
    cross_method = build_cross_method_interface_summary(
        caa_bootstrap,
        bootstrap_all,
        bootstrap_competent,
    )
    decision = build_cross_method_decision(cross_method)
    primary_decision = bootstrap_competent.copy()
    primary_decision.insert(0, "model_scope", "source_interface_competent_models")
    statistics = {
        "iti_probe_competence_by_model": competence,
        "iti_probe_model_interface_summary": build_model_interface_summary(
            effects,
            competence,
        ),
        "iti_probe_adjusted_pair_effects": effects,
        "iti_probe_adjusted_pair_effects_competent_models": competent_effects,
        "iti_probe_bootstrap_ci": bootstrap_all,
        "iti_probe_bootstrap_ci_all_models": bootstrap_all,
        "iti_probe_bootstrap_ci_competent_models": bootstrap_competent,
        "iti_probe_selected_minus_wrong_pair_effects": wrong_effects,
        "iti_probe_selected_minus_wrong_bootstrap_ci_all_models": wrong_bootstrap_all,
        "iti_probe_selected_minus_wrong_bootstrap_ci_competent_models": (
            wrong_bootstrap_competent
        ),
        "iti_probe_cross_method_interface_summary": cross_method,
        "iti_probe_cross_method_decision": decision,
        "iti_probe_decision": primary_decision,
    }
    write_tables(statistics, config.cross_interface.audit.output_dir / "statistics")
    tables.update(statistics)
    return tables


def _run_model(
    model: Any,
    tokenizer: Any,
    model_config: Any,
    config: ITIProbeAuditConfig,
) -> dict[str, pd.DataFrame]:
    cross = config.cross_interface
    dataset = cross.audit.dataset
    model_dir = cross.audit.output_dir / model_config.model.alias
    model_dir.mkdir(parents=True, exist_ok=True)
    base_items = load_base_items(dataset, seed=cross.audit.runtime.seed)
    canonical = _canonical_mapping(cross)
    canonical_items = build_mapping_items(base_items, dataset, canonical, canonical)
    probe_items = _probe_items(canonical_items, config)
    candidate_layers = resolve_candidate_layers(
        len(decoder_layers(model)), config.spec.candidate_layer_fractions
    )
    print(
        f"[ITI:{model_config.model.alias}] collect train/validation head activations "
        f"for layers {candidate_layers}",
        flush=True,
    )
    activations = collect_attention_head_activations(
        model,
        tokenizer,
        probe_items["prompt"].tolist(),
        layer_indices=candidate_layers,
        batch_size=cross.audit.runtime.batch_size,
        max_length=cross.audit.runtime.max_length,
    )
    print(f"[ITI:{model_config.model.alias}] fit and rank per-head probes", flush=True)
    ranked_banks, inventory = build_head_probe_bank(probe_items, activations, config)
    print(
        f"[ITI:{model_config.model.alias}] select top-k heads and alpha on validation",
        flush=True,
    )
    validation_grid, selection, selected_banks = _validation_grid(
        model,
        tokenizer,
        base_items,
        ranked_banks,
        config,
    )
    selected_alpha = float(selection.iloc[0]["alpha"])
    print(
        f"[ITI:{model_config.model.alias}] selected "
        f"k={int(selection.iloc[0]['top_k_heads'])}, alpha={selected_alpha:g}; "
        "evaluate the frozen intervention on all six mappings",
        flush=True,
    )
    banks_by_mode = {
        "iti_probe_direction": selected_banks,
        "iti_wrong_direction": selected_banks,
    }
    mode_multipliers = {
        "iti_probe_direction": 1.0,
        "iti_wrong_direction": -1.0,
    }
    for random_seed in config.spec.random_control_seeds:
        mode = f"iti_random_direction_seed_{random_seed}"
        banks_by_mode[mode] = randomize_head_bank(selected_banks, seed=random_seed)
        mode_multipliers[mode] = 1.0
    test_items = _build_eval_items(base_items, config, split=dataset.test_split)
    eval_rows = _evaluate_banks(
        model,
        tokenizer,
        base_items,
        test_items,
        config,
        banks_by_mode=banks_by_mode,
        mode_multipliers=mode_multipliers,
        alpha=selected_alpha,
        interfaces=cross.interfaces,
        templates=cross.templates,
    )
    for frame in [inventory, validation_grid, selection, eval_rows]:
        frame.insert(0, "model_alias", model_config.model.alias)
        frame.insert(1, "model_name", model_config.model.name)
    completion = pd.DataFrame(
        [
            {
                "model_alias": model_config.model.alias,
                "model_name": model_config.model.name,
                "status": "complete",
                "candidate_layers": ",".join(str(value) for value in candidate_layers),
                "selected_top_k": int(selection.iloc[0]["top_k_heads"]),
                "selected_alpha": selected_alpha,
                "validation_competence_pass": bool(
                    selection.iloc[0]["validation_competence_pass"]
                ),
                "n_random_controls": len(config.spec.random_control_seeds),
                "direction_method": config.spec.direction_method,
                "seed": config.spec.seed,
            }
        ]
    )
    tables = {
        "iti_probe_inventory": inventory,
        "iti_probe_validation_grid": validation_grid,
        "iti_probe_selection": selection,
        "iti_probe_eval_rows": eval_rows,
        "iti_probe_run_complete": completion,
    }
    write_tables(tables, model_dir)
    print(f"[ITI:{model_config.model.alias}] complete: {model_dir}", flush=True)
    return tables


def run_iti_probe_audit(
    config: ITIProbeAuditConfig,
    *,
    model_aliases: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    cross = config.cross_interface
    requested = set(
        model_aliases or [model.model.alias for model in cross.audit.models if model.model.enabled]
    )
    known = {model.model.alias for model in cross.audit.models}
    unknown = sorted(requested - known)
    if unknown:
        raise KeyError(f"Unknown ITI model aliases: {unknown}")
    input_items = load_base_items(
        cross.audit.dataset,
        seed=cross.audit.runtime.seed,
    )
    input_inventory = build_iti_input_inventory(input_items, config)
    write_tables(
        {"iti_probe_input_inventory": input_inventory},
        cross.audit.output_dir,
    )
    print(
        "[ITI] input inventory verified:\n"
        + input_inventory.to_string(index=False),
        flush=True,
    )
    errors = []
    for model_config in cross.audit.models:
        if not model_config.model.enabled or model_config.model.alias not in requested:
            continue
        model_dir = cross.audit.output_dir / model_config.model.alias
        if (
            (model_dir / "iti_probe_run_complete.csv").exists()
            and not cross.audit.runtime.force_rerun
        ):
            continue
        tokenizer = model = None
        try:
            print(f"[ITI:{model_config.model.alias}] load model", flush=True)
            tokenizer, model = load_tokenizer_and_model(
                model_config.model.load_source,
                device_map=model_config.model.device_map,
                torch_dtype=model_config.model.torch_dtype,
            )
            _run_model(model, tokenizer, model_config, config)
            (model_dir / "iti_probe_run_error.csv").unlink(missing_ok=True)
        except Exception as exc:
            error = {
                "model_alias": model_config.model.alias,
                "model_name": model_config.model.name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            model_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([error]).to_csv(model_dir / "iti_probe_run_error.csv", index=False)
            if not cross.audit.runtime.continue_on_error:
                raise
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            cleanup_model()
    tables = aggregate_iti_probe_audit(config)
    if errors:
        pd.DataFrame(errors).to_csv(
            cross.audit.output_dir / "iti_probe_errors.csv", index=False
        )
    else:
        (cross.audit.output_dir / "iti_probe_errors.csv").unlink(missing_ok=True)
    return tables


def run_iti_probe_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    config = ITIProbeAuditConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    if phase == "run":
        return run_iti_probe_audit(config, model_aliases=model_aliases)
    if phase == "aggregate":
        return aggregate_iti_probe_audit(config)
    if phase == "summarize":
        return summarize_iti_probe_audit(config)
    if phase == "all":
        run_iti_probe_audit(config, model_aliases=model_aliases)
        return summarize_iti_probe_audit(config)
    raise ValueError("ITI phase must be run, aggregate, summarize, or all")
