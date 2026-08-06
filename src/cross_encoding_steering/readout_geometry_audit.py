from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .io import write_tables
from .mapping_audit import (
    FixedDirectionMappingAuditConfig,
    MappingAuditModelConfig,
    _load_mapping_audit_config,
    annotate_model_frame,
    build_canonical_direction_bank,
    build_mapping_eval_items,
    build_mapping_items,
    build_mapping_mode_ci,
    build_pair_effects,
    evaluate_fixed_directions,
    load_base_items,
)
from .steering import (
    cleanup_model,
    collect_choice_probs,
    collect_choice_probs_and_activations,
    decoder_layers,
    load_tokenizer_and_model,
    patched_choice_probs,
    resolve_layer_index,
)


@dataclass(frozen=True)
class ReadoutGeometrySettings:
    subspace_rank: int = 0
    rank_candidates: tuple[int, ...] = (2, 4, 8, 16)
    validation_explained_energy_threshold: float = 0.9
    max_train_prompts_per_contrast: int = 64
    max_validation_prompts_per_contrast: int = 32
    max_control_test_pairs_per_contrast: int = 64
    gradient_batch_size: int = 1
    normalize_gradients: bool = True
    random_control_seeds: tuple[int, ...] = (13, 29, 47, 71, 101, 131, 173, 211, 257, 307)
    fidelity_alphas: tuple[float, ...] = (0.2, 0.4, 0.8)


@dataclass(frozen=True)
class ReadoutGeometryAuditConfig:
    mapping_audit: FixedDirectionMappingAuditConfig
    settings: ReadoutGeometrySettings

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "ReadoutGeometryAuditConfig":
        config_path = Path(path).expanduser().resolve()
        data = _load_mapping_audit_config(config_path)
        settings_data = dict(data.get("readout_geometry", {}))
        seeds = tuple(int(value) for value in settings_data.pop("random_control_seeds", [13, 29, 47]))
        rank_candidates = tuple(
            int(value) for value in settings_data.pop("rank_candidates", [2, 4, 8, 16])
        )
        fidelity_alphas = tuple(
            float(value) for value in settings_data.pop("fidelity_alphas", [0.2, 0.4, 0.8])
        )
        settings = ReadoutGeometrySettings(
            random_control_seeds=seeds,
            rank_candidates=rank_candidates,
            fidelity_alphas=fidelity_alphas,
            **settings_data,
        )
        if settings.subspace_rank < 0:
            raise ValueError("readout_geometry.subspace_rank must be zero or positive")
        if not settings.rank_candidates or min(settings.rank_candidates) <= 0:
            raise ValueError("rank_candidates must contain positive integers")
        if settings.max_train_prompts_per_contrast <= 0:
            raise ValueError("max_train_prompts_per_contrast must be positive")
        if settings.max_validation_prompts_per_contrast <= 0:
            raise ValueError("max_validation_prompts_per_contrast must be positive")
        if settings.max_control_test_pairs_per_contrast <= 0:
            raise ValueError("max_control_test_pairs_per_contrast must be positive")
        if settings.gradient_batch_size <= 0:
            raise ValueError("gradient_batch_size must be positive")
        if not settings.random_control_seeds:
            raise ValueError("At least one random-control seed is required")
        if not settings.fidelity_alphas or min(settings.fidelity_alphas) <= 0:
            raise ValueError("fidelity_alphas must contain positive values")
        return cls(
            mapping_audit=FixedDirectionMappingAuditConfig.from_json(
                config_path,
                project_root=project_root,
                model_source_overrides=model_source_overrides,
            ),
            settings=settings,
        )

    @property
    def output_dir(self) -> Path:
        return self.mapping_audit.output_dir


def _fingerprint(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()[:16]


def _strict_identifier_token_ids(tokenizer: Any, identifiers: tuple[str, ...]) -> list[int]:
    token_ids = []
    for identifier in identifiers:
        candidates = (f" {identifier}", identifier, f"\n{identifier}")
        matches = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in candidates
        ]
        singletons = [tokens for tokens in matches if len(tokens) == 1]
        if not singletons:
            raise ValueError(
                f"Identifier {identifier!r} is not a single token under any supported prefix"
            )
        token_ids.append(int(singletons[0][0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"Identifier token IDs are not unique: {token_ids}")
    return token_ids


def _last_token_indices(attention_mask: Any) -> Any:
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    return positions.expand_as(attention_mask).masked_fill(attention_mask.eq(0), -1).max(dim=1).values


def _model_input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def sample_gradient_prompts(
    canonical_items: pd.DataFrame,
    *,
    split_column: str,
    train_split: str,
    contrast_column: str,
    max_per_contrast: int,
    seed: int,
) -> pd.DataFrame:
    train = canonical_items[
        canonical_items[split_column].astype(str).eq(train_split)
    ].copy()
    sampled = []
    for index, (_, group) in enumerate(train.groupby(contrast_column, sort=True)):
        n = min(max_per_contrast, len(group))
        sampled.append(group.sample(n=n, random_state=seed + index))
    if not sampled:
        raise ValueError("No training prompts are available for readout-gradient estimation")
    return pd.concat(sampled, ignore_index=True, sort=False)


def sample_control_eval_items(
    eval_items: pd.DataFrame,
    *,
    max_pairs_per_contrast: int,
    seed: int,
) -> pd.DataFrame:
    selected_pair_ids = []
    for index, (pair_type, group) in enumerate(
        eval_items[["pair_type", "pair_id"]].drop_duplicates().groupby(
            "pair_type", sort=True
        )
    ):
        n = min(max_pairs_per_contrast, len(group))
        sampled = group.sample(n=n, random_state=seed + index).copy()
        selected_pair_ids.append(sampled)
    selected = pd.concat(selected_pair_ids, ignore_index=True, sort=False)
    return eval_items.merge(
        selected.assign(_selected_control=True),
        on=["pair_type", "pair_id"],
        how="inner",
        validate="many_to_one",
    ).drop(columns=["_selected_control"])


def collect_local_identifier_gradients(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    layer_index: int,
    identifiers: tuple[str, ...],
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Collect local gradients of pairwise identifier-logit margins at one layer."""
    import torch

    layers = decoder_layers(model)
    resolved_layer = resolve_layer_index(layer_index, len(layers))
    layer = layers[resolved_layer]
    token_ids = _strict_identifier_token_ids(tokenizer, identifiers)
    identifier_pairs = [
        (left_index, right_index)
        for left_index in range(len(identifiers))
        for right_index in range(left_index + 1, len(identifiers))
    ]
    input_device = _model_input_device(model)
    gradients: list[np.ndarray] = []
    inventory_rows = []

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        leaf = hidden.detach().requires_grad_(True)
        captured["leaf"] = leaf
        if isinstance(output, tuple):
            return (leaf, *output[1:])
        return leaf

    handle = layer.register_forward_hook(hook)
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            captured.clear()
            with torch.enable_grad():
                outputs = model(**encoded, use_cache=False)
                leaf = captured.get("leaf")
                if leaf is None:
                    raise RuntimeError("Decoder-layer hook did not capture a differentiable activation")
                positions = _last_token_indices(encoded["attention_mask"]).to(outputs.logits.device)
                batch_indices = torch.arange(len(batch_prompts), device=outputs.logits.device)
                final_logits = outputs.logits[batch_indices, positions, :]
                for pair_offset, (left_index, right_index) in enumerate(identifier_pairs):
                    score = (
                        final_logits[:, token_ids[left_index]]
                        - final_logits[:, token_ids[right_index]]
                    ).sum()
                    grad = torch.autograd.grad(
                        score,
                        leaf,
                        retain_graph=pair_offset < len(identifier_pairs) - 1,
                        create_graph=False,
                    )[0]
                    leaf_positions = positions.to(grad.device)
                    leaf_indices = torch.arange(len(batch_prompts), device=grad.device)
                    selected = grad[leaf_indices, leaf_positions, :].detach().float().cpu().numpy()
                    gradients.extend(selected)
                    for row_offset in range(len(batch_prompts)):
                        inventory_rows.append(
                            {
                                "prompt_index": start + row_offset,
                                "left_identifier": identifiers[left_index],
                                "right_identifier": identifiers[right_index],
                                "left_token_id": token_ids[left_index],
                                "right_token_id": token_ids[right_index],
                                "gradient_l2": float(np.linalg.norm(selected[row_offset])),
                                "layer_index": resolved_layer,
                            }
                        )
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    matrix = np.asarray(gradients, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("Readout-gradient collection produced no vectors")
    return matrix, pd.DataFrame(inventory_rows)


def estimate_readout_subspace(
    gradients: np.ndarray,
    *,
    rank: int,
    normalize_gradients: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    matrix = np.asarray(gradients, dtype=np.float64)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    keep = row_norms[:, 0] > 0
    matrix = matrix[keep]
    row_norms = row_norms[keep]
    if normalize_gradients:
        matrix = matrix / row_norms
    if len(matrix) < rank:
        raise ValueError(f"Readout rank {rank} exceeds {len(matrix)} nonzero gradients")
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    resolved_rank = min(rank, vh.shape[0])
    basis = vh[:resolved_rank].T.astype(np.float32)
    energy = np.square(singular_values)
    total = float(energy.sum())
    inventory = pd.DataFrame(
        [
            {
                "component": component + 1,
                "singular_value": float(singular_values[component]),
                "explained_energy": float(energy[component] / total) if total else np.nan,
                "cumulative_explained_energy": (
                    float(energy[: component + 1].sum() / total) if total else np.nan
                ),
                "selected": component < resolved_rank,
                "n_gradient_vectors": int(len(matrix)),
                "normalize_gradients": bool(normalize_gradients),
            }
            for component in range(len(singular_values))
        ]
    )
    return basis, inventory


def select_readout_rank(
    train_gradients: np.ndarray,
    validation_gradients: np.ndarray,
    *,
    rank_candidates: tuple[int, ...],
    explained_energy_threshold: float,
    normalize_gradients: bool,
    fixed_rank: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    max_rank = max(max(rank_candidates), fixed_rank, 2)
    max_basis, _ = estimate_readout_subspace(
        train_gradients,
        rank=max_rank,
        normalize_gradients=normalize_gradients,
    )
    validation = np.asarray(validation_gradients, dtype=np.float64)
    norms = np.linalg.norm(validation, axis=1, keepdims=True)
    validation = validation[norms[:, 0] > 0]
    norms = norms[norms[:, 0] > 0]
    if normalize_gradients:
        validation = validation / norms
    total_energy = np.square(validation).sum(axis=1)
    rows = []
    valid_candidates = sorted(
        {rank for rank in rank_candidates if rank <= max_basis.shape[1]}
    )
    for rank in valid_candidates:
        basis = max_basis[:, :rank].astype(np.float64)
        projected = validation @ basis
        explained = np.divide(
            np.square(projected).sum(axis=1),
            total_energy,
            out=np.zeros_like(total_energy),
            where=total_energy > 0,
        )
        rows.append(
            {
                "rank": rank,
                "validation_explained_energy_mean": float(explained.mean()),
                "validation_explained_energy_median": float(np.median(explained)),
                "meets_threshold": bool(explained.mean() >= explained_energy_threshold),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No candidate readout ranks fit the estimated train basis")
    if fixed_rank > 0:
        selected_rank = fixed_rank
    else:
        passing = table[table["meets_threshold"]]
        selected_rank = int(
            passing["rank"].min() if not passing.empty else table["rank"].max()
        )
    table["selected_rank"] = selected_rank
    strict_rank = min(2, max_basis.shape[1])
    return (
        max_basis[:, :selected_rank].astype(np.float32),
        max_basis[:, :strict_rank].astype(np.float32),
        table,
    )


def covariance_matched_vector(
    centered_activations: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    matrix = np.asarray(centered_activations, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("At least two centered activations are required")
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=len(matrix)) / np.sqrt(max(len(matrix) - 1, 1))
    return np.asarray(coefficients @ matrix, dtype=np.float32)


def _rescale(vector: np.ndarray, target_norm: float, *, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError(f"Cannot norm-match zero {label} vector")
    return np.asarray(vector * (target_norm / norm), dtype=np.float32)


def decompose_against_subspace(
    vector: np.ndarray,
    basis: np.ndarray,
) -> dict[str, np.ndarray | float]:
    raw = np.asarray(vector, dtype=np.float64)
    orthonormal_basis = np.asarray(basis, dtype=np.float64)
    projection = orthonormal_basis @ (orthonormal_basis.T @ raw)
    residual = raw - projection
    raw_norm = float(np.linalg.norm(raw))
    projection_norm = float(np.linalg.norm(projection))
    residual_norm = float(np.linalg.norm(residual))
    if raw_norm == 0 or projection_norm == 0 or residual_norm == 0:
        raise ValueError("Raw, projected, and orthogonal components must all be nonzero")
    return {
        "raw": raw.astype(np.float32),
        "projection": projection.astype(np.float32),
        "residual": residual.astype(np.float32),
        "projection_norm_matched": (projection * raw_norm / projection_norm).astype(np.float32),
        "residual_norm_matched": (residual * raw_norm / residual_norm).astype(np.float32),
        "raw_l2": raw_norm,
        "projection_l2": projection_norm,
        "residual_l2": residual_norm,
        "projection_ratio": projection_norm / raw_norm,
        "residual_ratio": residual_norm / raw_norm,
        "projection_energy_fraction": (projection_norm / raw_norm) ** 2,
        "residual_energy_fraction": (residual_norm / raw_norm) ** 2,
        "reconstruction_error": float(np.linalg.norm(raw - projection - residual)),
        "projection_residual_cosine": float(
            np.dot(projection, residual) / (projection_norm * residual_norm)
        ),
    }


def build_readout_direction_bank(
    raw_directions: dict[tuple[str, str], np.ndarray],
    basis: np.ndarray,
    strict_rank_two_basis: np.ndarray,
    centered_activations: np.ndarray,
    settings: ReadoutGeometrySettings,
    *,
    extraction_mapping: str,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    directions: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for (pair_type, mode), raw in sorted(raw_directions.items()):
        if mode != "raw_canonical_direction":
            continue
        decomposition = decompose_against_subspace(raw, basis)
        mode_vectors = {
            "raw_canonical_direction": decomposition["raw"],
            "readout_projection_natural": decomposition["projection"],
            "readout_orthogonal_natural": decomposition["residual"],
            "readout_projection_norm_matched": decomposition["projection_norm_matched"],
            "readout_orthogonal_norm_matched": decomposition["residual_norm_matched"],
        }
        if basis.shape[1] != strict_rank_two_basis.shape[1]:
            strict = decompose_against_subspace(raw, strict_rank_two_basis)
            mode_vectors.update(
                {
                    "readout_projection_natural_rank2": strict["projection"],
                    "readout_orthogonal_natural_rank2": strict["residual"],
                    "readout_projection_norm_matched_rank2": strict[
                        "projection_norm_matched"
                    ],
                    "readout_orthogonal_norm_matched_rank2": strict[
                        "residual_norm_matched"
                    ],
                }
            )
        raw_norm = float(decomposition["raw_l2"])
        for control_seed in settings.random_control_seeds:
            pair_digest = hashlib.sha256(pair_type.encode("utf-8")).digest()
            pair_seed = int.from_bytes(pair_digest[:4], "little")
            random_vector = covariance_matched_vector(
                centered_activations,
                seed=control_seed + pair_seed,
            )
            random_projection = basis @ (basis.T @ random_vector)
            random_orthogonal = random_vector - random_projection
            mode_vectors[f"cov_global_seed_{control_seed}"] = _rescale(
                random_vector, raw_norm, label="global covariance"
            )
            mode_vectors[f"cov_readout_seed_{control_seed}"] = _rescale(
                random_projection,
                float(decomposition["projection_l2"]),
                label="readout-subspace covariance",
            )
            mode_vectors[f"cov_orthogonal_seed_{control_seed}"] = _rescale(
                random_orthogonal,
                float(decomposition["residual_l2"]),
                label="orthogonal covariance",
            )
        for output_mode, output_vector in mode_vectors.items():
            directions[(pair_type, output_mode)] = np.asarray(output_vector, dtype=np.float32)
            rows.append(
                {
                    "pair_type": pair_type,
                    "mode": output_mode,
                    "extraction_mapping": extraction_mapping,
                    "direction_l2": float(np.linalg.norm(output_vector)),
                    "direction_sha256": _fingerprint(np.asarray(output_vector)),
                    "raw_l2": decomposition["raw_l2"],
                    "readout_projection_l2": decomposition["projection_l2"],
                    "readout_orthogonal_l2": decomposition["residual_l2"],
                    "readout_projection_ratio": decomposition["projection_ratio"],
                    "readout_orthogonal_ratio": decomposition["residual_ratio"],
                    "readout_projection_energy_fraction": decomposition[
                        "projection_energy_fraction"
                    ],
                    "readout_orthogonal_energy_fraction": decomposition[
                        "residual_energy_fraction"
                    ],
                    "reconstruction_error": decomposition["reconstruction_error"],
                    "projection_residual_cosine": decomposition["projection_residual_cosine"],
                    "readout_subspace_rank": basis.shape[1],
                }
            )
    if not directions:
        raise ValueError("No raw canonical directions were available for readout decomposition")
    return directions, pd.DataFrame(rows)


def build_first_order_fidelity(
    model: Any,
    tokenizer: Any,
    validation_items: pd.DataFrame,
    validation_gradients: np.ndarray,
    validation_gradient_inventory: pd.DataFrame,
    raw_directions: dict[tuple[str, str], np.ndarray],
    *,
    contrast_column: str,
    layer_index: int,
    alphas: tuple[float, ...],
    batch_size: int,
    max_length: int,
    identifiers: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = collect_choice_probs(
        model,
        tokenizer,
        validation_items["prompt"].tolist(),
        batch_size=batch_size,
        max_length=max_length,
        choice_letters=list(identifiers),
    )
    identifier_index = {identifier: index for index, identifier in enumerate(identifiers)}
    rows = []
    for pair_type, item_indices in validation_items.groupby(
        contrast_column, sort=True
    ).groups.items():
        raw = raw_directions[(str(pair_type), "raw_canonical_direction")]
        selected_indices = np.asarray(list(item_indices), dtype=int)
        prompts = validation_items.iloc[selected_indices]["prompt"].tolist()
        base = baseline[selected_indices]
        gradient_rows = validation_gradient_inventory[
            validation_gradient_inventory["prompt_index"].isin(selected_indices)
        ].copy()
        for alpha in alphas:
            patched = patched_choice_probs(
                model,
                tokenizer,
                prompts,
                layer_index=layer_index,
                direction=np.asarray(raw, dtype=np.float32),
                alpha=float(alpha),
                batch_size=batch_size,
                max_length=max_length,
                choice_letters=list(identifiers),
            )
            local_lookup = {
                global_index: local_index
                for local_index, global_index in enumerate(selected_indices)
            }
            for gradient_index, gradient_row in gradient_rows.iterrows():
                prompt_index = int(gradient_row["prompt_index"])
                local_index = local_lookup[prompt_index]
                left = identifier_index[str(gradient_row["left_identifier"])]
                right = identifier_index[str(gradient_row["right_identifier"])]
                base_margin = float(
                    np.log(np.clip(base[local_index, left], 1e-12, 1.0))
                    - np.log(np.clip(base[local_index, right], 1e-12, 1.0))
                )
                patched_margin = float(
                    np.log(np.clip(patched[local_index, left], 1e-12, 1.0))
                    - np.log(np.clip(patched[local_index, right], 1e-12, 1.0))
                )
                predicted = float(
                    alpha
                    * np.dot(
                        validation_gradients[int(gradient_index)],
                        np.asarray(raw, dtype=np.float32),
                    )
                )
                rows.append(
                    {
                        "pair_type": str(pair_type),
                        "prompt_index": prompt_index,
                        "left_identifier": gradient_row["left_identifier"],
                        "right_identifier": gradient_row["right_identifier"],
                        "alpha": float(alpha),
                        "predicted_logit_margin_change": predicted,
                        "actual_logit_margin_change": patched_margin - base_margin,
                    }
                )
    fidelity_rows = pd.DataFrame(rows)
    summaries = []
    for (alpha, pair_type), group in fidelity_rows.groupby(
        ["alpha", "pair_type"], sort=True
    ):
        predicted = group["predicted_logit_margin_change"].to_numpy(float)
        actual = group["actual_logit_margin_change"].to_numpy(float)
        denominator = float(np.dot(predicted, predicted))
        slope = float(np.dot(predicted, actual) / denominator) if denominator else np.nan
        residual = actual - slope * predicted if np.isfinite(slope) else actual
        total = float(np.square(actual - actual.mean()).sum())
        summaries.append(
            {
                "alpha": alpha,
                "pair_type": pair_type,
                "n_observations": len(group),
                "pearson_r": float(np.corrcoef(predicted, actual)[0, 1]),
                "spearman_rho": float(
                    pd.Series(predicted).rank().corr(pd.Series(actual).rank())
                ),
                "slope_through_origin": slope,
                "r_squared": (
                    float(1.0 - np.square(residual).sum() / total)
                    if total > 0
                    else np.nan
                ),
                "mean_absolute_error": float(np.abs(actual - predicted).mean()),
            }
        )
    return fidelity_rows, pd.DataFrame(summaries)


def _summarize_model_effects(pair_effects: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "semantic_prob_gain",
        "original_letter_prob_gain",
        "semantic_minus_original_prob_gain_moved",
        "js_shift",
    ]
    return (
        pair_effects.groupby(["mapping_name", "mapping_family", "mode"], as_index=False)[metrics]
        .mean()
        .rename(columns={metric: f"mean_{metric}" for metric in metrics})
    )


def run_single_model_readout_geometry(
    model: Any,
    tokenizer: Any,
    config: ReadoutGeometryAuditConfig,
    model_config: MappingAuditModelConfig,
) -> dict[str, pd.DataFrame]:
    audit = config.mapping_audit
    alias = model_config.model.alias
    model_dir = audit.output_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    base_items = load_base_items(audit.dataset, seed=audit.runtime.seed)
    mapping_by_name = {mapping.name: mapping for mapping in audit.mappings}
    canonical_mapping = mapping_by_name[audit.dataset.canonical_mapping]
    canonical_items = build_mapping_items(
        base_items, audit.dataset, canonical_mapping, canonical_mapping
    )
    raw_directions, _ = build_canonical_direction_bank(
        model,
        tokenizer,
        canonical_items,
        audit.dataset,
        layer_index=model_config.locked_layer,
        subspace_dim=audit.subspace_dim,
        batch_size=audit.runtime.batch_size,
        max_length=audit.runtime.max_length,
        seed=audit.runtime.seed,
        extraction_position="pre_answer",
        enabled_modes=("raw_canonical_direction",),
    )
    train_gradient_items = sample_gradient_prompts(
        canonical_items,
        split_column=audit.dataset.split_column,
        train_split=audit.dataset.train_split,
        contrast_column=audit.dataset.contrast_column,
        max_per_contrast=config.settings.max_train_prompts_per_contrast,
        seed=audit.runtime.seed,
    )
    validation_gradient_items = sample_gradient_prompts(
        canonical_items,
        split_column=audit.dataset.split_column,
        train_split=audit.dataset.validation_split,
        contrast_column=audit.dataset.contrast_column,
        max_per_contrast=config.settings.max_validation_prompts_per_contrast,
        seed=audit.runtime.seed + 1,
    )
    identifiers = tuple(
        chr(ord("A") + index) for index in range(len(audit.dataset.label_ranks))
    )
    train_gradients, train_gradient_inventory = collect_local_identifier_gradients(
        model,
        tokenizer,
        train_gradient_items["prompt"].tolist(),
        layer_index=model_config.locked_layer,
        identifiers=identifiers,
        batch_size=config.settings.gradient_batch_size,
        max_length=audit.runtime.max_length,
    )
    validation_gradients, validation_gradient_inventory = (
        collect_local_identifier_gradients(
            model,
            tokenizer,
            validation_gradient_items["prompt"].tolist(),
            layer_index=model_config.locked_layer,
            identifiers=identifiers,
            batch_size=config.settings.gradient_batch_size,
            max_length=audit.runtime.max_length,
        )
    )
    basis, strict_rank_two_basis, rank_selection = select_readout_rank(
        train_gradients,
        validation_gradients,
        rank_candidates=config.settings.rank_candidates,
        explained_energy_threshold=(
            config.settings.validation_explained_energy_threshold
        ),
        normalize_gradients=config.settings.normalize_gradients,
        fixed_rank=config.settings.subspace_rank,
    )
    _, basis_inventory = estimate_readout_subspace(
        train_gradients,
        rank=max(config.settings.rank_candidates),
        normalize_gradients=config.settings.normalize_gradients,
    )
    train_gradient_inventory["gradient_split"] = "train"
    validation_gradient_inventory["gradient_split"] = "validation"
    gradient_inventory = pd.concat(
        [train_gradient_inventory, validation_gradient_inventory],
        ignore_index=True,
        sort=False,
    )
    resolved_layer = resolve_layer_index(
        model_config.locked_layer, len(decoder_layers(model))
    )
    _, train_activations = collect_choice_probs_and_activations(
        model,
        tokenizer,
        train_gradient_items["prompt"].tolist(),
        layer_indices=[resolved_layer],
        batch_size=audit.runtime.batch_size,
        max_length=audit.runtime.max_length,
        choice_letters=list(identifiers),
    )
    activation_matrix = np.asarray(train_activations[resolved_layer], dtype=np.float32)
    centered_activations = activation_matrix - activation_matrix.mean(axis=0, keepdims=True)
    directions, direction_inventory = build_readout_direction_bank(
        raw_directions,
        basis,
        strict_rank_two_basis,
        centered_activations,
        config.settings,
        extraction_mapping=audit.dataset.canonical_mapping,
    )
    direction_inventory["layer_index"] = resolved_layer
    fidelity_rows, fidelity_summary = build_first_order_fidelity(
        model,
        tokenizer,
        validation_gradient_items,
        validation_gradients,
        validation_gradient_inventory,
        raw_directions,
        contrast_column=audit.dataset.contrast_column,
        layer_index=model_config.locked_layer,
        alphas=config.settings.fidelity_alphas,
        batch_size=audit.runtime.batch_size,
        max_length=audit.runtime.max_length,
        identifiers=identifiers,
    )
    gradient_inventory["model_alias"] = alias
    basis_inventory["model_alias"] = alias
    np.savez_compressed(
        model_dir / "readout_geometry_arrays.npz",
        readout_basis=basis,
        strict_rank_two_basis=strict_rank_two_basis,
        train_local_gradients=train_gradients,
        validation_local_gradients=validation_gradients,
        **{
            f"{pair_type}__{mode}": vector
            for (pair_type, mode), vector in directions.items()
        },
    )

    all_eval_rows = []
    all_pair_effects = []
    all_ci = []
    for mapping in audit.mappings:
        mapping_items = build_mapping_items(
            base_items, audit.dataset, mapping, canonical_mapping
        )
        eval_items = build_mapping_eval_items(
            mapping_items, audit.dataset, mapping, canonical_mapping
        )
        main_directions = {
            key: value
            for key, value in directions.items()
            if not key[1].startswith("cov_")
        }
        main_modes = {mode for _, mode in main_directions}
        main_inventory = direction_inventory[
            direction_inventory["mode"].isin(main_modes)
        ].copy()
        eval_rows_main = evaluate_fixed_directions(
            model,
            tokenizer,
            eval_items,
            main_directions,
            main_inventory,
            layer_index=model_config.locked_layer,
            alpha=model_config.locked_alpha,
            n_choices=len(audit.dataset.label_ranks),
            batch_size=audit.runtime.batch_size,
            max_length=audit.runtime.max_length,
            injection_position="pre_answer",
        )
        control_directions = {
            key: value
            for key, value in directions.items()
            if key[1].startswith("cov_")
        }
        control_modes = {mode for _, mode in control_directions}
        control_inventory = direction_inventory[
            direction_inventory["mode"].isin(control_modes)
        ].copy()
        control_items = sample_control_eval_items(
            eval_items,
            max_pairs_per_contrast=(
                config.settings.max_control_test_pairs_per_contrast
            ),
            seed=audit.runtime.seed,
        )
        eval_rows_control = evaluate_fixed_directions(
            model,
            tokenizer,
            control_items,
            control_directions,
            control_inventory,
            layer_index=model_config.locked_layer,
            alpha=model_config.locked_alpha,
            n_choices=len(audit.dataset.label_ranks),
            batch_size=audit.runtime.batch_size,
            max_length=audit.runtime.max_length,
            injection_position="pre_answer",
        )
        eval_rows_main["evaluation_scope"] = "full_test"
        eval_rows_control["evaluation_scope"] = "control_pair_subset"
        eval_rows = pd.concat(
            [eval_rows_main, eval_rows_control],
            ignore_index=True,
            sort=False,
        )
        effects = build_pair_effects(eval_rows)
        all_eval_rows.append(eval_rows)
        all_pair_effects.append(effects)
        all_ci.append(
            build_mapping_mode_ci(
                effects,
                n_boot=audit.statistics.n_boot,
                confidence=audit.statistics.confidence,
                seed=audit.runtime.seed,
            )
        )
    pair_effects = pd.concat(all_pair_effects, ignore_index=True, sort=False)
    tables = {
        "readout_gradient_inventory": gradient_inventory,
        "readout_basis_inventory": basis_inventory,
        "readout_rank_selection": rank_selection,
        "readout_direction_inventory": direction_inventory,
        "readout_first_order_fidelity_rows": fidelity_rows,
        "readout_first_order_fidelity_summary": fidelity_summary,
        "readout_eval_rows": pd.concat(all_eval_rows, ignore_index=True, sort=False),
        "readout_pair_effects": pair_effects,
        "readout_mapping_mode_ci": pd.concat(all_ci, ignore_index=True, sort=False),
        "readout_model_summary": _summarize_model_effects(pair_effects),
    }
    tables = {
        name: annotate_model_frame(
            frame,
            model_alias=alias,
            model_name=model_config.model.name,
        )
        for name, frame in tables.items()
    }
    write_tables(tables, model_dir)
    pd.DataFrame(
        [
            {
                "model_alias": alias,
                "model_name": model_config.model.name,
                "status": "complete",
                "locked_layer": model_config.locked_layer,
                "locked_alpha": model_config.locked_alpha,
                "readout_subspace_rank": basis.shape[1],
                "n_train_gradient_prompts": len(train_gradient_items),
                "n_validation_gradient_prompts": len(validation_gradient_items),
                "n_train_gradient_vectors": len(train_gradients),
                "n_validation_gradient_vectors": len(validation_gradients),
                "n_eval_mappings": len(audit.mappings),
            }
        ]
    ).to_csv(model_dir / "readout_geometry_run_complete.csv", index=False)
    return tables


def readout_geometry_run_is_complete(model_dir: str | Path) -> bool:
    root = Path(model_dir)
    return all(
        (root / filename).exists()
        for filename in [
            "readout_geometry_run_complete.csv",
            "readout_direction_inventory.csv",
            "readout_pair_effects.csv",
            "readout_mapping_mode_ci.csv",
        ]
    )


def aggregate_readout_geometry(config: ReadoutGeometryAuditConfig) -> dict[str, pd.DataFrame]:
    audit = config.mapping_audit
    names = [
        "readout_geometry_run_complete",
        "readout_basis_inventory",
        "readout_rank_selection",
        "readout_direction_inventory",
        "readout_first_order_fidelity_summary",
        "readout_model_summary",
        "readout_mapping_mode_ci",
        "readout_pair_effects",
    ]
    collected: dict[str, list[pd.DataFrame]] = {}
    for model_config in audit.models:
        model_dir = audit.output_dir / model_config.model.alias
        for name in names:
            path = model_dir / f"{name}.csv"
            if path.exists():
                collected.setdefault(name, []).append(pd.read_csv(path))
    aggregated = {
        name: pd.concat(parts, ignore_index=True, sort=False)
        for name, parts in collected.items()
    }
    effects = aggregated.get("readout_pair_effects", pd.DataFrame())
    if effects.empty:
        global_summary = pd.DataFrame()
        control_ensemble = pd.DataFrame()
    else:
        metrics = [
            "semantic_prob_gain",
            "original_letter_prob_gain",
            "semantic_minus_original_prob_gain_moved",
            "js_shift",
        ]
        strata = (
            effects.groupby(
                ["model_alias", "pair_type", "mapping_name", "mapping_family", "mode"],
                as_index=False,
            )[metrics]
            .mean()
        )
        global_summary = (
            strata.groupby(["mapping_name", "mapping_family", "mode"], as_index=False)
            .agg(
                n_models=("model_alias", "nunique"),
                n_contrasts=("pair_type", "nunique"),
                mean_semantic_prob_gain=("semantic_prob_gain", "mean"),
                mean_original_letter_prob_gain=("original_letter_prob_gain", "mean"),
                mean_semantic_minus_original=(
                    "semantic_minus_original_prob_gain_moved",
                    "mean",
                ),
                mean_js_shift=("js_shift", "mean"),
            )
        )
        control_rows = effects[
            effects["mode"].astype(str).str.startswith("cov_")
        ].copy()
        if control_rows.empty:
            control_ensemble = pd.DataFrame()
        else:
            control_rows["control_family"] = control_rows["mode"].str.replace(
                r"_seed_\d+$", "", regex=True
            )
            seed_averaged = (
                control_rows.groupby(
                    [
                        "model_alias",
                        "pair_type",
                        "pair_id",
                        "mapping_name",
                        "mapping_family",
                        "control_family",
                    ],
                    as_index=False,
                )[metrics]
                .mean()
            )
            control_ensemble = (
                seed_averaged.groupby(
                    ["mapping_name", "mapping_family", "control_family"],
                    as_index=False,
                )
                .agg(
                    n_models=("model_alias", "nunique"),
                    n_contrasts=("pair_type", "nunique"),
                    mean_semantic_prob_gain=("semantic_prob_gain", "mean"),
                    mean_original_letter_prob_gain=("original_letter_prob_gain", "mean"),
                    mean_semantic_minus_original=(
                        "semantic_minus_original_prob_gain_moved",
                        "mean",
                    ),
                    mean_js_shift=("js_shift", "mean"),
                )
            )
    aggregated["readout_geometry_global_summary"] = global_summary
    aggregated["readout_geometry_control_ensemble_summary"] = control_ensemble
    fidelity = aggregated.get("readout_first_order_fidelity_summary", pd.DataFrame())
    if fidelity.empty:
        aggregated["readout_fidelity_global_summary"] = pd.DataFrame()
    else:
        aggregated["readout_fidelity_global_summary"] = (
            fidelity.groupby("alpha", as_index=False)
            .agg(
                n_models=("model_alias", "nunique"),
                n_contrasts=("pair_type", "nunique"),
                mean_pearson_r=("pearson_r", "mean"),
                mean_spearman_rho=("spearman_rho", "mean"),
                mean_slope=("slope_through_origin", "mean"),
                mean_r_squared=("r_squared", "mean"),
                mean_absolute_error=("mean_absolute_error", "mean"),
            )
        )
    write_tables(aggregated, audit.output_dir)
    return aggregated


def run_readout_geometry_audit(
    config: ReadoutGeometryAuditConfig,
    *,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
    model_loader: Callable[..., tuple[Any, Any]] = load_tokenizer_and_model,
    progress: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    if phase not in {"run", "aggregate", "all"}:
        raise ValueError("phase must be run, aggregate, or all")
    audit = config.mapping_audit
    audit.output_dir.mkdir(parents=True, exist_ok=True)
    if phase == "aggregate":
        return aggregate_readout_geometry(config)
    requested = set(
        model_aliases
        or [item.model.alias for item in audit.models if item.model.enabled]
    )
    errors = []
    for model_config in audit.models:
        public_model = model_config.model
        if not public_model.enabled or public_model.alias not in requested:
            continue
        model_dir = audit.output_dir / public_model.alias
        if readout_geometry_run_is_complete(model_dir) and not audit.runtime.force_rerun:
            progress(f"[{public_model.alias}] complete; skipping")
            continue
        tokenizer = None
        model = None
        try:
            progress(f"[{public_model.alias}] loading {public_model.load_source}")
            tokenizer, model = model_loader(
                public_model.load_source,
                device_map=public_model.device_map,
                torch_dtype=public_model.torch_dtype,
            )
            run_single_model_readout_geometry(
                model, tokenizer, config, model_config
            )
            (model_dir / "readout_geometry_run_error.csv").unlink(missing_ok=True)
            progress(f"[{public_model.alias}] complete")
        except Exception as exc:
            error = {
                "model_alias": public_model.alias,
                "model_name": public_model.name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            model_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([error]).to_csv(
                model_dir / "readout_geometry_run_error.csv", index=False
            )
            progress(f"[{public_model.alias}] failed: {type(exc).__name__}: {exc}")
            if not audit.runtime.continue_on_error:
                raise
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            cleanup_model()
    if errors:
        pd.DataFrame(errors).to_csv(
            audit.output_dir / "readout_geometry_errors.csv", index=False
        )
    else:
        (audit.output_dir / "readout_geometry_errors.csv").unlink(missing_ok=True)
    if phase == "run":
        completion_parts = []
        for model_config in audit.models:
            path = (
                audit.output_dir
                / model_config.model.alias
                / "readout_geometry_run_complete.csv"
            )
            if path.exists():
                completion_parts.append(pd.read_csv(path))
        return {
            "readout_geometry_run_complete": (
                pd.concat(completion_parts, ignore_index=True, sort=False)
                if completion_parts
                else pd.DataFrame()
            )
        }
    return aggregate_readout_geometry(config)


def run_readout_geometry_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
    phase: str = "all",
) -> dict[str, pd.DataFrame]:
    config = ReadoutGeometryAuditConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    return run_readout_geometry_audit(
        config,
        model_aliases=model_aliases,
        phase=phase,
    )
