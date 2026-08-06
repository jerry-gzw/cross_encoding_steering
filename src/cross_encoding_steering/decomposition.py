from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import unit_vector


@dataclass
class DirectionBank:
    vectors: dict[tuple[str, str, int, float], np.ndarray]
    inventory: pd.DataFrame


def contrast_directions(
    pairs: pd.DataFrame,
    activations: np.ndarray,
    *,
    item_id_column: str = "item_index",
    contrast_column: str = "pair_type",
    negative_column: str = "negative_item_index",
    positive_column: str = "positive_item_index",
    split_column: str = "split",
    train_split: str = "train",
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    train_pairs = pairs[pairs[split_column].astype(str).eq(train_split)].copy()
    directions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for contrast, group in train_pairs.groupby(contrast_column, sort=True):
        deltas = []
        for _, row in group.iterrows():
            negative = int(row[negative_column])
            positive = int(row[positive_column])
            deltas.append(np.asarray(activations[positive] - activations[negative], dtype=np.float32))
        if not deltas:
            continue
        matrix = np.stack(deltas, axis=0)
        pair_norms = np.linalg.norm(matrix, axis=1)
        median_pair_norm = float(np.median(pair_norms))
        raw = unit_vector(matrix.mean(axis=0)) * median_pair_norm
        if float(np.linalg.norm(raw)) < 1e-8:
            continue
        directions[str(contrast)] = raw
        rows.append(
            {
                "pair_type": str(contrast),
                "n_pairs": int(len(group)),
                "hidden_dim": int(raw.shape[0]),
                "mean_pair_delta_l2": float(pair_norms.mean()),
                "median_pair_delta_l2": median_pair_norm,
                "raw_direction_l2": float(np.linalg.norm(raw)),
            }
        )
    return directions, pd.DataFrame(rows)


def shared_basis(raw_directions: dict[str, np.ndarray], subspace_dim: int) -> tuple[np.ndarray, float]:
    vectors = [unit_vector(vector) for _, vector in sorted(raw_directions.items())]
    vectors = [vector for vector in vectors if float(np.linalg.norm(vector)) > 1e-8]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), float("nan")
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    if subspace_dim <= 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.float32), 0.0
    n_components = min(int(subspace_dim), matrix.shape[0], matrix.shape[1])
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    basis = vt[:n_components].astype(np.float32)
    total = float(np.square(singular_values).sum())
    used = float(np.square(singular_values[:n_components]).sum())
    return basis, used / total if total > 1e-12 else float("nan")


def project_onto_basis(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if basis.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (basis.T @ (basis @ arr)).astype(np.float32)


def decompose_directions(
    raw_directions: dict[str, np.ndarray],
    *,
    subspace_dims: list[int],
    mixing_grid: list[float],
    seed: int = 13,
    include_loco: bool = False,
    loco_subspace_dim: int = 1,
) -> DirectionBank:
    rng = np.random.default_rng(seed)
    vectors: dict[tuple[str, str, int, float], np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for subspace_dim in subspace_dims:
        basis, explained = shared_basis(raw_directions, int(subspace_dim))
        for pair_type, raw in sorted(raw_directions.items()):
            raw_norm = float(np.linalg.norm(raw))
            shared = project_onto_basis(raw, basis)
            residual = raw - shared
            components = {
                "raw_caa_direction": unit_vector(raw) * raw_norm,
                "shared_only_direction": unit_vector(shared) * raw_norm,
                "residual_only_direction": unit_vector(residual) * raw_norm,
                "wrong_direction_control": -unit_vector(raw) * raw_norm,
                "random_direction_control": unit_vector(rng.normal(size=raw.shape).astype(np.float32)) * raw_norm,
            }
            for mode, vector in components.items():
                if float(np.linalg.norm(vector)) < 1e-8:
                    continue
                key = (pair_type, mode, int(subspace_dim), 1.0)
                vectors[key] = vector.astype(np.float32)
                rows.append(_inventory_row(pair_type, mode, subspace_dim, 1.0, vector, shared, residual, explained))

            for mix_lambda in mixing_grid:
                mix = unit_vector(shared + float(mix_lambda) * residual) * raw_norm
                if float(np.linalg.norm(mix)) < 1e-8:
                    continue
                key = (pair_type, "mixed_shared_residual", int(subspace_dim), float(mix_lambda))
                vectors[key] = mix.astype(np.float32)
                rows.append(
                    _inventory_row(
                        pair_type,
                        "mixed_shared_residual",
                        subspace_dim,
                        float(mix_lambda),
                        mix,
                        shared,
                        residual,
                        explained,
                    )
                )
    if include_loco:
        loco_vectors, loco_rows = leave_one_contrast_out_decomposition(
            raw_directions,
            subspace_dim=loco_subspace_dim,
        )
        vectors.update(loco_vectors)
        rows.extend(loco_rows)
    return DirectionBank(vectors=vectors, inventory=pd.DataFrame(rows))


def leave_one_contrast_out_decomposition(
    raw_directions: dict[str, np.ndarray],
    *,
    subspace_dim: int = 1,
) -> tuple[dict[tuple[str, str, int, float], np.ndarray], list[dict[str, Any]]]:
    """Project each contrast onto a basis estimated from other contrasts only."""
    if int(subspace_dim) != 1:
        raise ValueError(
            "LOCO decomposition currently requires subspace_dim=1; with only "
            "two source contrasts, higher ranks become a span-reconstruction test"
        )
    if len(raw_directions) < 3:
        raise ValueError("LOCO decomposition requires at least three contrasts")

    vectors: dict[tuple[str, str, int, float], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for pair_type, raw in sorted(raw_directions.items()):
        source = {
            name: vector
            for name, vector in raw_directions.items()
            if name != pair_type
        }
        basis, explained = shared_basis(source, 1)
        shared = project_onto_basis(raw, basis)
        residual = np.asarray(raw, dtype=np.float32) - shared
        raw_norm = float(np.linalg.norm(raw))
        shared_norm = float(np.linalg.norm(shared))
        residual_norm = float(np.linalg.norm(residual))
        projection_ratio = shared_norm / raw_norm if raw_norm > 1e-8 else np.nan
        projection_energy_ratio = (
            (shared_norm * shared_norm) / (raw_norm * raw_norm)
            if raw_norm > 1e-8
            else np.nan
        )
        source_names = ",".join(sorted(source))
        for mode, component in [
            ("loco_shared_direction", shared),
            ("loco_residual_direction", residual),
        ]:
            if float(np.linalg.norm(component)) <= 1e-8:
                continue
            vector = unit_vector(component) * raw_norm
            vectors[(pair_type, mode, 1, 1.0)] = vector.astype(np.float32)
            rows.append(
                {
                    "pair_type": pair_type,
                    "mode": mode,
                    "subspace_dim": 1,
                    "mix_lambda": 1.0,
                    "direction_l2": float(np.linalg.norm(vector)),
                    "shared_component_l2": shared_norm,
                    "residual_component_l2": residual_norm,
                    "shared_explained_variance": float(explained),
                    "basis_protocol": "leave_one_contrast_out",
                    "basis_source_contrasts": source_names,
                    "n_basis_contrasts": len(source),
                    "self_included_in_basis": False,
                    "held_out_raw_l2": raw_norm,
                    "unnormalized_projection_l2": shared_norm,
                    "unnormalized_residual_l2": residual_norm,
                    "projection_ratio": projection_ratio,
                    "projection_energy_ratio": projection_energy_ratio,
                    "steering_norm_matched": True,
                }
            )
    return vectors, rows


def _inventory_row(
    pair_type: str,
    mode: str,
    subspace_dim: int,
    mix_lambda: float,
    vector: np.ndarray,
    shared: np.ndarray,
    residual: np.ndarray,
    explained: float,
) -> dict[str, Any]:
    return {
        "pair_type": pair_type,
        "mode": mode,
        "subspace_dim": int(subspace_dim),
        "mix_lambda": float(mix_lambda),
        "direction_l2": float(np.linalg.norm(vector)),
        "shared_component_l2": float(np.linalg.norm(shared)),
        "residual_component_l2": float(np.linalg.norm(residual)),
        "shared_explained_variance": float(explained),
        "basis_protocol": "in_sample",
        "basis_source_contrasts": "__all__",
        "n_basis_contrasts": np.nan,
        "self_included_in_basis": True,
        "held_out_raw_l2": np.nan,
        "unnormalized_projection_l2": float(np.linalg.norm(shared)),
        "unnormalized_residual_l2": float(np.linalg.norm(residual)),
        "projection_ratio": np.nan,
        "projection_energy_ratio": np.nan,
        "steering_norm_matched": True,
    }
