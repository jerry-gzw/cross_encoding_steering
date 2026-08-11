from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .decomposition import contrast_directions, decompose_directions
from .io import write_tables
from .mapping_audit import (
    CONTROL_MODES,
    PAPER_MODES,
    FixedDirectionMappingAuditConfig,
    MappingAuditModelConfig,
    annotate_model_frame,
    build_canonical_train_index,
    build_contrast_summary,
    build_mapping_eval_items,
    build_mapping_items,
    build_mapping_mode_ci,
    build_mode_comparisons,
    build_pair_effects,
    evaluate_fixed_directions,
    load_base_items,
)
from .statistics import holm_adjust, paired_sign_flip_pvalue
from .steering import (
    cleanup_model,
    collect_choice_probs_and_position_activations,
    decoder_layers,
    load_tokenizer_and_model,
    resolve_layer_index,
)


SUPPORTED_EXTRACTION_POSITIONS = ("scenario_end", "pre_answer")
POSITION_COMPARISON_METRICS = (
    "semantic_accuracy_gain",
    "semantic_prob_gain",
    "original_letter_accuracy_gain",
    "original_letter_prob_gain",
    "semantic_minus_original_prob_gain_moved",
    "semantic_minus_original_accuracy_gain_moved",
    "js_shift",
)


@dataclass(frozen=True)
class ExtractionPositionAuditConfig:
    mapping_audit: FixedDirectionMappingAuditConfig
    extraction_positions: tuple[str, ...] = SUPPORTED_EXTRACTION_POSITIONS
    intervention_position: str = "pre_answer"
    transport_control_mappings: tuple[str, ...] = ()

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "ExtractionPositionAuditConfig":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        positions = tuple(
            str(value) for value in data.get("extraction_positions", SUPPORTED_EXTRACTION_POSITIONS)
        )
        unknown = sorted(set(positions) - set(SUPPORTED_EXTRACTION_POSITIONS))
        if unknown:
            raise ValueError(f"Unsupported extraction positions: {unknown}")
        if set(positions) != set(SUPPORTED_EXTRACTION_POSITIONS):
            raise ValueError(
                "The position audit requires both scenario_end and pre_answer for a paired comparison"
            )
        intervention = str(data.get("intervention_position", "pre_answer"))
        if intervention != "pre_answer":
            raise ValueError(
                "This audit holds intervention_position fixed at pre_answer to isolate extraction position"
            )
        return cls(
            mapping_audit=FixedDirectionMappingAuditConfig.from_json(
                config_path,
                project_root=project_root,
                model_source_overrides=model_source_overrides,
            ),
            extraction_positions=positions,
            intervention_position=intervention,
            transport_control_mappings=tuple(
                str(value) for value in data.get("transport_control_mappings", [])
            ),
        )

    @property
    def output_dir(self) -> Path:
        return self.mapping_audit.output_dir


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else np.nan


def build_position_direction_banks(
    model: Any,
    tokenizer: Any,
    canonical_items: pd.DataFrame,
    config: ExtractionPositionAuditConfig,
    model_config: MappingAuditModelConfig,
) -> tuple[dict[tuple[str, str, str], np.ndarray], pd.DataFrame, pd.DataFrame]:
    audit = config.mapping_audit
    dataset = audit.dataset
    resolved_layer = resolve_layer_index(model_config.locked_layer, len(decoder_layers(model)))
    train_items, train_pairs = build_canonical_train_index(canonical_items, dataset)
    position_offsets = {
        "scenario_end": train_items["scenario_char_end"].astype(int).tolist(),
        "pre_answer": [None] * len(train_items),
    }
    _, position_activations = collect_choice_probs_and_position_activations(
        model,
        tokenizer,
        train_items["prompt"].tolist(),
        position_character_offsets={name: position_offsets[name] for name in config.extraction_positions},
        layer_indices=[resolved_layer],
        batch_size=audit.runtime.batch_size,
        max_length=audit.runtime.max_length,
        choice_letters=[chr(ord("A") + index) for index in range(len(dataset.label_ranks))],
    )

    mode_map = {
        "raw_caa_direction": "raw_canonical_direction",
        "shared_only_direction": "shared_canonical_direction",
        "residual_only_direction": "residual_canonical_direction",
        "loco_shared_direction": "loco_shared_canonical_direction",
        "loco_residual_direction": "loco_residual_canonical_direction",
        "random_direction_control": "random_direction_control",
        "wrong_direction_control": "wrong_direction_control",
    }
    vectors: dict[tuple[str, str, str], np.ndarray] = {}
    inventory_rows = []
    raw_by_position: dict[str, dict[str, np.ndarray]] = {}
    enabled_modes = set(audit.enabled_modes)

    for extraction_position in config.extraction_positions:
        raw_directions, _ = contrast_directions(
            train_pairs,
            position_activations[extraction_position][resolved_layer],
            train_split=dataset.train_split,
        )
        raw_by_position[extraction_position] = raw_directions
        bank = decompose_directions(
            raw_directions,
            subspace_dims=[audit.subspace_dim],
            mixing_grid=[],
            seed=audit.runtime.seed,
            include_loco=True,
            loco_subspace_dim=1,
        )
        for (pair_type, source_mode, _, _), vector in bank.vectors.items():
            if source_mode not in mode_map:
                continue
            mode = mode_map[source_mode]
            if mode not in enabled_modes:
                continue
            vector = np.asarray(vector, dtype=np.float32)
            vectors[(extraction_position, pair_type, mode)] = vector
            source_row = bank.inventory[
                bank.inventory["pair_type"].astype(str).eq(pair_type)
                & bank.inventory["mode"].astype(str).eq(source_mode)
            ].iloc[0]
            inventory_rows.append(
                {
                    "extraction_position": extraction_position,
                    "intervention_position": config.intervention_position,
                    "pair_type": pair_type,
                    "mode": mode,
                    "extraction_mapping": dataset.canonical_mapping,
                    "layer_index": resolved_layer,
                    "subspace_dim": audit.subspace_dim,
                    "direction_l2": float(np.linalg.norm(vector)),
                    "shared_explained_variance": float(source_row["shared_explained_variance"]),
                    "basis_protocol": source_row.get("basis_protocol", "in_sample"),
                    "basis_source_contrasts": source_row.get("basis_source_contrasts", "__all__"),
                    "n_basis_contrasts": source_row.get("n_basis_contrasts", np.nan),
                    "self_included_in_basis": source_row.get("self_included_in_basis", True),
                    "unnormalized_projection_l2": source_row.get("unnormalized_projection_l2", np.nan),
                    "unnormalized_residual_l2": source_row.get("unnormalized_residual_l2", np.nan),
                    "projection_ratio": source_row.get("projection_ratio", np.nan),
                    "n_train_pairs": int(
                        train_pairs[train_pairs["pair_type"].eq(pair_type)]["pair_id"].nunique()
                    ),
                }
            )

    if "raw_position_norm_matched" in enabled_modes:
        for pair_type in sorted(raw_by_position["pre_answer"]):
            pre_answer = raw_by_position["pre_answer"][pair_type]
            pre_norm = float(np.linalg.norm(pre_answer))
            for extraction_position in config.extraction_positions:
                source = raw_by_position[extraction_position][pair_type]
                source_norm = float(np.linalg.norm(source))
                if source_norm == 0:
                    raise ValueError(
                        f"Cannot norm-match zero direction for {extraction_position}/{pair_type}"
                    )
                vector = np.asarray(source * (pre_norm / source_norm), dtype=np.float32)
                vectors[
                    (extraction_position, pair_type, "raw_position_norm_matched")
                ] = vector
                inventory_rows.append(
                    {
                        "extraction_position": extraction_position,
                        "intervention_position": config.intervention_position,
                        "pair_type": pair_type,
                        "mode": "raw_position_norm_matched",
                        "extraction_mapping": dataset.canonical_mapping,
                        "layer_index": resolved_layer,
                        "subspace_dim": 0,
                        "direction_l2": float(np.linalg.norm(vector)),
                        "natural_direction_l2": source_norm,
                        "norm_reference_position": "pre_answer",
                        "shared_explained_variance": np.nan,
                        "n_train_pairs": int(
                            train_pairs[train_pairs["pair_type"].eq(pair_type)][
                                "pair_id"
                            ].nunique()
                        ),
                    }
                )
    if "zero_direction_control" in enabled_modes:
        for extraction_position in config.extraction_positions:
            raw_directions = raw_by_position[extraction_position]
            hidden_dim = next(iter(raw_directions.values())).shape[0]
            for pair_type in raw_directions:
                zero = np.zeros(hidden_dim, dtype=np.float32)
                vectors[(extraction_position, pair_type, "zero_direction_control")] = zero
                inventory_rows.append(
                    {
                        "extraction_position": extraction_position,
                        "intervention_position": config.intervention_position,
                        "pair_type": pair_type,
                        "mode": "zero_direction_control",
                        "extraction_mapping": dataset.canonical_mapping,
                        "layer_index": resolved_layer,
                        "subspace_dim": 0,
                        "direction_l2": 0.0,
                        "shared_explained_variance": np.nan,
                        "n_train_pairs": int(
                            train_pairs[train_pairs["pair_type"].eq(pair_type)][
                                "pair_id"
                            ].nunique()
                        ),
                    }
                )

    inventory = pd.DataFrame(inventory_rows)
    geometry_rows = []
    for extraction_position, raw_directions in raw_by_position.items():
        pair_types = sorted(raw_directions)
        cosines = [
            _cosine(raw_directions[left], raw_directions[right])
            for index, left in enumerate(pair_types)
            for right in pair_types[index + 1 :]
        ]
        explained = inventory[
            inventory["extraction_position"].eq(extraction_position)
            & inventory["mode"].eq("raw_canonical_direction")
        ]["shared_explained_variance"].dropna()
        geometry_rows.append(
            {
                "geometry_type": "within_position_sharedness",
                "extraction_position": extraction_position,
                "pair_type": "__all__",
                "mode": "raw_canonical_direction",
                "mean_pairwise_raw_cosine": float(np.mean(cosines)) if cosines else np.nan,
                "shared_explained_variance": float(explained.mean()) if len(explained) else np.nan,
                "cross_position_cosine": np.nan,
            }
        )
    for pair_type in sorted({key[1] for key in vectors}):
        for mode in [*PAPER_MODES, *CONTROL_MODES]:
            left = vectors.get(("scenario_end", pair_type, mode))
            right = vectors.get(("pre_answer", pair_type, mode))
            if left is None or right is None:
                continue
            geometry_rows.append(
                {
                    "geometry_type": "cross_position_alignment",
                    "extraction_position": "scenario_end_vs_pre_answer",
                    "pair_type": pair_type,
                    "mode": mode,
                    "mean_pairwise_raw_cosine": np.nan,
                    "shared_explained_variance": np.nan,
                    "cross_position_cosine": _cosine(left, right),
                }
            )
    return vectors, inventory, pd.DataFrame(geometry_rows)


def _bootstrap_equal_pair_type(
    values: pd.DataFrame,
    *,
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    parts = [group["difference"].dropna().to_numpy(float) for _, group in values.groupby("pair_type")]
    parts = [part for part in parts if len(part)]
    if not parts:
        return np.nan, np.nan, np.nan, 0
    observed = float(np.mean([part.mean() for part in parts]))
    boot = np.zeros(n_boot, dtype=float)
    for part in parts:
        indices = rng.integers(0, len(part), size=(n_boot, len(part)))
        boot += part[indices].mean(axis=1) / len(parts)
    tail = (1.0 - confidence) / 2.0
    return observed, float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail)), int(sum(map(len, parts)))


def build_position_comparisons(
    pair_effects: pd.DataFrame,
    *,
    n_boot: int,
    n_permutations: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for (mapping_name, mapping_family, mode), group in pair_effects.groupby(
        ["mapping_name", "mapping_family", "mode"], sort=True
    ):
        for metric in POSITION_COMPARISON_METRICS:
            wide = group.pivot_table(
                index=["pair_id", "pair_type"],
                columns="extraction_position",
                values=metric,
                aggfunc="mean",
            )
            if "scenario_end" not in wide or "pre_answer" not in wide:
                continue
            paired = wide[["scenario_end", "pre_answer"]].dropna().reset_index()
            if paired.empty:
                continue
            paired["difference"] = paired["scenario_end"] - paired["pre_answer"]
            mean, low, high, n = _bootstrap_equal_pair_type(
                paired,
                n_boot=n_boot,
                confidence=confidence,
                rng=rng,
            )
            records.append(
                {
                    "mapping_name": mapping_name,
                    "mapping_family": mapping_family,
                    "mode": mode,
                    "metric": metric,
                    "n_pairs": n,
                    "scenario_end_mean": float(paired["scenario_end"].mean()),
                    "pre_answer_mean": float(paired["pre_answer"].mean()),
                    "scenario_end_minus_pre_answer": mean,
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "p_two_sided": paired_sign_flip_pvalue(
                        paired["difference"].to_numpy(),
                        n_permutations=n_permutations,
                        rng=rng,
                    ),
                }
            )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for _, indices in out.groupby(["mode", "metric"], sort=True).groups.items():
        idx = list(indices)
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_two_sided"])
    out["significant_holm"] = out["p_holm"].lt(0.05)
    return out


def run_single_model_position_audit(
    model: Any,
    tokenizer: Any,
    config: ExtractionPositionAuditConfig,
    model_config: MappingAuditModelConfig,
) -> dict[str, pd.DataFrame]:
    audit = config.mapping_audit
    alias = model_config.model.alias
    model_dir = audit.output_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    base_items = load_base_items(audit.dataset, seed=audit.runtime.seed)
    mapping_by_name = {mapping.name: mapping for mapping in audit.mappings}
    canonical_mapping = mapping_by_name[audit.dataset.canonical_mapping]
    canonical_items = build_mapping_items(base_items, audit.dataset, canonical_mapping, canonical_mapping)
    vectors, direction_inventory, direction_geometry = build_position_direction_banks(
        model, tokenizer, canonical_items, config, model_config
    )
    for extraction_position in config.extraction_positions:
        payload = {
            f"{pair_type}__{mode}": vector
            for (position, pair_type, mode), vector in vectors.items()
            if position == extraction_position
        }
        np.savez_compressed(model_dir / f"direction_bank__{extraction_position}.npz", **payload)

    all_mapping_items = []
    all_eval_rows = []
    all_pair_effects = []
    all_ci = []
    all_mode_comparisons = []
    all_contrasts = []
    transport_eval_rows = []
    transport_pair_effects = []
    transport_ci = []
    for mapping in audit.mappings:
        mapping_items = build_mapping_items(base_items, audit.dataset, mapping, canonical_mapping)
        all_mapping_items.append(mapping_items)
        eval_items = build_mapping_eval_items(mapping_items, audit.dataset, mapping, canonical_mapping)
        for extraction_position in config.extraction_positions:
            position_vectors = {
                (pair_type, mode): vector
                for (position, pair_type, mode), vector in vectors.items()
                if position == extraction_position
            }
            position_inventory = direction_inventory[
                direction_inventory["extraction_position"].eq(extraction_position)
            ].copy()
            eval_rows = evaluate_fixed_directions(
                model,
                tokenizer,
                eval_items,
                position_vectors,
                position_inventory,
                layer_index=model_config.locked_layer,
                alpha=model_config.locked_alpha,
                n_choices=len(audit.dataset.label_ranks),
                batch_size=audit.runtime.batch_size,
                max_length=audit.runtime.max_length,
            )
            eval_rows["extraction_position"] = extraction_position
            eval_rows["intervention_position"] = config.intervention_position
            eval_rows["evaluation_condition"] = "fixed_pre_answer_injection"
            all_eval_rows.append(eval_rows)
            effects = build_pair_effects(eval_rows)
            effects["extraction_position"] = extraction_position
            effects["intervention_position"] = config.intervention_position
            all_pair_effects.append(effects)
            ci = build_mapping_mode_ci(
                effects,
                n_boot=audit.statistics.n_boot,
                confidence=audit.statistics.confidence,
                seed=audit.runtime.seed,
            )
            ci["extraction_position"] = extraction_position
            ci["intervention_position"] = config.intervention_position
            all_ci.append(ci)
            comparisons = build_mode_comparisons(
                effects,
                n_permutations=audit.statistics.n_permutations,
                seed=audit.runtime.seed + 1,
            )
            comparisons["extraction_position"] = extraction_position
            comparisons["intervention_position"] = config.intervention_position
            all_mode_comparisons.append(comparisons)
            contrasts = build_contrast_summary(eval_rows)
            contrasts["extraction_position"] = extraction_position
            contrasts["intervention_position"] = config.intervention_position
            all_contrasts.append(contrasts)

            if (
                extraction_position == "scenario_end"
                and mapping.name in set(config.transport_control_mappings)
            ):
                transport_rows = evaluate_fixed_directions(
                    model,
                    tokenizer,
                    eval_items,
                    position_vectors,
                    position_inventory,
                    layer_index=model_config.locked_layer,
                    alpha=model_config.locked_alpha,
                    n_choices=len(audit.dataset.label_ranks),
                    batch_size=audit.runtime.batch_size,
                    max_length=audit.runtime.max_length,
                    injection_position="scenario_end",
                )
                transport_rows["extraction_position"] = "scenario_end"
                transport_rows["intervention_position"] = "scenario_end"
                transport_rows["evaluation_condition"] = "matched_position_transport_control"
                transport_eval_rows.append(transport_rows)
                transport_effects = build_pair_effects(transport_rows)
                transport_effects["extraction_position"] = "scenario_end"
                transport_effects["intervention_position"] = "scenario_end"
                transport_effects["evaluation_condition"] = (
                    "matched_position_transport_control"
                )
                transport_pair_effects.append(transport_effects)
                transport_ci.append(
                    build_mapping_mode_ci(
                        transport_effects,
                        n_boot=audit.statistics.n_boot,
                        confidence=audit.statistics.confidence,
                        seed=audit.runtime.seed,
                    )
                )

    pair_effects_all = pd.concat(all_pair_effects, ignore_index=True, sort=False)
    tables = {
        "base_items": base_items,
        "mapping_items": pd.concat(all_mapping_items, ignore_index=True, sort=False),
        "position_direction_inventory": direction_inventory,
        "position_direction_geometry": direction_geometry,
        "position_eval_rows": pd.concat(all_eval_rows, ignore_index=True, sort=False),
        "position_pair_effects": pair_effects_all,
        "position_mapping_mode_ci": pd.concat(all_ci, ignore_index=True, sort=False),
        "position_mode_comparisons": pd.concat(all_mode_comparisons, ignore_index=True, sort=False),
        "position_contrast_summary": pd.concat(all_contrasts, ignore_index=True, sort=False),
        "position_comparisons": build_position_comparisons(
            pair_effects_all,
            n_boot=audit.statistics.n_boot,
            n_permutations=audit.statistics.n_permutations,
            confidence=audit.statistics.confidence,
            seed=audit.runtime.seed + 2,
        ),
        "transport_eval_rows": (
            pd.concat(transport_eval_rows, ignore_index=True, sort=False)
            if transport_eval_rows
            else pd.DataFrame()
        ),
        "transport_pair_effects": (
            pd.concat(transport_pair_effects, ignore_index=True, sort=False)
            if transport_pair_effects
            else pd.DataFrame()
        ),
        "transport_mapping_mode_ci": (
            pd.concat(transport_ci, ignore_index=True, sort=False)
            if transport_ci
            else pd.DataFrame()
        ),
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
    completion = pd.DataFrame(
        [
            {
                "model_alias": alias,
                "model_name": model_config.model.name,
                "status": "complete",
                "locked_layer": model_config.locked_layer,
                "locked_alpha": model_config.locked_alpha,
                "subspace_dim": audit.subspace_dim,
                "extraction_positions": ",".join(config.extraction_positions),
                "intervention_position": config.intervention_position,
                "n_eval_mappings": len(audit.mappings),
            }
        ]
    )
    completion.to_csv(model_dir / "position_audit_run_complete.csv", index=False)
    return tables


def position_audit_run_is_complete(model_dir: str | Path) -> bool:
    root = Path(model_dir)
    return all(
        (root / filename).exists()
        for filename in [
            "position_audit_run_complete.csv",
            "position_mapping_mode_ci.csv",
            "position_comparisons.csv",
            "position_direction_inventory.csv",
        ]
    )


def aggregate_position_audit(config: ExtractionPositionAuditConfig) -> dict[str, pd.DataFrame]:
    audit = config.mapping_audit
    names = [
        "position_audit_run_complete",
        "position_mapping_mode_ci",
        "position_mode_comparisons",
        "position_comparisons",
        "position_direction_inventory",
        "position_direction_geometry",
        "position_contrast_summary",
        "transport_mapping_mode_ci",
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
    ci = aggregated.get("position_mapping_mode_ci", pd.DataFrame()).copy()
    if ci.empty:
        global_summary = pd.DataFrame()
    else:
        focus = ci[ci["n_pairs_with_moved_targets"].gt(0)].copy()
        focus["semantic_preferred"] = focus[
            "semantic_minus_original_prob_gain_moved_ci_low"
        ].gt(0)
        focus["letter_preferred"] = focus[
            "semantic_minus_original_prob_gain_moved_ci_high"
        ].lt(0)
        global_summary = (
            focus.groupby(["extraction_position", "mode"], as_index=False)
            .agg(
                n_model_mapping_cells=("mapping_name", "size"),
                n_models=("model_alias", "nunique"),
                mean_semantic_prob_gain=("moved_semantic_prob_gain_mean", "mean"),
                mean_original_letter_prob_gain=("moved_original_letter_prob_gain_mean", "mean"),
                mean_semantic_minus_original=(
                    "semantic_minus_original_prob_gain_moved_mean",
                    "mean",
                ),
                n_semantic_preferred=("semantic_preferred", "sum"),
                n_letter_preferred=("letter_preferred", "sum"),
            )
        )
    aggregated["position_global_summary"] = global_summary
    write_tables(aggregated, audit.output_dir)
    return aggregated


def run_extraction_position_audit(
    config: ExtractionPositionAuditConfig,
    *,
    model_aliases: Iterable[str] | None = None,
    model_loader: Callable[..., tuple[Any, Any]] = load_tokenizer_and_model,
    progress: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    audit = config.mapping_audit
    audit.output_dir.mkdir(parents=True, exist_ok=True)
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
        if position_audit_run_is_complete(model_dir) and not audit.runtime.force_rerun:
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
            run_single_model_position_audit(model, tokenizer, config, model_config)
            (model_dir / "position_audit_run_error.csv").unlink(missing_ok=True)
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
            pd.DataFrame([error]).to_csv(model_dir / "position_audit_run_error.csv", index=False)
            progress(f"[{public_model.alias}] failed: {type(exc).__name__}: {exc}")
            if not audit.runtime.continue_on_error:
                raise
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            cleanup_model()
        aggregate_position_audit(config)
    if errors:
        pd.DataFrame(errors).to_csv(audit.output_dir / "position_audit_errors.csv", index=False)
    else:
        (audit.output_dir / "position_audit_errors.csv").unlink(missing_ok=True)
    return aggregate_position_audit(config)


def run_extraction_position_audit_from_json(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    model_source_overrides: dict[str, str] | None = None,
    model_aliases: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    config = ExtractionPositionAuditConfig.from_json(
        config_path,
        project_root=project_root,
        model_source_overrides=model_source_overrides,
    )
    return run_extraction_position_audit(config, model_aliases=model_aliases)
