from __future__ import annotations

import numpy as np
import pandas as pd

from cross_encoding_steering.iti_probe_audit import (
    HeadDirection,
    adjusted_pair_effects,
    bootstrap_adjusted_effects,
    build_cross_method_decision,
    build_cross_method_interface_summary,
    build_iti_competence_summary,
    build_iti_input_inventory,
    filter_competent_models,
    fit_ridge_head_probe,
    randomize_head_bank,
    resolve_candidate_layers,
    selected_minus_wrong_pair_effects,
    select_top_heads,
)


def test_candidate_layers_are_unique_and_in_range() -> None:
    assert resolve_candidate_layers(32, [0.5, 0.625, 0.75, 0.875]) == (16, 19, 23, 27)
    assert resolve_candidate_layers(1, [0.0, 0.5, 1.0]) == (0,)


def test_ridge_head_probe_orients_toward_positive_class() -> None:
    rng = np.random.default_rng(13)
    train_negative = rng.normal(-2.0, 0.2, size=(40, 8))
    train_positive = rng.normal(2.0, 0.2, size=(40, 8))
    validation_negative = rng.normal(-2.0, 0.2, size=(20, 8))
    validation_positive = rng.normal(2.0, 0.2, size=(20, 8))
    state = fit_ridge_head_probe(
        np.concatenate([train_negative, train_positive]),
        np.asarray([0] * 40 + [1] * 40),
        np.concatenate([validation_negative, validation_positive]),
        np.asarray([0] * 20 + [1] * 20),
        ridge=0.01,
    )
    assert state["validation_accuracy"] == 1.0
    assert np.isclose(np.linalg.norm(state["direction"]), 1.0)
    assert np.dot(state["direction"], np.ones(8)) > 0
    assert np.isclose(np.linalg.norm(state["mass_mean_shift_direction"]), 1.0)
    assert np.dot(state["mass_mean_shift_direction"], np.ones(8)) > 0
    assert state["mass_mean_shift_scale"] > 0


def test_top_head_selection_and_random_controls_are_deterministic() -> None:
    directions = [
        HeadDirection(10, index, np.asarray([1.0, 0.0]), 2.0, 0.9 - index * 0.1, 1.0)
        for index in range(4)
    ]
    selected = select_top_heads({"a_vs_b": directions}, 2)
    first = randomize_head_bank(selected, seed=13)
    second = randomize_head_bank(selected, seed=13)
    assert [item.head_index for item in selected["a_vs_b"]] == [0, 1]
    for left, right, source in zip(
        first["a_vs_b"], second["a_vs_b"], selected["a_vs_b"], strict=True
    ):
        assert np.allclose(left.vector, right.vector)
        assert np.isclose(np.linalg.norm(left.vector), 1.0)
        assert left.scale == source.scale


def test_adjusted_effects_and_bootstrap_use_random_mean() -> None:
    rows = []
    for model in ["m1", "m2"]:
        for pair_type in ["a_vs_b", "b_vs_c"]:
            for pair_id in ["p1", "p2", "p3"]:
                for mode, target, slot in [
                    ("iti_probe_direction", 0.6, 0.4),
                    ("iti_random_direction_seed_13", 0.1, 0.1),
                    ("iti_random_direction_seed_29", 0.2, 0.0),
                ]:
                    rows.append(
                        {
                            "model_alias": model,
                            "model_name": model,
                            "pair_id": f"{model}-{pair_type}-{pair_id}",
                            "pair_type": pair_type,
                            "mode": mode,
                            "interface": "letter_canonical",
                            "template": "classify",
                            "alpha": 0.5,
                            "target_margin_gain": target,
                            "original_slot_margin_gain": slot,
                            "n_endpoints": 2,
                            "n_intervened_heads": 8,
                        }
                    )
    effects = adjusted_pair_effects(pd.DataFrame(rows))
    current = effects[effects["metric"].eq("current_label_margin")]
    assert np.allclose(current["adjusted_gain"], 0.45)
    summary = bootstrap_adjusted_effects(
        effects,
        n_boot=200,
        confidence=0.95,
        seed=13,
    )
    row = summary[
        summary["metric"].eq("current_label_margin")
        & summary["interface"].eq("letter_canonical")
    ].iloc[0]
    assert np.isclose(row["mean_adjusted_gain"], 0.45)
    assert row["ci_low"] > 0


def test_input_inventory_requires_all_configured_splits() -> None:
    class Dataset:
        contrast_column = "pair_type"
        split_column = "split"
        pair_id_column = "pair_id"
        label_column = "label"
        train_split = "train"
        validation_split = "validation"
        test_split = "test"

    class Audit:
        dataset = Dataset()

    class Cross:
        audit = Audit()

    class Config:
        cross_interface = Cross()

    rows = pd.DataFrame(
        [
            {
                "pair_type": pair_type,
                "split": split,
                "pair_id": f"{pair_type}-{split}",
                "label": label,
            }
            for pair_type in ["a_vs_b", "b_vs_c"]
            for split in ["train", "validation", "test"]
            for label in ["low", "high"]
        ]
    )
    inventory = build_iti_input_inventory(rows, Config())
    assert len(inventory) == 6

    missing = rows[
        ~(
            rows["pair_type"].eq("b_vs_c")
            & rows["split"].eq("validation")
        )
    ]
    try:
        build_iti_input_inventory(missing, Config())
    except ValueError as exc:
        assert "('b_vs_c', 'validation')" in str(exc)
    else:
        raise AssertionError("Missing validation cells should fail before model loading")


def test_competence_gate_excludes_failed_models_without_hiding_probe_readability() -> None:
    selection = pd.DataFrame(
        [
            {
                "model_alias": "pass",
                "model_name": "Pass",
                "top_k_heads": 1,
                "alpha": 1.0,
                "mean_validation_target_margin_gain": 0.5,
                "min_contrast_validation_gain": 0.2,
                "n_positive_contrasts": 2,
                "n_contrasts": 2,
                "validation_competence_pass": True,
                "minimum_validation_gain": 0.0,
            },
            {
                "model_alias": "fail",
                "model_name": "Fail",
                "top_k_heads": 1,
                "alpha": 1.0,
                "mean_validation_target_margin_gain": -0.1,
                "min_contrast_validation_gain": -0.2,
                "n_positive_contrasts": 0,
                "n_contrasts": 2,
                "validation_competence_pass": False,
                "minimum_validation_gain": 0.0,
            },
        ]
    )
    inventory = pd.DataFrame(
        [
            {
                "model_alias": model,
                "pair_type": pair_type,
                "validation_rank": 1,
                "validation_accuracy": accuracy,
            }
            for model, accuracy in [("pass", 0.9), ("fail", 0.8)]
            for pair_type in ["a_vs_b", "b_vs_c"]
        ]
    )
    summary = build_iti_competence_summary(selection, inventory)
    assert summary["validation_competence_pass"].sum() == 1
    assert summary.set_index("model_alias").loc["fail", "max_head_validation_accuracy"] == 0.8
    frame = pd.DataFrame({"model_alias": ["pass", "fail"], "value": [1, 2]})
    filtered = filter_competent_models(frame, summary)
    assert filtered["model_alias"].tolist() == ["pass"]


def test_selected_minus_wrong_effects_preserve_interface_metrics() -> None:
    rows = []
    for template in ["a", "b"]:
        for mode, current, slot in [
            ("iti_probe_direction", 0.6, 0.4),
            ("iti_wrong_direction", -0.2, -0.1),
        ]:
            rows.append(
                {
                    "model_alias": "m",
                    "model_name": "M",
                    "pair_id": "p",
                    "pair_type": "a_vs_b",
                    "mode": mode,
                    "interface": "letter_canonical",
                    "template": template,
                    "target_margin_gain": current,
                    "original_slot_margin_gain": slot,
                }
            )
    effects = selected_minus_wrong_pair_effects(pd.DataFrame(rows))
    values = effects.set_index("metric")["adjusted_gain"].to_dict()
    assert np.isclose(values["current_label_margin"], 0.8)
    assert np.isclose(values["extraction_slot_margin"], 0.5)
    assert np.isclose(values["current_label_minus_extraction_slot"], 0.3)


def test_cross_method_summary_uses_competence_gated_iti_as_primary() -> None:
    caa = pd.DataFrame(
        [
            {
                "metric": metric,
                "interface": interface,
                "n_models": 4,
                "n_contrasts": 3,
                "n_pairs": 10,
                "mean_adjusted_gain": value,
                "mean_adjusted_ci_low": value - 0.1,
                "mean_adjusted_ci_high": value + 0.1,
                "confidence": 0.95,
                "n_boot": 100,
            }
            for interface in ["letter_canonical", "letter_reversed"]
            for metric, value in [
                ("semantic_target_margin", 1.0),
                ("original_slot_margin", 2.0),
                ("semantic_minus_original_slot", -1.0),
            ]
        ]
    )
    iti_all = bootstrap_adjusted_effects(
        pd.DataFrame(),
        n_boot=10,
        confidence=0.95,
        seed=13,
    )
    iti_competent = pd.DataFrame(
        [
            {
                "metric": metric,
                "interface": interface,
                "n_models": 3,
                "n_contrasts": 3,
                "n_pairs": 10,
                "mean_adjusted_gain": value,
                "ci_low": value - 0.1,
                "ci_high": value + 0.1,
                "confidence": 0.95,
                "n_boot": 100,
            }
            for interface in ["letter_canonical", "letter_reversed"]
            for metric, value in [
                ("current_label_margin", 1.0),
                ("extraction_slot_margin", 2.0),
                ("current_label_minus_extraction_slot", -1.0),
            ]
        ]
    )
    summary = build_cross_method_interface_summary(caa, iti_all, iti_competent)
    decision = build_cross_method_decision(summary)
    assert set(decision["method"]) == {"caa_residual_stream", "iti_head_probe"}
    iti = decision[decision["method"].eq("iti_head_probe")].iloc[0]
    assert iti["model_scope"] == "source_interface_competent_models"
    assert iti["all_non_source_extraction_slot_positive"]
    assert iti["all_non_source_extraction_slot_dominant"]
