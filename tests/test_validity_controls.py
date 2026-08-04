from pathlib import Path

import pandas as pd

from cross_interface_steering.validity_controls import (
    ValidityControlsConfig,
    build_multi_random_context_statistics,
    build_multi_random_margin_statistics,
    prepare_human_adjudication,
    prepare_judge_validation,
    summarize_human_adjudication,
    summarize_judge_validation,
)


def _config(tmp_path: Path) -> ValidityControlsConfig:
    output = tmp_path / "validity"
    return ValidityControlsConfig(
        project_root=tmp_path,
        output_dir=output,
        multi_random_output_dir=output / "multi_random",
        target_pair_effects=tmp_path / "target_pairs.csv",
        target_context_rows=tmp_path / "target_context.csv",
        context_pair_metadata=tmp_path / "context_pairs.csv",
        published_caa_judgments=tmp_path / "judgments.csv",
        published_caa_multi_judge_scores=tmp_path / "multi_judge_scores.csv",
        annotation_files=(output / "annotator_1.csv", output / "annotator_2.csv"),
        adjudication_file=output / "adjudicator.csv",
        panel_judge_aliases=("judge_a", "judge_b", "judge_c"),
        target_modes=("raw_pre_answer",),
        interfaces=("letter_canonical",),
        required_multipliers=(-2.0, 0.0, 2.0),
        judge_sample_per_model_behavior=2,
        judge_pair_items_across_models=False,
        adjudication_score_gap=3,
        minimum_base_gap=0.05,
        n_boot=200,
        confidence=0.95,
        seed=13,
    )


def test_multi_random_statistics_keep_seed_variability(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.multi_random_output_dir.mkdir(parents=True)
    target_pair_rows = []
    random_pair_rows = []
    target_context_rows = []
    random_eval_rows = []
    for contrast_index, contrast in enumerate(("low_mid", "mid_high")):
        for pair_index in range(4):
            pair_id = f"{contrast}-{pair_index}"
            target_gain = 2.0 + 0.1 * pair_index
            target_pair_rows.append({
                "model_alias": "model",
                "model_name": "Model",
                "pair_id": pair_id,
                "pair_type": contrast,
                "mode": "raw_pre_answer",
                "interface": "letter_canonical",
                "template": "one",
                "target_margin_gain": target_gain,
            })
            target_context_rows.append({
                "model_alias": "model",
                "model_name": "Model",
                "pair_id": pair_id,
                "pair_type": contrast,
                "mode": "raw_pre_answer",
                "interface": "letter_canonical",
                "template": "one",
                "base_discrimination_gap": 1.0,
                "base_rank_correct": True,
                "discrimination_gap_change": -0.2,
                "patched_rank_correct": 0.75,
            })
            for seed, random_gain in ((13, 0.1), (29, 0.3)):
                mode = f"random_direction_control_seed_{seed}"
                random_pair_rows.append({
                    "model_alias": "model",
                    "model_name": "Model",
                    "pair_id": pair_id,
                    "pair_type": contrast,
                    "mode": mode,
                    "interface": "letter_canonical",
                    "template": "one",
                    "target_margin_gain": random_gain,
                })
                for endpoint, base, patched in (
                    ("low", 0.0, 0.05 * (seed == 29)),
                    ("high", 1.0, 1.1 + 0.05 * (seed == 29)),
                ):
                    random_eval_rows.append({
                        "model_alias": "model",
                        "model_name": "Model",
                        "pair_id": pair_id,
                        "pair_type": contrast,
                        "endpoint": endpoint,
                        "evaluation_policy": "context_plus",
                        "mode": mode,
                        "interface": "letter_canonical",
                        "template": "one",
                        "base_high_low_margin": base,
                        "patched_high_low_margin": patched,
                    })
    pd.DataFrame(target_pair_rows).to_csv(config.target_pair_effects, index=False)
    pd.DataFrame(target_context_rows).to_csv(config.target_context_rows, index=False)
    metadata_rows = []
    for contrast in ("low_mid", "mid_high"):
        for pair_index in range(4):
            metadata_rows.append({
                "pair_id": f"{contrast}-{pair_index}",
                "pair_type": contrast,
                "group_key": f"shared-group-{pair_index}",
                "negative_text": "same" if pair_index == 0 else f"negative {pair_index}",
                "positive_text": "same" if pair_index == 0 else f"positive {pair_index}",
            })
    pd.DataFrame(metadata_rows).to_csv(config.context_pair_metadata, index=False)
    pd.DataFrame(random_pair_rows).to_csv(
        config.multi_random_output_dir / "cross_interface_pair_effects.csv", index=False
    )
    pd.DataFrame(random_eval_rows).to_csv(
        config.multi_random_output_dir / "cross_interface_eval_rows.csv", index=False
    )

    margin = build_multi_random_margin_statistics(config)
    by_seed = margin["multi_random_standardized_gain_by_seed"]
    assert set(by_seed["random_seed"]) == {13, 29}
    assert margin["multi_random_standardized_gain_summary"].iloc[0]["n_random_seeds"] == 2
    assert bool(margin["multi_random_standardized_gain_summary"].iloc[0]["all_seed_means_positive"])
    margin_ci = margin["multi_random_standardized_gain_bootstrap_ci"].iloc[0]
    assert margin_ci["ci_low"] <= margin_ci["estimate"] <= margin_ci["ci_high"]
    assert margin_ci["n_random_seeds"] == 2

    context = build_multi_random_context_statistics(config)
    filter_audit = context["context_pair_filter_audit"].iloc[0]
    assert filter_audit["n_input_pairs"] == 8
    assert filter_audit["n_excluded_identical_text_pairs"] == 2
    assert filter_audit["n_retained_pairs"] == 6
    assert context["multi_random_context_pair_effects"]["pair_id"].nunique() == 6
    assert set(context["multi_random_context_by_seed"]["random_seed"]) == {13, 29}
    assert set(context["multi_random_context_by_seed"]["metric"]) == {
        "discrimination_gap_change",
        "patched_rank_correct",
    }
    context_ci = context["multi_random_context_bootstrap_ci"]
    assert set(context_ci["metric"]) == {"discrimination_gap_change", "patched_rank_correct"}
    assert (context_ci["ci_low"] <= context_ci["estimate"]).all()
    assert (context_ci["estimate"] <= context_ci["ci_high"]).all()
    baseline = context["multi_random_context_baseline_bootstrap_ci"]
    assert set(baseline["metric"]) == {"base_discrimination_gap", "base_rank_accuracy"}
    assert baseline.set_index("metric").loc["base_discrimination_gap", "estimate"] == 1.0
    assert baseline.set_index("metric").loc["base_rank_accuracy", "estimate"] == 1.0
    per_model = context["multi_random_context_per_model_bootstrap_ci"]
    assert set(per_model["metric"]) == {"discrimination_gap_change", "patched_rank_correct"}
    assert set(per_model["model_alias"]) == {"model"}
    cluster_ci = context["multi_random_context_cluster_bootstrap_ci"]
    assert set(cluster_ci["metric"]) == {"discrimination_gap_change", "patched_rank_correct"}
    assert cluster_ci["n_clusters"].eq(3).all()


def test_judge_validation_is_blind_and_summarizes_completed_forms(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = []
    for item_index in range(3):
        for multiplier, score in ((-2.0, 0.0), (0.0, 2.0), (2.0, 5.0)):
            rows.append({
                "model_alias": "model",
                "model_name": "Model",
                "behavior": "hallucination",
                "item_id": f"item-{item_index}",
                "multiplier": multiplier,
                "question": "Question",
                "response": f"Response {item_index} {multiplier}",
                "judge_score": score,
            })
    pd.DataFrame(rows).to_csv(config.published_caa_judgments, index=False)
    tables = prepare_judge_validation(config)
    assert len(tables["judge_validation_sample_key"]) == 6
    key = tables["judge_validation_sample_key"]
    for path in config.annotation_files:
        form = pd.read_csv(path)
        assert "model_alias" not in form
        assert "multiplier" not in form
        assert "judge_score" not in form
        form = form.merge(key[["annotation_id", "judge_score"]], on="annotation_id", how="left")
        form["human_score"] = form["judge_score"]
        form["evidence_span"] = "decisive evidence"
        form["confidence"] = "high"
        form.drop(columns="judge_score").to_csv(path, index=False)

    summary = summarize_judge_validation(config)
    agreement = summary["judge_validation_agreement"]
    human_row = agreement[agreement["scope"].eq("human_1_vs_human_2")].iloc[0]
    assert human_row["quadratic_weighted_kappa"] == 1.0
    assert summary["judge_validation_verdict_agreement"]["verdict_agrees"].all()


def test_adjudication_and_three_judge_panel_are_summarized(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = []
    panel_rows = []
    for item_index in range(2):
        for multiplier, score in ((-2.0, 0.0), (0.0, 2.0), (2.0, 5.0)):
            row = {
                "model_alias": "model",
                "model_name": "Model",
                "behavior": "hallucination",
                "item_id": f"item-{item_index}",
                "multiplier": multiplier,
                "question": "Question",
                "response": f"Response {item_index} {multiplier}",
                "judge_score": score,
            }
            rows.append(row)
            for alias in config.panel_judge_aliases:
                panel_rows.append({
                    **{key: row[key] for key in (
                        "model_alias", "model_name", "behavior", "item_id", "multiplier"
                    )},
                    "judge_alias": alias,
                    "judge_score": score,
                })
    pd.DataFrame(rows).to_csv(config.published_caa_judgments, index=False)
    pd.DataFrame(panel_rows).to_csv(config.published_caa_multi_judge_scores, index=False)
    prepared = prepare_judge_validation(config)
    key = prepared["judge_validation_sample_key"]
    disagreement_id = key.iloc[0]["annotation_id"]
    for index, path in enumerate(config.annotation_files):
        form = pd.read_csv(path)
        form = form.merge(key[["annotation_id", "judge_score"]], on="annotation_id", how="left")
        form["human_score"] = form["judge_score"]
        if index == 1:
            form.loc[form["annotation_id"].eq(disagreement_id), "human_score"] = 10
        form["evidence_span"] = "decisive evidence"
        form["confidence"] = "high"
        form.drop(columns="judge_score").to_csv(path, index=False)

    summary = summarize_judge_validation(config)
    panel = summary["judge_validation_agreement"]
    assert "three_judge_panel_vs_human_mean" in set(panel["scope"])

    adjudication = prepare_human_adjudication(config)
    inventory = adjudication["judge_validation_adjudication_inventory"]
    assert inventory.loc[inventory["behavior"].eq("__all__"), "n_adjudication_rows"].item() == 1
    form = pd.read_csv(config.adjudication_file)
    assert "model_alias" not in form
    assert "multiplier" not in form
    form["adjudicated_score"] = form["candidate_score_a"]
    form["adjudication_reason"] = "Resolved from rubric and evidence."
    form["confidence"] = "high"
    form.to_csv(config.adjudication_file, index=False)

    final = summarize_human_adjudication(config)
    assert final["judge_validation_adjudication_status"].iloc[0]["status"] == "complete"
    assert "three_judge_panel_vs_human_final" in set(
        final["judge_validation_final_agreement"]["scope"]
    )
    assert final["judge_validation_final_scored_rows"]["adjudicated_score"].notna().sum() == 1
