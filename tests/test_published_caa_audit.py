from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import cross_interface_steering.published_caa_audit as published_caa_audit

from cross_interface_steering.published_caa_audit import (
    CAAJudgeConfig,
    PublishedCAAAuditConfig,
    _collect_multi_judge_tables,
    _gemini_batch_json_schema,
    _gemini_batch_judge_request,
    _gemini_batch_prompt,
    _reason_score_contradiction,
    _judge_json_schema,
    _judge_prompt,
    _read_csv_if_nonempty,
    aggregate_published_caa_audit,
    _interface_record,
    _interface_summaries,
    _opaque_competence_records,
    _swap_ab_question,
    parse_ab_question,
    prepare_published_caa_audit,
)


def test_gemini_batch_schema_and_response_are_item_aligned(monkeypatch) -> None:
    judge = CAAJudgeConfig(
        alias="gemini",
        provider="gemini",
        model="gemini-test",
        include_reason=True,
        request_batch_size=2,
    )
    rows = [
        {"behavior": "refusal", "question": "Question one", "response": "Answer one"},
        {
            "behavior": "hallucination",
            "question": "Question two",
            "response": "Answer two",
        },
    ]
    schema = _gemini_batch_json_schema(True)
    assert schema["type"] == "array"
    assert schema["items"]["required"] == ["request_id", "reason", "score"]
    prompt = _gemini_batch_prompt(judge, rows)
    assert '"request_id": 0' in prompt and '"request_id": 1' in prompt

    def fake_call(client, config, batch):
        assert batch == rows
        return (
            json.dumps(
                [
                    {"request_id": 0, "reason": "First", "score": 2},
                    {"request_id": 1, "reason": "Second", "score": 8},
                ]
            ),
            {
                "judge_prompt_tokens": 100,
                "judge_completion_tokens": 20,
                "judge_reasoning_tokens": 10,
                "judge_total_tokens": 130,
                "judge_response_model": "gemini-test",
                "judge_system_fingerprint": "",
            },
        )

    monkeypatch.setattr(published_caa_audit, "_gemini_batch_judge_call", fake_call)
    results = _gemini_batch_judge_request(object(), judge, rows)
    assert [result["judge_score"] for result in results] == [2.0, 8.0]
    assert all(result["judge_batch_size"] == 2 for result in results)
    assert sum(result["judge_total_tokens"] for result in results) == 130


def test_provider_specific_judge_schemas() -> None:
    generic = _judge_json_schema(True)
    assert generic["properties"]["score"]["minimum"] == 0
    assert generic["properties"]["score"]["maximum"] == 10
    anthropic = _judge_json_schema(True, provider="anthropic")
    assert anthropic["properties"]["score"]["type"] == "integer"
    assert "minimum" not in anthropic["properties"]["score"]
    assert "maximum" not in anthropic["properties"]["score"]
    assert anthropic["required"] == ["reason", "score"]


def test_anchored_judge_prompt_uses_full_scale_without_zero_example() -> None:
    judge = CAAJudgeConfig(include_reason=True)
    prompt = _judge_prompt(
        judge,
        {
            "behavior": "refusal",
            "question": "What is your social security number?",
            "response": "I do not have one.",
        },
    )
    assert "0:" in prompt and "5:" in prompt and "10:" in prompt
    assert "not by itself a refusal" in prompt
    assert '{"score": 0' not in prompt
    assert "caa_open_v3_calibrated" in prompt
    assert prompt.index('field "reason"') < prompt.index('field "score"')


def test_reason_score_contradiction_catches_explicit_mismatch() -> None:
    reason = "The response rejects the false premise. Therefore, the correct score is 0."
    assert _reason_score_contradiction(10, reason)
    assert not _reason_score_contradiction(0, reason)


def test_multi_judge_aggregation_builds_consensus_and_leave_one_out(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = []
    for item_index in range(3):
        for multiplier in (-2.0, 0.0, 2.0):
            base_rows.append(
                {
                    "model_alias": "llama2_7b_chat",
                    "model_name": "meta-llama/Llama-2-7b-chat-hf",
                    "behavior": "hallucination",
                    "item_id": f"item-{item_index}",
                    "layer": 13,
                    "multiplier": multiplier,
                    "judge_score": 5.0 + multiplier,
                    "judge_error": "",
                }
            )
    for alias, offset, filename in (
        ("gpt5_1", 0.0, "published_caa_open_ended_judgments.csv"),
        ("deepseek_v4_pro", 0.2, "published_caa_open_ended_judgments__deepseek_v4_pro.csv"),
        ("gemini_3_1_pro", -0.2, "published_caa_open_ended_judgments__gemini_3_1_pro.csv"),
    ):
        frame = pd.DataFrame(base_rows)
        frame["judge_score"] += offset
        frame["judge_alias"] = alias
        frame["judge_provider"] = alias.split("_")[0]
        frame["judge_model"] = alias
        frame.to_csv(config.output_dir / filename, index=False)

    tables = _collect_multi_judge_tables(config)
    assert len(tables["published_caa_multi_judge_agreement"]) == 3
    assert tables["published_caa_multi_judge_completeness"]["is_complete"].all()
    aliases = set(tables["published_caa_multi_judge_paired_statistics"]["judge_alias"])
    assert "median_consensus" in aliases
    assert "consensus_without_gpt5_1" in aliases
    assert "consensus_without_deepseek_v4_pro" in aliases
    assert "consensus_without_gemini_3_1_pro" in aliases


def test_configured_multi_judge_aggregation_ignores_historical_files(tmp_path: Path) -> None:
    base = _config(tmp_path)
    judges = tuple(
        CAAJudgeConfig(
            alias=alias,
            provider=provider,
            enabled=True,
            model=model,
            output_filename=filename,
        )
        for alias, provider, model, filename in (
            ("gpt_v2", "openai_compatible", "gpt-5.1", "gpt_v2.csv"),
            ("deepseek_v2", "openai_compatible", "deepseek-v4-pro", "deepseek_v2.csv"),
            ("claude_v2", "anthropic", "claude-sonnet-4-5-20250929", "claude_v2.csv"),
        )
    )
    config = replace(base, judges=judges)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item_index in range(2):
        for multiplier in (-2.0, 0.0, 2.0):
            rows.append(
                {
                    "model_alias": "llama2_7b_chat",
                    "model_name": "meta-llama/Llama-2-7b-chat-hf",
                    "behavior": "hallucination",
                    "item_id": f"item-{item_index}",
                    "layer": 13,
                    "multiplier": multiplier,
                    "judge_score": 5.0 + multiplier,
                    "judge_error": "",
                    "judge_rubric_version": "caa_open_v3_calibrated",
                }
            )
    for judge in judges:
        frame = pd.DataFrame(rows)
        frame["judge_alias"] = judge.alias
        frame["judge_provider"] = judge.provider
        frame["judge_model"] = judge.model
        frame.to_csv(config.output_dir / judge.output_filename, index=False)
    historical = pd.DataFrame(rows)
    historical["judge_alias"] = "historical_should_not_be_read"
    historical.to_csv(
        config.output_dir / "published_caa_open_ended_judgments.csv", index=False
    )

    tables = _collect_multi_judge_tables(config)
    aliases = set(tables["published_caa_multi_judge_scores"]["judge_alias"])
    assert aliases == {judge.alias for judge in judges}
    quality = tables["published_caa_multi_judge_quality"]
    assert set(quality["judge_alias"]) == {judge.alias for judge in judges}
    assert quality["coverage_rate"].eq(1.0).all()
    consensus = tables["published_caa_multi_judge_effects"]
    consensus = consensus[consensus["judge_alias"].eq("median_consensus")]
    assert len(consensus) == len(rows)


def _write_reference_dataset(root: Path, behavior: str) -> None:
    generate_dir = root / "datasets" / "generate" / behavior
    test_dir = root / "datasets" / "test" / behavior
    generate_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    ab = [
        {
            "question": "A scenario with a prose reference to (A).\nChoices:\n(A) agree\n(B) disagree",
            "answer_matching_behavior": "(B)",
            "answer_not_matching_behavior": "(A)",
        }
    ]
    (generate_dir / "generate_dataset.json").write_text(json.dumps(ab), encoding="utf-8")
    (test_dir / "test_dataset_ab.json").write_text(json.dumps(ab), encoding="utf-8")
    (test_dir / "test_dataset_open_ended.json").write_text(
        json.dumps([{"question": "A scenario."}]), encoding="utf-8"
    )


def _config(tmp_path: Path) -> PublishedCAAAuditConfig:
    reference = tmp_path / "datasets" / "CAA"
    for behavior in ["sycophancy", "hallucination", "refusal"]:
        _write_reference_dataset(reference, behavior)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "output_dir": "outputs/audit",
                "protocol": {
                    "reference_repo": "datasets/CAA",
                    "behaviors": ["sycophancy", "hallucination", "refusal"],
                    "normalization_behaviors": ["sycophancy", "hallucination", "refusal"],
                },
                "models": [
                    {
                        "alias": "llama2_7b_chat",
                        "name": "meta-llama/Llama-2-7b-chat-hf",
                        "layer": 13,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return PublishedCAAAuditConfig.from_json(config_path, project_root=tmp_path)


def test_parse_and_swap_only_final_ab_pair() -> None:
    question = "The speaker mentioned (A) earlier.\nChoices:\n(A) first answer\n(B) second answer"
    stem, answer_a, answer_b = parse_ab_question(question)
    assert stem == "The speaker mentioned (A) earlier.\nChoices:"
    assert answer_a == "first answer"
    assert answer_b == "second answer"
    swapped = _swap_ab_question(question)
    assert "mentioned (A) earlier" in swapped
    assert swapped.endswith("(A) second answer\n (B) first answer")


def test_interface_record_tracks_semantic_target_after_swap() -> None:
    item = {
        "question": "Question?\n(A) safe answer\n(B) risky answer",
        "answer_matching_behavior": "(B)",
        "answer_not_matching_behavior": "(A)",
    }
    original = _interface_record(item, interface="original_ab", codewords=("dax", "blicket"))
    swapped = _interface_record(item, interface="swapped_ab", codewords=("dax", "blicket"))
    direct = _interface_record(item, interface="direct_answer_text", codewords=("dax", "blicket"))
    assert original[2:] == (1, 0, "Question?")
    assert swapped[2:] == (0, 1, "Question?")
    assert direct[1] == [" safe answer", " risky answer"]
    assert direct[2:4] == (1, 0)


def test_opaque_competence_records_query_both_actual_answers() -> None:
    items = [{
        "_source_index": 4,
        "question": "Question?\n(A) safe answer\n(B) risky answer",
    }]
    records = _opaque_competence_records(
        items,
        behavior="refusal",
        codewords=("dax", "blicket"),
    )
    assert len(records) == 4
    assert [record["target_index"] for record in records] == [0, 1, 1, 0]
    assert [record["target_codeword"] for record in records] == [
        "dax", "dax", "blicket", "blicket"
    ]
    assert {record["choice_order"] for record in records} == {
        "dax,blicket", "blicket,dax"
    }
    assert all("Temporary answer key" in record["prompt"] for record in records)


def test_prepare_validates_public_caa_layout(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tables = prepare_published_caa_audit(config)
    inventory = tables["published_caa_dataset_inventory"]
    assert len(inventory) == 9
    assert inventory["exists"].all()
    assert inventory["schema_ok"].all()


def test_interface_summary_uses_original_ab_as_retention_reference() -> None:
    rows = []
    for interface, values in {"original_ab": [0.2, 0.4], "swapped_ab": [0.1, 0.2]}.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "model_alias": "m",
                    "model_name": "model",
                    "behavior": "sycophancy",
                    "interface": interface,
                    "layer": 13,
                    "multiplier": 1.0,
                    "item_id": str(index),
                    "base_margin": 0.0,
                    "steered_margin": value,
                    "delta_margin": value,
                }
            )
    summary, retention = _interface_summaries(pd.DataFrame(rows), seed=3)
    assert len(summary) == 2
    swapped = retention.loc[retention["interface"].eq("swapped_ab")].iloc[0]
    assert np.isclose(swapped["retention_ratio"], 0.5)


def test_bare_caa_answer_tokens_are_not_sentencepiece_word_initials() -> None:
    class Tokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, value: str) -> int:
            return {"A": 7, "B": 8, "▁A": 17, "▁B": 18}.get(value, 0)

        def encode(self, value: str, add_special_tokens: bool = False) -> list[int]:
            return [{"A": 17, "B": 18}[value]]

    tokenizer = Tokenizer()
    assert tokenizer.convert_tokens_to_ids("A") == 7
    assert tokenizer.encode("A", add_special_tokens=False) == [17]


def test_headerless_judge_placeholder_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "judgments.csv"
    path.touch()
    assert _read_csv_if_nonempty(path).empty


def test_aggregate_builds_behavior_and_matched_item_correlations(tmp_path: Path) -> None:
    config = _config(tmp_path)
    behavior_dir = config.output_dir / "llama2_7b_chat" / "sycophancy"
    behavior_dir.mkdir(parents=True, exist_ok=True)
    interface_rows = []
    generation_rows = []
    judgment_rows = []
    for index, effect in enumerate([0.1, 0.2, 0.4]):
        interface_rows.append(
            {
                "model_alias": "llama2_7b_chat",
                "model_name": "meta-llama/Llama-2-7b-chat-hf",
                "behavior": "sycophancy",
                "item_id": f"sycophancy:ab:{index}",
                "interface": "original_ab",
                "layer": 13,
                "multiplier": 1.0,
                "base_margin": 0.0,
                "steered_margin": effect,
                "delta_margin": effect,
            }
        )
        for multiplier, score in [(0.0, 2.0), (2.0, 2.0 + effect * 10)]:
            row = {
                "model_alias": "llama2_7b_chat",
                "model_name": "meta-llama/Llama-2-7b-chat-hf",
                "behavior": "sycophancy",
                "item_id": f"sycophancy:open:{index}",
                "source_ab_item_id": f"sycophancy:ab:{index}",
                "layer": 13,
                "multiplier": multiplier,
                "question": "question",
                "response": "response",
            }
            generation_rows.append(row)
            judgment_rows.append(dict(row, judge_score=score, judge_reason="", judge_raw="", judge_error=""))
    pd.DataFrame(interface_rows).to_csv(
        behavior_dir / "published_caa_interface_effects.csv", index=False
    )
    pd.DataFrame(generation_rows).to_csv(
        behavior_dir / "published_caa_open_ended_generations.csv", index=False
    )
    pd.DataFrame(judgment_rows).to_csv(
        config.output_dir / "published_caa_open_ended_judgments.csv", index=False
    )
    pd.DataFrame([
        {
            "model_alias": "llama2_7b_chat",
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "behavior": "sycophancy",
            "item_id": "sycophancy:ab:0:key:0",
            "source_item_id": "sycophancy:ab:0",
            "queried_answer": "agree",
            "target_codeword": "dax",
            "predicted_codeword": "dax",
            "target_choice": "A",
            "predicted_choice": "A",
            "choice_order": "dax,blicket",
            "scoring_protocol": "counterbalanced_letter_choice_v2",
            "target_logprob": -0.1,
            "distractor_logprob": -2.0,
            "target_margin": 1.9,
            "correct": True,
        }
    ]).to_csv(
        behavior_dir / "published_caa_opaque_competence.csv", index=False
    )
    tables = aggregate_published_caa_audit(config)
    correlations = tables["published_caa_cross_evaluation_correlation"]
    matched = correlations.loc[correlations["level"].eq("matched_item")].iloc[0]
    assert matched["n_units"] == 3
    assert np.isclose(matched["pearson_r"], 1.0)

    completeness = tables["published_caa_judgment_completeness"].iloc[0]
    assert bool(completeness["is_complete"])
    assert completeness["n_valid_judgments"] == 6

    paired = tables["published_caa_open_ended_paired_bootstrap_ci"]
    model_effect = paired[
        paired["scope"].eq("per_model")
        & paired["model_alias"].eq("llama2_7b_chat")
        & paired["behavior"].eq("sycophancy")
        & paired["multiplier"].eq(2.0)
    ].iloc[0]
    assert np.isclose(model_effect["mean_delta_judge_score"], 7.0 / 3.0)
    assert model_effect["ci_low"] > 0
    assert bool(model_effect["ci_supports_expected_direction"])

    comparison = tables["published_caa_mcq_open_ended_comparison"].iloc[0]
    assert bool(comparison["mcq_positive_supported"])
    assert bool(comparison["open_positive_supported"])
    assert comparison["evidence_signature"] == "positive_in_both"

    competence = tables["published_caa_opaque_competence_summary"]
    behavior_competence = competence[competence["behavior"].eq("sycophancy")].iloc[0]
    assert bool(behavior_competence["all_queries_correct"])
    assert np.isclose(behavior_competence["accuracy"], 1.0)
    assert np.isclose(behavior_competence["target_a_accuracy"], 1.0)
