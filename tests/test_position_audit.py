import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_interface_steering.mapping_audit import (
    MappingDefinition,
    PromptField,
    RankedChoiceDatasetConfig,
    build_mapping_items,
)
from cross_interface_steering.position_audit import (
    ExtractionPositionAuditConfig,
    build_position_comparisons,
)
from cross_interface_steering.steering import token_indices_from_character_offsets


def _dataset(tmp_path: Path) -> RankedChoiceDatasetConfig:
    return RankedChoiceDatasetConfig(
        dataset_name="ranked",
        items_path=tmp_path / "items.csv",
        pair_id_column="pair_id",
        contrast_column="pair_type",
        split_column="split",
        label_column="label",
        label_ranks={"low": 0.0, "high": 1.0},
        canonical_mapping="canonical",
        prompt_header="Classify.",
        prompt_fields=(PromptField("scenario", "Scenario: "),),
        question="Question?",
        answer_instruction="Answer A or B.",
    )


def test_scenario_boundary_is_inside_the_same_full_prompt(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    mapping = MappingDefinition(
        name="canonical",
        option_order=("low", "high"),
        option_texts={"low": "low", "high": "high"},
    )
    base = pd.DataFrame(
        [{"pair_id": "p", "pair_type": "contrast", "split": "train", "label": "low", "scenario": "one", "semantic_rank": 0.0}]
    )

    item = build_mapping_items(base, dataset, mapping, mapping).iloc[0]
    boundary = int(item["scenario_char_end"])

    assert item["prompt"][:boundary].endswith("Scenario: one")
    assert item["prompt"][boundary:].startswith("\nQuestion?")
    assert item["prompt"].endswith("Answer:")


def test_character_boundaries_resolve_without_using_padding() -> None:
    offsets = np.asarray(
        [
            [[0, 0], [0, 4], [5, 8], [9, 15], [0, 0]],
            [[0, 0], [0, 0], [0, 3], [4, 7], [8, 10]],
        ]
    )
    attention = np.asarray([[1, 1, 1, 1, 0], [0, 0, 1, 1, 1]])

    positions = token_indices_from_character_offsets(offsets, attention, [8, None])

    assert positions.tolist() == [2, 4]


def test_position_comparison_is_scenario_minus_pre_answer() -> None:
    rows = []
    for pair_id, pair_type, scenario_value, answer_value in [
        ("p1", "c1", 0.4, -0.2),
        ("p2", "c2", 0.2, -0.1),
    ]:
        for position, value in [("scenario_end", scenario_value), ("pre_answer", answer_value)]:
            rows.append(
                {
                    "mapping_name": "reversed",
                    "mapping_family": "semantic_words",
                    "pair_id": pair_id,
                    "pair_type": pair_type,
                    "mode": "raw_canonical_direction",
                    "extraction_position": position,
                    "semantic_accuracy_gain": value,
                    "semantic_prob_gain": value,
                    "original_letter_accuracy_gain": -value,
                    "original_letter_prob_gain": -value,
                    "semantic_minus_original_prob_gain_moved": 2 * value,
                    "semantic_minus_original_accuracy_gain_moved": 2 * value,
                    "js_shift": abs(value),
                }
            )

    comparisons = build_position_comparisons(
        pd.DataFrame(rows),
        n_boot=100,
        n_permutations=100,
        confidence=0.95,
        seed=13,
    )
    row = comparisons[comparisons["metric"].eq("semantic_prob_gain")].iloc[0]

    assert np.isclose(row["scenario_end_mean"], 0.3)
    assert np.isclose(row["pre_answer_mean"], -0.15)
    assert np.isclose(row["scenario_end_minus_pre_answer"], 0.45)


def test_position_config_requires_both_positions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "extraction_positions": ["scenario_end"],
                "dataset": {
                    "dataset_name": "ranked",
                    "items_path": "items.csv",
                    "pair_id_column": "pair_id",
                    "contrast_column": "pair_type",
                    "split_column": "split",
                    "label_column": "label",
                    "label_ranks": {"low": 0, "high": 1},
                    "canonical_mapping": "canonical",
                    "prompt_header": "Classify.",
                    "prompt_fields": [{"column": "text", "prefix": "Text: "}],
                    "question": "Question?",
                    "answer_instruction": "Answer A or B."
                },
                "mappings": [
                    {"name": "canonical", "option_order": ["low", "high"], "option_texts": {"low": "low", "high": "high"}}
                ],
                "models": [
                    {"alias": "model", "name": "org/model", "locked_layer": 7, "locked_alpha": 0.8}
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        ExtractionPositionAuditConfig.from_json(config_path, project_root=tmp_path)
    except ValueError as exc:
        assert "requires both" in str(exc)
    else:
        raise AssertionError("Expected a paired-position configuration error")
