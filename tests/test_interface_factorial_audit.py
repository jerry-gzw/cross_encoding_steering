from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cross_encoding_steering.cross_interface_audit import PromptTemplate
from cross_encoding_steering.interface_factorial_audit import (
    FactorialCondition,
    IdentifierSet,
    InterfaceFactorialConfig,
    RowOrder,
    _attribution_labels,
    _macro_f1,
    _prompt_and_candidates,
)
from cross_encoding_steering.mapping_audit import (
    FixedDirectionMappingAuditConfig,
    MappingAuditRuntimeConfig,
    MappingAuditStatisticsConfig,
    MappingDefinition,
    PromptField,
    RankedChoiceDatasetConfig,
)


def _config() -> InterfaceFactorialConfig:
    labels = {"taboo": 0.0, "normal": 1.0, "expected": 2.0}
    canonical = MappingDefinition(
        name="canonical",
        option_order=("taboo", "normal", "expected"),
        option_texts={label: label for label in labels},
    )
    audit = FixedDirectionMappingAuditConfig(
        project_root=Path("."),
        output_dir=Path("."),
        dataset=RankedChoiceDatasetConfig(
            dataset_name="normbank",
            items_path=Path("items.csv"),
            pair_id_column="pair_id",
            contrast_column="pair_type",
            split_column="split",
            label_column="label",
            label_ranks=labels,
            canonical_mapping="canonical",
            prompt_header="Header",
            prompt_fields=(PromptField(column="text", prefix="Scenario: "),),
            question="Question?",
            answer_instruction="Answer.",
        ),
        mappings=(canonical,),
        models=(),
        subspace_dim=1,
        extraction_position="pre_answer",
        injection_position="pre_answer",
        enabled_modes=("raw_canonical_direction",),
        statistics=MappingAuditStatisticsConfig(),
        runtime=MappingAuditRuntimeConfig(),
    )
    return InterfaceFactorialConfig(
        audit=audit,
        templates=(PromptTemplate("primary", "Question?", "Use the key."),),
        mapping_names=("canonical",),
        identifier_sets=(IdentifierSet("xyz", ("X", "Y", "Z")),),
        row_orders=(RowOrder("reverse", (2, 1, 0)),),
        key_templates=("Which identifier means {label}?",),
        competence_threshold=0.8,
    )


def test_factorial_prompt_keeps_identifier_assignment_when_rows_move() -> None:
    config = _config()
    condition = FactorialCondition(
        config.audit.mappings[0],
        config.identifier_sets[0],
        config.row_orders[0],
    )
    prompt, candidates = _prompt_and_candidates(
        pd.Series({"text": "Example"}),
        config,
        config.templates[0],
        condition,
    )
    assert prompt.index("Z. expected") < prompt.index("X. taboo")
    assert candidates == {"taboo": "X", "normal": "Y", "expected": "Z"}


def test_attribution_separates_semantics_identifier_and_row() -> None:
    config = _config()
    remapped = MappingDefinition(
        name="remapped",
        option_order=("expected", "taboo", "normal"),
        option_texts={label: label for label in config.audit.dataset.label_ranks},
    )
    condition = FactorialCondition(
        remapped,
        IdentifierSet("abc", ("A", "B", "C")),
        RowOrder("order_231", (1, 2, 0)),
    )
    labels = _attribution_labels(
        condition,
        config.audit.mappings[0],
        target_label="expected",
        source_label="taboo",
    )
    assert labels["semantic"] == ("expected", "taboo")
    assert labels["extraction_identifier"] == ("normal", "expected")
    assert labels["extraction_row"] == ("expected", "taboo")


def test_macro_f1_is_computed_over_all_labels() -> None:
    value = _macro_f1(
        ["taboo", "normal", "expected"],
        ["taboo", "taboo", "expected"],
        ["taboo", "normal", "expected"],
    )
    assert np.isclose(value, (2.0 / 3.0 + 0.0 + 1.0) / 3.0)
