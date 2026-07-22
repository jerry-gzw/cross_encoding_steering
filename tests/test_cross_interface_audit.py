from __future__ import annotations

from pathlib import Path

import numpy as np

from cross_interface_steering.cross_interface_audit import (
    CrossInterfaceConfig,
    InterfaceDefinition,
    _mapping_norm_diagnostics,
    _original_slot_labels,
)
from cross_interface_steering.mapping_audit import (
    FixedDirectionMappingAuditConfig,
    MappingDefinition,
    MappingAuditRuntimeConfig,
    MappingAuditStatisticsConfig,
    RankedChoiceDatasetConfig,
)


def test_mapping_average_norm_retention_and_rescale() -> None:
    vectors = [
        np.asarray([3.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 4.0], dtype=np.float32),
    ]
    mean_vector = np.mean(np.stack(vectors), axis=0)
    diagnostics = _mapping_norm_diagnostics(vectors, mean_vector, target_norm=7.0)

    assert np.isclose(diagnostics["mean_mapping_direction_l2"], 3.5)
    assert np.isclose(diagnostics["mapping_mean_norm_retention_ratio"], 2.5 / 3.5)
    assert np.isclose(diagnostics["mapping_balance_rescale_factor"], 7.0 / 2.5)


def test_original_slot_labels_follow_canonical_letters_after_remapping() -> None:
    labels = {"taboo": 0, "normal": 1, "expected": 2}
    canonical = MappingDefinition(
        name="canonical",
        option_order=("taboo", "normal", "expected"),
        option_texts={label: label for label in labels},
    )
    remapped = MappingDefinition(
        name="remapped",
        option_order=("expected", "taboo", "normal"),
        option_texts={label: label for label in labels},
    )
    audit = FixedDirectionMappingAuditConfig(
        project_root=Path("."),
        output_dir=Path("."),
        subspace_dim=1,
        dataset=RankedChoiceDatasetConfig(
            dataset_name="normbank",
            items_path=Path("items.csv"),
            pair_id_column="pair_id",
            contrast_column="pair_type",
            split_column="split",
            label_column="label",
            label_ranks=labels,
            canonical_mapping="canonical",
            prompt_header="",
            prompt_fields=(),
            question="",
            answer_instruction="",
            extraction_filters={},
            train_split="train",
            test_split="test",
            max_train_pairs_per_contrast=1,
            max_test_pairs_per_contrast=1,
        ),
        mappings=(canonical, remapped),
        statistics=MappingAuditStatisticsConfig(),
        runtime=MappingAuditRuntimeConfig(),
        models=(),
        extraction_position="pre_answer",
        injection_position="pre_answer",
        enabled_modes=("raw_canonical_direction",),
    )
    config = CrossInterfaceConfig(
        audit=audit,
        templates=(),
        interfaces=(),
        reference_interface="canonical",
        primary_score="mean_logprob",
        minimum_reference_effect=0.01,
        direction_sources=(),
        mapping_balance_mappings=(),
        random_control_seeds=(),
        key_competence_threshold=0.9,
    )
    interface = InterfaceDefinition(
        name="remapped",
        kind="letter_mcq",
        mapping_name="remapped",
        label_values={label: label for label in labels},
    )

    target_slot_label, source_slot_label = _original_slot_labels(
        config,
        interface,
        target_label="expected",
        source_label="taboo",
    )

    assert target_slot_label == "normal"
    assert source_slot_label == "expected"
