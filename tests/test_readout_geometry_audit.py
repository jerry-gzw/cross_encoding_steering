import json
from pathlib import Path

import numpy as np

from cross_encoding_steering.readout_geometry_audit import (
    ReadoutGeometryAuditConfig,
    decompose_against_subspace,
    estimate_readout_subspace,
)


def test_readout_decomposition_reconstructs_and_orthogonalizes() -> None:
    vector = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
    basis = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)

    parts = decompose_against_subspace(vector, basis)

    assert np.allclose(parts["projection"] + parts["residual"], vector)
    assert np.isclose(np.dot(parts["projection"], parts["residual"]), 0.0)
    assert np.isclose(np.linalg.norm(parts["projection_norm_matched"]), np.linalg.norm(vector))
    assert np.isclose(np.linalg.norm(parts["residual_norm_matched"]), np.linalg.norm(vector))
    assert parts["reconstruction_error"] < 1e-7


def test_gradient_svd_returns_orthonormal_basis() -> None:
    gradients = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    basis, inventory = estimate_readout_subspace(
        gradients,
        rank=2,
        normalize_gradients=True,
    )

    assert basis.shape == (3, 2)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-6)
    assert inventory["selected"].sum() == 2


def test_readout_config_inherits_mapping_audit(tmp_path: Path) -> None:
    items_path = tmp_path / "items.csv"
    items_path.write_text("pair_id,pair_type,split,label,text\n", encoding="utf-8")
    base_path = tmp_path / "base.json"
    base_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "dataset_name": "ranked",
                    "items_path": str(items_path),
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
                    {
                        "name": "canonical",
                        "option_order": ["low", "high"],
                        "option_texts": {"low": "low", "high": "high"}
                    }
                ],
                "models": [
                    {
                        "alias": "model",
                        "name": "org/model",
                        "locked_layer": 7,
                        "locked_alpha": 0.8
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "audit.json"
    config_path.write_text(
        json.dumps(
            {
                "base_config": "base.json",
                "readout_geometry": {
                    "subspace_rank": 2,
                    "max_train_prompts_per_contrast": 12,
                    "gradient_batch_size": 1,
                    "random_control_seeds": [13, 29]
                }
            }
        ),
        encoding="utf-8",
    )

    config = ReadoutGeometryAuditConfig.from_json(
        config_path,
        project_root=tmp_path,
    )

    assert config.settings.subspace_rank == 2
    assert config.settings.random_control_seeds == (13, 29)
    assert config.mapping_audit.dataset.dataset_name == "ranked"
