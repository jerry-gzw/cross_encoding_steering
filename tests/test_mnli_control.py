from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cross_encoding_steering.mnli_control import (
    MnliControlConfig,
    _candidate_values,
    build_mnli_attribution_statistics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exhaustive_config_contains_all_six_letter_mappings_and_five_randoms():
    config = MnliControlConfig.from_json(
        ROOT / "configs/mnli_exhaustive_attribution.example.json",
        project_root=ROOT,
    )
    assignments = {
        tuple(_candidate_values(interface)[label] for label in ("entailment", "neutral", "contradiction"))
        for interface in config.interfaces
    }
    assert len(assignments) == 6
    assert config.attribution_analysis
    assert config.random_seeds == (13, 29, 47, 71, 101)
    assert not config.include_inverse_control


def test_attribution_statistics_random_adjust_and_cluster_by_premise():
    config = MnliControlConfig.from_json(
        ROOT / "configs/mnli_exhaustive_attribution.example.json",
        project_root=ROOT,
    )
    config = replace(config, n_boot=200, bootstrap_chunk_size=50)
    rows = []
    for model in ("model_a", "model_b"):
        for contrast in ("c1", "c2"):
            for group_index in range(4):
                for interface in ("m0", "m1"):
                    for template in ("t0", "t1"):
                        for mode, current, identifier, advantage in (
                            ("raw_mnli_direction", 1.0, 3.0, 2.0),
                            ("random_direction_control_seed_13", 0.1, 0.2, 0.1),
                            ("random_direction_control_seed_29", 0.1, 0.2, 0.1),
                        ):
                            rows.append(
                                {
                                    "model_alias": model,
                                    "model_name": model,
                                    "group_id": f"g{group_index}",
                                    "pair_id": f"{contrast}:g{group_index}",
                                    "pair_type": contrast,
                                    "mode": mode,
                                    "interface": interface,
                                    "template": template,
                                    "current_label_effect": current,
                                    "extraction_id_effect": identifier,
                                    "id_advantage": advantage,
                                }
                            )
    baseline = pd.DataFrame(
        [
            {
                "model_alias": model,
                "model_name": model,
                "group_id": f"g{group_index}",
                "item_id": f"{model}:g{group_index}",
                "interface": interface,
                "template": template,
                "correct": True,
            }
            for model in ("model_a", "model_b")
            for group_index in range(4)
            for interface in ("m0", "m1")
            for template in ("t0", "t1")
        ]
    )
    tables = build_mnli_attribution_statistics(pd.DataFrame(rows), baseline, config)
    decision = tables["mnli_attribution_decision"].set_index("interface")
    assert np.isclose(decision.loc["m0", "current_label_effect"], 0.9)
    assert np.isclose(decision.loc["m0", "extraction_id_effect"], 2.8)
    assert np.isclose(decision.loc["m0", "id_advantage"], 1.9)
    assert decision.loc["m0", "profile"] == "extraction_id_dominant"
    baseline_ci = tables["mnli_baseline_accuracy_group_cluster_ci"]
    assert len(baseline_ci) == 4
    assert baseline_ci["estimate"].eq(1.0).all()
