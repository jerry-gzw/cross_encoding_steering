from __future__ import annotations

import pandas as pd
import numpy as np

from cross_interface_steering.readout_vocabulary_transfer import (
    CORE_MODES,
    _wide_transfer_units,
)


def test_wide_transfer_units_pairs_all_three_components() -> None:
    rows = []
    for mode, scale in zip(CORE_MODES, (1.0, 0.8, 0.2)):
        rows.append(
            {
                "model_alias": "model",
                "model_name": "Model",
                "pair_type": "taboo_vs_expected",
                "pair_id": "pair-1",
                "group_key": "setting-behavior",
                "semantic_mapping": "permutation",
                "identifier_set": "letters_xyz",
                "mode": mode,
                "semantic_margin_gain": 0.1 * scale,
                "extraction_identifier_margin_gain": 1.0 * scale,
                "extraction_row_margin_gain": 0.05 * scale,
            }
        )
    wide = _wide_transfer_units(pd.DataFrame(rows))
    assert len(wide) == 1
    assert np.isclose(wide.loc[0, "raw_id_advantage"], 0.9)
    assert np.isclose(
        wide.loc[0, "projection_extraction_identifier_margin_gain"], 0.8
    )
    assert np.isclose(wide.loc[0, "projection_minus_residual_id"], 0.6)
