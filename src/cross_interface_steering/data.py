from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .io import first_existing, read_table
from .pairs import apply_generated_split, build_grouped_label_pairs


def discover_dataset(config: DatasetConfig, project_root: Path) -> dict[str, Any]:
    paths = config.candidate_paths(project_root)
    existing = [path for path in paths if path.exists()]
    return {
        "dataset": config.name,
        "status": "present" if existing else "missing",
        "found_paths": "; ".join(str(path) for path in existing),
        "candidate_paths": "; ".join(str(path) for path in paths),
    }


def load_dataset(config: DatasetConfig, project_root: Path) -> tuple[Path, pd.DataFrame]:
    path = first_existing(config.candidate_paths(project_root))
    if path is None:
        raise FileNotFoundError(
            f"No local file found for dataset {config.name}: {config.path}"
        )
    return path, read_table(path, fmt=config.format)


def build_normbank_pairs(
    config: DatasetConfig,
    project_root: Path,
    *,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.adapter != "generic_contrastive":
        raise ValueError(
            "The AAAI artifact includes only the generic NormBank pair adapter; "
            f"received adapter={config.adapter!r}."
        )

    source_path, frame = load_dataset(config, project_root)
    items, pairs = build_grouped_label_pairs(
        frame,
        dataset_name=config.name,
        text_fields=config.text_fields,
        label_field=str(config.label_field),
        label_order=config.label_order,
        group_fields=config.group_fields,
        split_field=config.split_field,
        id_field=config.id_field,
        prompt_template=config.prompt_template,
        max_pairs_per_type_split=config.max_pairs_per_type_split,
        include_splits=config.include_splits,
        seed=seed,
    )
    items, pairs, split_inventory = apply_generated_split(
        items,
        pairs,
        config.generated_split,
        seed=seed,
    )
    metadata = pd.DataFrame(
        [
            {
                "dataset": config.name,
                "source_path": str(source_path),
                "n_raw_rows": int(len(frame)),
                "n_items": int(len(items)),
                "n_pairs": int(len(pairs)),
                "generated_split": not split_inventory.empty,
            }
        ]
    )
    return items, pairs, metadata
