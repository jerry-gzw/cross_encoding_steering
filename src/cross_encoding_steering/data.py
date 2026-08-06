from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .io import first_existing, read_table
from .pairs import apply_generated_split, build_grouped_label_pairs
from .text import clean_text


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


def build_sc101_pairs(
    config: DatasetConfig,
    project_root: Path,
    *,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.adapter != "social_chemistry_101":
        raise ValueError(
            "The SC101 adapter requires adapter='social_chemistry_101'; "
            f"received adapter={config.adapter!r}."
        )

    source_path, frame = load_dataset(config, project_root)
    rows = frame.copy()
    if config.include_splits and "split" in rows.columns:
        rows = rows[rows["split"].astype(str).isin(config.include_splits)].copy()
    if "rot-bad" in rows.columns:
        rows = rows[
            pd.to_numeric(rows["rot-bad"], errors="coerce").fillna(1).eq(0)
        ].copy()
    if (
        "rot-categorization" in rows.columns
        and config.metadata.get("social_norms_only", True)
    ):
        rows = rows[
            rows["rot-categorization"]
            .astype(str)
            .str.contains("social-norms", na=False)
        ].copy()

    def collapse_label(value: Any) -> str | None:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(number):
            return None
        number = int(number)
        if number in {-2, -1}:
            return "bad"
        if number == 0:
            return "ok"
        if number in {1, 2}:
            return "good"
        return None

    rows["collapsed_label"] = rows["action-moral-judgment"].map(collapse_label)
    rows = rows[rows["collapsed_label"].notna()].copy()
    for column in ("action", "situation", "rot", "area"):
        if column not in rows.columns:
            rows[column] = ""
        rows[column] = rows[column].map(clean_text)

    rows["sc101_text"] = rows["action"]
    if config.metadata.get("prompt_variant", "action_only") == "full":
        rows["sc101_text"] = (
            "Situation: "
            + rows["situation"]
            + " Action: "
            + rows["action"]
            + " Rule: "
            + rows["rot"]
        )
    rows["sc101_group"] = rows["situation"].where(
        rows["situation"].str.len().gt(0),
        rows["area"],
    )

    items, pairs = build_grouped_label_pairs(
        rows,
        dataset_name=config.name,
        text_fields=["sc101_text"],
        label_field="collapsed_label",
        label_order=["bad", "ok", "good"],
        group_fields=["sc101_group"],
        split_field="split" if "split" in rows.columns else None,
        id_field="rot-id" if "rot-id" in rows.columns else None,
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
                "n_filtered_rows": int(len(rows)),
                "n_items": int(len(items)),
                "n_pairs": int(len(pairs)),
                "generated_split": not split_inventory.empty,
                "prompt_variant": config.metadata.get("prompt_variant", "action_only"),
            }
        ]
    )
    return items, pairs, metadata
