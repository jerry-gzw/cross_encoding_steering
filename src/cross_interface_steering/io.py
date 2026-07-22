from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path, *, fmt: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    table_path = Path(path).expanduser()
    suffix = (fmt or table_path.suffix.lstrip(".")).lower()
    if suffix == "csv":
        return pd.read_csv(table_path, nrows=nrows)
    if suffix == "tsv":
        return pd.read_csv(table_path, sep="\t", nrows=nrows)
    if suffix in {"jsonl", "ndjson"}:
        frame = pd.read_json(table_path, lines=True)
        return frame.head(nrows) if nrows else frame
    if suffix == "json":
        try:
            frame = pd.read_json(table_path)
        except ValueError:
            frame = pd.read_json(table_path, lines=True)
        return frame.head(nrows) if nrows else frame
    if suffix == "parquet":
        frame = pd.read_parquet(table_path)
        return frame.head(nrows) if nrows else frame
    raise ValueError(f"Unsupported table format for {table_path}")


def write_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for name, frame in tables.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        saved[name] = path
    return saved


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
