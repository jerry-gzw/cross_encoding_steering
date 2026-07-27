from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


@dataclass
class DatasetConfig:
    name: str
    adapter: str
    path: str | list[str]
    format: str | None = None
    text_fields: list[str] = field(default_factory=list)
    label_field: str | None = None
    label_order: list[str] = field(default_factory=list)
    group_fields: list[str] = field(default_factory=list)
    split_field: str | None = None
    id_field: str | None = None
    prompt_template: str | None = None
    max_pairs_per_type_split: int = 2000
    include_splits: list[str] = field(default_factory=list)
    generated_split: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DatasetConfig":
        kwargs = dict(data)
        for key in ("text_fields", "label_order", "group_fields", "include_splits"):
            kwargs[key] = list(_as_list(kwargs.get(key)))
        return cls(**kwargs)

    def candidate_paths(self, project_root: Path) -> list[Path]:
        paths = []
        for item in _as_list(self.path):
            path = Path(str(item)).expanduser()
            paths.append(path if path.is_absolute() else project_root / path)
        return paths


@dataclass
class ExperimentConfig:
    project_root: Path
    datasets: list[DatasetConfig]
    output_dir: Path
    seed: int = 13

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> "ExperimentConfig":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()
        output_dir = Path(data.get("output_dir", "outputs/prepared")).expanduser()
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        return cls(
            project_root=root,
            datasets=[DatasetConfig.from_mapping(item) for item in data.get("datasets", [])],
            output_dir=output_dir.resolve(),
            seed=int(data.get("seed", 13)),
        )

    def dataset(self, name: str) -> DatasetConfig:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        known = ", ".join(sorted(item.name for item in self.datasets))
        raise KeyError(f"Unknown dataset {name!r}. Known datasets: {known}")


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    name: str
    source: str | None = None
    enabled: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"

    @property
    def load_source(self) -> str:
        return self.source or self.name

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            alias=str(data["alias"]),
            name=str(data["name"]),
            source=str(data["source"]) if data.get("source") else None,
            enabled=bool(data.get("enabled", True)),
            device_map=str(data.get("device_map", "auto")),
            torch_dtype=str(data.get("torch_dtype", "auto")),
        )
