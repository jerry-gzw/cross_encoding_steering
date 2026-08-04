from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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
    filters: list[dict[str, Any]] = field(default_factory=list)
    paired_fields: dict[str, str] = field(default_factory=dict)
    generated_split: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DatasetConfig":
        kwargs = dict(data)
        for key in ["text_fields", "label_order", "group_fields", "include_splits", "filters"]:
            kwargs[key] = list(_as_list(kwargs.get(key)))
        return cls(**kwargs)

    def candidate_paths(self, project_root: Path) -> list[Path]:
        paths = []
        for item in _as_list(self.path):
            path = Path(str(item)).expanduser()
            if not path.is_absolute():
                path = project_root / path
            paths.append(path)
        return paths


@dataclass
class ExperimentConfig:
    project_root: Path
    datasets: list[DatasetConfig]
    output_dir: Path = Path("outputs")
    seed: int = 13

    @classmethod
    def from_json(cls, path: str | Path, *, project_root: str | Path | None = None) -> "ExperimentConfig":
        config_path = Path(path).expanduser()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()
        datasets = [DatasetConfig.from_mapping(item) for item in data.get("datasets", [])]
        output_dir = Path(data.get("output_dir", "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        return cls(
            project_root=root,
            datasets=datasets,
            output_dir=output_dir,
            seed=int(data.get("seed", 13)),
        )

    def dataset(self, name: str) -> DatasetConfig:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        known = ", ".join(sorted(item.name for item in self.datasets))
        raise KeyError(f"Unknown dataset {name!r}. Known datasets: {known}")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "datasets": [dataset.__dict__ for dataset in self.datasets],
        }


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


@dataclass(frozen=True)
class EvaluationConfig:
    dataset_name: str
    pairs_path: Path
    group_column: str
    group_values: tuple[str, ...]
    negative_label: str
    positive_label: str
    layer_fractions: tuple[float, ...] = (0.60, 0.75, 0.90)
    alpha_grid: tuple[float, ...] = (0.2, 0.4, 0.8)
    subspace_dims: tuple[int, ...] = (1, 2)
    mixing_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    prompt_variants: tuple[str, ...] = ("canonical", "label_order_flip")
    eval_directions: tuple[str, ...] = ("negative_to_positive", "positive_to_negative")
    extraction_position: str = "pre_answer"
    include_loco: bool = False
    loco_subspace_dim: int = 1
    max_train_pairs_per_group: int = 256
    max_validation_pairs_per_group: int = 128
    max_test_pairs_per_group: int = 128
    scope: str = "cross_interface_steering"
    report_title: str = "Norm Direction Decomposition Evaluation"


@dataclass(frozen=True)
class StatisticsConfig:
    n_boot: int = 10_000
    n_permutations: int = 20_000
    confidence: float = 0.95
    group_column: str = "pair_type"


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int = 2
    max_length: int = 512
    seed: int = 13
    continue_on_error: bool = True
    force_rerun: bool = False


@dataclass(frozen=True)
class SteeringExperimentConfig:
    project_root: Path
    output_dir: Path
    evaluation: EvaluationConfig
    statistics: StatisticsConfig
    runtime: RuntimeConfig
    models: tuple[ModelConfig, ...]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        model_source_overrides: dict[str, str] | None = None,
    ) -> "SteeringExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        root = Path(project_root or data.get("project_root", ".")).expanduser().resolve()

        evaluation_data = dict(data.get("evaluation", {}))
        pairs_path = Path(str(evaluation_data.pop("pairs_path"))).expanduser()
        if not pairs_path.is_absolute():
            pairs_path = root / pairs_path
        evaluation = EvaluationConfig(
            dataset_name=str(evaluation_data.pop("dataset_name")),
            pairs_path=pairs_path.resolve(),
            group_column=str(evaluation_data.pop("group_column", "pair_type")),
            group_values=tuple(str(value) for value in evaluation_data.pop("group_values", [])),
            negative_label=str(evaluation_data.pop("negative_label")),
            positive_label=str(evaluation_data.pop("positive_label")),
            layer_fractions=tuple(float(value) for value in evaluation_data.pop("layer_fractions", [0.60, 0.75, 0.90])),
            alpha_grid=tuple(float(value) for value in evaluation_data.pop("alpha_grid", [0.2, 0.4, 0.8])),
            subspace_dims=tuple(int(value) for value in evaluation_data.pop("subspace_dims", [1, 2])),
            mixing_grid=tuple(float(value) for value in evaluation_data.pop("mixing_grid", [0, 0.25, 0.5, 0.75, 1])),
            prompt_variants=tuple(str(value) for value in evaluation_data.pop("prompt_variants", ["canonical", "label_order_flip"])),
            eval_directions=tuple(str(value) for value in evaluation_data.pop("eval_directions", ["negative_to_positive", "positive_to_negative"])),
            extraction_position=str(evaluation_data.pop("extraction_position", "pre_answer")),
            include_loco=bool(evaluation_data.pop("include_loco", False)),
            loco_subspace_dim=int(evaluation_data.pop("loco_subspace_dim", 1)),
            max_train_pairs_per_group=int(evaluation_data.pop("max_train_pairs_per_group", 256)),
            max_validation_pairs_per_group=int(evaluation_data.pop("max_validation_pairs_per_group", 128)),
            max_test_pairs_per_group=int(evaluation_data.pop("max_test_pairs_per_group", 128)),
            scope=str(evaluation_data.pop("scope", "cross_interface_steering")),
            report_title=str(evaluation_data.pop("report_title", "Norm Direction Decomposition Evaluation")),
        )
        if evaluation_data:
            raise ValueError(f"Unknown evaluation config keys: {sorted(evaluation_data)}")
        if evaluation.extraction_position not in {"scenario_end", "pre_answer"}:
            raise ValueError(
                "evaluation.extraction_position must be scenario_end or pre_answer"
            )
        if evaluation.include_loco and evaluation.loco_subspace_dim != 1:
            raise ValueError("LOCO controls require loco_subspace_dim=1")

        statistics = StatisticsConfig(**data.get("statistics", {}))
        runtime = RuntimeConfig(**data.get("runtime", {}))
        overrides = model_source_overrides or {}
        models = []
        for item in data.get("models", []):
            model_data = dict(item)
            alias = str(model_data["alias"])
            if alias in overrides:
                model_data["source"] = overrides[alias]
            models.append(ModelConfig.from_mapping(model_data))
        if not models:
            raise ValueError("Steering experiment config must include at least one model")

        output_dir = Path(str(data.get("output_dir", "outputs/experiment"))).expanduser()
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        return cls(
            project_root=root,
            output_dir=output_dir.resolve(),
            evaluation=evaluation,
            statistics=statistics,
            runtime=runtime,
            models=tuple(models),
        )
