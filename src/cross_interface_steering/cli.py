from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .context_discrimination import run_context_discrimination_from_json
from .cross_interface_audit import run_cross_interface_audit_from_json
from .data import build_normbank_pairs, build_sc101_pairs, discover_dataset
from .endpoints import build_ranked_endpoints_from_csv
from .interface_factorial_audit import (
    run_interface_factorial_audit_from_json,
    summarize_interface_factorial_statistics,
)
from .interface_statistics import run_interface_statistics_from_json
from .io import write_tables
from .iti_probe_audit import run_iti_probe_audit_from_json
from .letter_permutation_audit import run_letter_permutation_audit_from_json
from .mnli_control import run_mnli_control_from_json, run_mnli_statistics_from_json
from .pairs import audit_split_isolation, summarize_pairs
from .published_caa_audit import run_published_caa_audit_from_json
from .validity_controls import run_evaluation_validity_controls_from_json


def _model_source_overrides(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    if value.lstrip().startswith("{"):
        text = value
    else:
        candidate = Path(value).expanduser()
        text = candidate.read_text(encoding="utf-8") if candidate.exists() else value
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Model source overrides must be a JSON object")
    return {str(key): str(item) for key, item in parsed.items()}


def _aliases(value: str | None) -> list[str] | None:
    aliases = [item.strip() for item in (value or "").split(",") if item.strip()]
    return aliases or None


def cmd_prepare_normbank(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_json(args.config, project_root=args.project_root)
    dataset = config.dataset("normbank")
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = config.project_root / output_dir
    output_dir = output_dir.resolve()
    items, pairs, metadata = build_normbank_pairs(dataset, config.project_root, seed=config.seed)
    tables = {"items": items, "pairs": pairs, "dataset_metadata": metadata}
    tables.update(summarize_pairs(pairs))
    if "group_key" in pairs.columns:
        tables.update(audit_split_isolation(pairs))
    write_tables(tables, output_dir)
    endpoints = build_ranked_endpoints_from_csv(output_dir / "pairs.csv", output_dir)
    print(endpoints["ranked_endpoint_inventory"].to_string(index=False))
    print(f"Prepared NormBank pairs and endpoints in {output_dir}")


def cmd_prepare_sc101(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_json(args.config, project_root=args.project_root)
    dataset = config.dataset("social_chemistry_101")
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = config.project_root / output_dir
    output_dir = output_dir.resolve()
    items, pairs, metadata = build_sc101_pairs(
        dataset,
        config.project_root,
        seed=config.seed,
    )
    tables = {"items": items, "pairs": pairs, "dataset_metadata": metadata}
    tables.update(summarize_pairs(pairs))
    write_tables(tables, output_dir)
    endpoints = build_ranked_endpoints_from_csv(output_dir / "pairs.csv", output_dir)
    print(endpoints["ranked_endpoint_inventory"].to_string(index=False))
    print(f"Prepared SC101 action-only pairs and endpoints in {output_dir}")


def cmd_check_data(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_json(args.config, project_root=args.project_root)
    rows = [discover_dataset(dataset, config.project_root) for dataset in config.datasets]
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_cross_interface(args: argparse.Namespace) -> None:
    tables = run_cross_interface_audit_from_json(
        args.config,
        project_root=args.project_root,
        model_source_overrides=_model_source_overrides(args.model_source_overrides),
        model_aliases=_aliases(args.models),
        inventory_only=bool(args.inventory_only),
    )
    print(tables.get("cross_interface_summary", pd.DataFrame()).to_string(index=False))


def cmd_letter_permutations(args: argparse.Namespace) -> None:
    tables = run_letter_permutation_audit_from_json(
        args.config,
        project_root=args.project_root,
        model_source_overrides=_model_source_overrides(args.model_source_overrides),
        model_aliases=_aliases(args.models),
        phase=args.phase,
    )
    summary = tables.get("letter_permutation_decision", pd.DataFrame())
    if summary.empty:
        summary = tables.get("audit__cross_interface_summary", pd.DataFrame())
    print(summary.to_string(index=False))


def cmd_context_selectivity(args: argparse.Namespace) -> None:
    tables = run_context_discrimination_from_json(args.config, project_root=args.project_root)
    print(tables.get("context_discrimination_summary", pd.DataFrame()).to_string(index=False))


def cmd_mnli(args: argparse.Namespace) -> None:
    tables = run_mnli_control_from_json(
        args.config,
        project_root=args.project_root,
        model_source_overrides=_model_source_overrides(args.model_source_overrides),
        model_aliases=_aliases(args.models),
    )
    print(tables.get("mnli_interface_summary", pd.DataFrame()).to_string(index=False))


def cmd_mnli_statistics(args: argparse.Namespace) -> None:
    tables = run_mnli_statistics_from_json(args.config, project_root=args.project_root)
    print(tables.get("mnli_global_bootstrap_ci", pd.DataFrame()).to_string(index=False))


def cmd_interface_statistics(args: argparse.Namespace) -> None:
    tables = run_interface_statistics_from_json(args.config, project_root=args.project_root)
    print(tables.get("nuisance_mode_global_bootstrap_ci", pd.DataFrame()).to_string(index=False))


def cmd_interface_factorial(args: argparse.Namespace) -> None:
    tables = run_interface_factorial_audit_from_json(
        args.config,
        project_root=args.project_root,
        model_source_overrides=_model_source_overrides(args.model_source_overrides),
        model_aliases=_aliases(args.models),
        phase=args.phase,
    )
    print(tables.get("factorial_condition_summary", pd.DataFrame()).to_string(index=False))


def cmd_interface_factorial_statistics(args: argparse.Namespace) -> None:
    tables = summarize_interface_factorial_statistics(
        args.config,
        project_root=args.project_root,
    )
    print(tables.get("factorial_paper_model_summary", pd.DataFrame()).to_string(index=False))


def cmd_iti_probe(args: argparse.Namespace) -> None:
    tables = run_iti_probe_audit_from_json(
        args.config,
        project_root=args.project_root,
        model_source_overrides=_model_source_overrides(args.model_source_overrides),
        model_aliases=_aliases(args.models),
        phase=args.phase,
    )
    for key in (
        "iti_cross_method_decision",
        "iti_probe_competence_summary",
        "iti_probe_selection",
    ):
        frame = tables.get(key, pd.DataFrame())
        if not frame.empty:
            print(frame.to_string(index=False))
            break


def cmd_published_caa(args: argparse.Namespace) -> None:
    tables = run_published_caa_audit_from_json(
        args.config,
        project_root=args.project_root,
        model_source_overrides=_model_source_overrides(args.model_source_overrides),
        phase=args.phase,
        model_aliases=_aliases(args.models),
        behaviors=_aliases(args.behaviors),
        judge_aliases=_aliases(args.judges),
    )
    key = "published_caa_opaque_competence_summary" if args.phase == "competence" else "published_caa_interface_retention"
    print(tables.get(key, pd.DataFrame()).to_string(index=False))


def cmd_validity(args: argparse.Namespace) -> None:
    tables = run_evaluation_validity_controls_from_json(
        args.config,
        project_root=args.project_root,
        model_source_overrides=_model_source_overrides(args.model_source_overrides),
        model_aliases=_aliases(args.models),
        phase=args.phase,
    )
    for key in (
        "multi_random_standardized_gain_summary",
        "normbank_key_competence_summary",
        "judge_validation_sampling_inventory",
    ):
        frame = tables.get(key, pd.DataFrame())
        if not frame.empty:
            print(frame.to_string(index=False))
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cross-interface-steering")
    sub = parser.add_subparsers(dest="command", required=True)

    def config_flags(command: argparse.ArgumentParser, *, models: bool = False) -> None:
        command.add_argument("--config", required=True)
        command.add_argument("--project-root", default=None)
        if models:
            command.add_argument("--models", default=None)
            command.add_argument("--model-source-overrides", default=None)

    command = sub.add_parser("check-data")
    config_flags(command)
    command.set_defaults(func=cmd_check_data)

    command = sub.add_parser("prepare-normbank")
    config_flags(command)
    command.add_argument("--output-dir", required=True)
    command.set_defaults(func=cmd_prepare_normbank)

    command = sub.add_parser("prepare-sc101")
    config_flags(command)
    command.add_argument("--output-dir", required=True)
    command.set_defaults(func=cmd_prepare_sc101)

    command = sub.add_parser("run-cross-interface")
    config_flags(command, models=True)
    command.add_argument("--inventory-only", action="store_true")
    command.set_defaults(func=cmd_cross_interface)

    command = sub.add_parser("run-letter-permutations")
    config_flags(command, models=True)
    command.add_argument("--phase", choices=["run", "aggregate", "summarize", "all"], default="all")
    command.set_defaults(func=cmd_letter_permutations)

    command = sub.add_parser("run-context-selectivity")
    config_flags(command)
    command.set_defaults(func=cmd_context_selectivity)

    command = sub.add_parser("run-mnli-control")
    config_flags(command, models=True)
    command.set_defaults(func=cmd_mnli)

    command = sub.add_parser("summarize-mnli")
    config_flags(command)
    command.set_defaults(func=cmd_mnli_statistics)

    command = sub.add_parser("summarize-interface-controls")
    config_flags(command)
    command.set_defaults(func=cmd_interface_statistics)

    command = sub.add_parser("run-interface-factorial")
    config_flags(command, models=True)
    command.add_argument("--phase", choices=["run", "aggregate", "all"], default="all")
    command.set_defaults(func=cmd_interface_factorial)

    command = sub.add_parser("summarize-interface-factorial")
    config_flags(command)
    command.set_defaults(func=cmd_interface_factorial_statistics)

    command = sub.add_parser("run-iti-probe")
    config_flags(command, models=True)
    command.add_argument(
        "--phase",
        choices=["run", "aggregate", "summarize", "all"],
        default="all",
    )
    command.set_defaults(func=cmd_iti_probe)

    command = sub.add_parser("run-published-caa")
    config_flags(command, models=True)
    command.add_argument("--phase", choices=["prepare", "run", "interfaces", "competence", "judge", "aggregate", "all"], default="all")
    command.add_argument("--behaviors", default=None)
    command.add_argument("--judges", default=None)
    command.set_defaults(func=cmd_published_caa)

    command = sub.add_parser("run-validity-controls")
    config_flags(command, models=True)
    command.add_argument("--phase", choices=["random", "prepare-judge", "summarize", "all"], default="all")
    command.set_defaults(func=cmd_validity)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
