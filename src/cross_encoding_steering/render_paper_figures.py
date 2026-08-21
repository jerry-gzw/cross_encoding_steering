from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = ["Qwen", "Llama", "Mistral", "Gemma"]
FIGURE_COLORS = {
    "ink": "#1F2430", "muted": "#6F768A", "grid": "#E6E8F0",
    "axis": "#D7DBE7", "panel": "#FFFFFF", "surface": "#FCFCFD",
    "blue": "#5477C4", "orange": "#CC6F47", "olive": "#71B436",
    "pink": "#BD569B", "neutral": "#C5CAD3", "neutral_dark": "#464C55",
}
MODEL_COLORS = {
    "Qwen": FIGURE_COLORS["blue"], "Llama": FIGURE_COLORS["orange"],
    "Mistral": FIGURE_COLORS["olive"], "Gemma": FIGURE_COLORS["pink"],
}
FACTORIAL_IDENTIFIER_LABELS = {
    "letters_abc": "A/B/C", "letters_xyz": "X/Y/Z", "numbers_123": "1/2/3",
}
MNLI_MAPPING_LABELS = {
    "letter_entailment_neutral_contradiction": "E/N/C (extraction)",
    "letter_entailment_contradiction_neutral": "E/C/N",
    "letter_neutral_entailment_contradiction": "N/E/C",
    "letter_neutral_contradiction_entailment": "N/C/E",
    "letter_contradiction_entailment_neutral": "C/E/N",
    "letter_contradiction_neutral_entailment": "C/N/E",
}
MNLI_EXTRACTION_INTERFACE = "letter_entailment_neutral_contradiction"
def _load_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            "Generating paper figures requires matplotlib. Install it with "
            "`pip install matplotlib` or run in the project environment."
        ) from exc
    return plt, Line2D

def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(FIGURE_COLORS["axis"])
    ax.spines["bottom"].set_color(FIGURE_COLORS["axis"])
    ax.tick_params(colors=FIGURE_COLORS["muted"])
    ax.xaxis.label.set_color(FIGURE_COLORS["ink"])
    ax.yaxis.label.set_color(FIGURE_COLORS["ink"])

def _save_matplotlib_figure(
    fig,
    *,
    figure_dir: Path,
    mirror_dir: Path | None,
    name: str,
) -> dict[str, object]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{name}.png"
    pdf_path = figure_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.025)
    legacy_svg_path = figure_dir / f"{name}.svg"
    legacy_svg_path.unlink(missing_ok=True)
    mirror_png_path = ""
    mirror_pdf_path = ""
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        mirror_png = mirror_dir / png_path.name
        mirror_pdf = mirror_dir / pdf_path.name
        shutil.copy2(png_path, mirror_png)
        shutil.copy2(pdf_path, mirror_pdf)
        (mirror_dir / legacy_svg_path.name).unlink(missing_ok=True)
        mirror_png_path = str(mirror_png)
        mirror_pdf_path = str(mirror_pdf)
    return {
        "figure_name": name,
        "pdf_path": str(pdf_path),
        "png_path": str(png_path),
        "renderer": "matplotlib",
        "mirror_pdf_path": mirror_pdf_path,
        "mirror_png_path": mirror_png_path,
    }

def _configure_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.8,
            "figure.facecolor": FIGURE_COLORS["surface"],
            "axes.facecolor": FIGURE_COLORS["panel"],
            "axes.edgecolor": FIGURE_COLORS["axis"],
            "axes.grid": False,
            "savefig.facecolor": FIGURE_COLORS["surface"],
        }
    )

def _figure_mapping_balanced_factorial_shift(
    frame: pd.DataFrame,
    figure_dir: Path,
    mirror_dir: Path | None,
) -> dict[str, object] | None:
    """Show how mapping averaging changes extraction-index advantage."""
    required = {
        "model_label",
        "raw_id_advantage",
        "mapping_balanced_id_advantage",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None
    plot = frame.dropna(subset=list(required)).copy()
    plot["model_order"] = plot["model_label"].map(
        {label: index for index, label in enumerate(MODEL_ORDER)}
    )
    plot = plot.sort_values("model_order")
    if plot.empty:
        return None

    plt, Line2D = _load_plotting()
    _configure_plot_style(plt)
    fig, ax = plt.subplots(figsize=(5.4, 2.65))
    y = np.arange(len(plot))
    for y_value, row in zip(y, plot.itertuples(index=False)):
        ax.plot(
            [row.raw_id_advantage, row.mapping_balanced_id_advantage],
            [y_value, y_value],
            color=FIGURE_COLORS["axis"],
            linewidth=1.5,
            zorder=1,
        )
    ax.scatter(
        plot["raw_id_advantage"],
        y,
        color=FIGURE_COLORS["blue"],
        marker="s",
        s=34,
        label="Canonical CAA",
        zorder=3,
    )
    ax.scatter(
        plot["mapping_balanced_id_advantage"],
        y,
        color=FIGURE_COLORS["orange"],
        marker="o",
        s=38,
        label="Mapping-balanced",
        zorder=3,
    )
    ax.axvline(
        0,
        color=FIGURE_COLORS["neutral_dark"],
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot["model_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Extraction-index advantage")
    ax.grid(
        axis="x",
        color=FIGURE_COLORS["grid"],
        linestyle="--",
        linewidth=0.7,
        zorder=0,
    )
    for tick, gridline in zip(ax.get_xticks(), ax.get_xgridlines()):
        if np.isclose(float(tick), 0.0):
            gridline.set_visible(False)
    _style_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.84, bottom=0.22)
    row = _save_matplotlib_figure(
        fig,
        figure_dir=figure_dir,
        mirror_dir=mirror_dir,
        name="fig_appendix_mapping_balanced_factorial_shift",
    )
    plt.close(fig)
    row.update(
        {
            "section": "Appendix",
            "chart_type": "paired model-level dumbbell plot",
            "source": "main_mapping_balanced_factorial_by_model.csv",
        }
    )
    return row

def _figure_interface_factorial_attribution(
    attribution: pd.DataFrame,
    vocabulary: pd.DataFrame,
    figure_dir: Path,
    mirror_dir: Path | None,
) -> dict[str, object] | None:
    attribution_required = {
        "model_label",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
    }
    vocabulary_required = {
        "model_label",
        "identifier_set",
        "mean",
        "ci_low",
        "ci_high",
    }
    if (
        attribution.empty
        or vocabulary.empty
        or not attribution_required.issubset(attribution.columns)
        or not vocabulary_required.issubset(vocabulary.columns)
    ):
        return None

    plt, Line2D = _load_plotting()
    _configure_plot_style(plt)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.05),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )

    difference_styles = [
        (
            "identifier_minus_semantics",
            "Extraction-index advantage",
            FIGURE_COLORS["blue"],
            "o",
        ),
        (
            "identifier_minus_row",
            "Extraction-index effect minus\nextraction-row effect",
            FIGURE_COLORS["olive"],
            "s",
        ),
    ]
    y = np.arange(len(MODEL_ORDER), dtype=float)[::-1]
    offsets = [-0.11, 0.11]
    for offset, (metric, label, color, marker) in zip(
        offsets,
        difference_styles,
    ):
        rows = (
            attribution.loc[attribution["metric"].eq(metric)]
            .set_index("model_label")
            .reindex(MODEL_ORDER)
        )
        means = rows["mean"].to_numpy(dtype=float)
        valid = np.isfinite(means)
        positions = y[valid] + offset
        axes[0].hlines(
            positions,
            0,
            means[valid],
            color=color,
            linewidth=1.35,
            alpha=0.82,
            zorder=2,
        )
        axes[0].scatter(
            means[valid],
            positions,
            color=color,
            marker=marker,
            s=30,
            edgecolors="white",
            linewidths=0.5,
            label=label,
            zorder=3,
        )
        for value, position in zip(means[valid], positions):
            axes[0].text(
                value + (0.045 if value >= 0 else -0.045),
                position,
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=7.1,
                color=color,
            )
    axes[0].axvline(
        0,
        color=FIGURE_COLORS["axis"],
        linewidth=0.9,
        linestyle="--",
        zorder=0,
    )
    axes[0].set_yticks(y, MODEL_ORDER)
    # Leave room above the offset Qwen row so it does not collide with the
    # panel heading after the figure is reduced to the paper column width.
    # Reserve a dedicated band above the Qwen row for the two-line legend.
    # Keeping the legend inside that empty band avoids covering any estimate.
    # axes[0].set_ylim(-0.35, float(y.max()) + 1.20)
    axes[0].set_xlabel("Paired mean effect difference")
    axes[0].set_xlim(-0.86, 2.12)
    axes[0].grid(
        axis="x",
        color=FIGURE_COLORS["grid"],
        linestyle="--",
        linewidth=0.7,
        zorder=0,
    )
    axes[0].legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1, 1.05),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    axes[0].text(
        0.0,
        1.035,
        "A  Paired attribution differences",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=FIGURE_COLORS["ink"],
    )
    _style_axes(axes[0])

    identifier_order = ["letters_abc", "letters_xyz", "numbers_123"]
    identifier_labels = [
        FACTORIAL_IDENTIFIER_LABELS[value] for value in identifier_order
    ]
    x_identifier = np.arange(len(identifier_order), dtype=float)
    for model_label in MODEL_ORDER:
        rows = (
            vocabulary.loc[vocabulary["model_label"].eq(model_label)]
            .set_index("identifier_set")
            .reindex(identifier_order)
        )
        means = rows["mean"].to_numpy(dtype=float)
        lows = rows["ci_low"].to_numpy(dtype=float)
        highs = rows["ci_high"].to_numpy(dtype=float)
        valid = np.isfinite(means) & np.isfinite(lows) & np.isfinite(highs)
        if not valid.any():
            continue
        color = MODEL_COLORS[model_label]
        axes[1].plot(
            x_identifier[valid],
            means[valid],
            marker="o",
            markersize=4.3,
            linewidth=1.45,
            color=color,
            label=model_label,
            zorder=3,
        )
        axes[1].vlines(
            x_identifier[valid],
            lows[valid],
            highs[valid],
            color=color,
            linewidth=0.9,
            alpha=0.75,
            zorder=2,
        )
    axes[1].axhline(
        0,
        color=FIGURE_COLORS["axis"],
        linewidth=0.9,
        linestyle="--",
        zorder=0,
    )
    axes[1].set_xticks(x_identifier, identifier_labels)
    axes[1].set_xlabel("Option-identifier vocabulary")
    axes[1].set_ylabel("Extraction-index effect")
    axes[1].grid(
        axis="y",
        color=FIGURE_COLORS["grid"],
        linestyle="--",
        linewidth=0.7,
        zorder=0,
    )
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[label],
                marker="o",
                linewidth=1.4,
                markersize=4,
                label=label,
            )
            for label in MODEL_ORDER
        ],
        frameon=False,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    axes[1].text(
        0.0,
        1.035,
        "B  Identifier-vocabulary sensitivity",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=FIGURE_COLORS["ink"],
    )
    _style_axes(axes[1])

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.24,
        wspace=0.27,
    )
    row = _save_matplotlib_figure(
        fig,
        figure_dir=figure_dir,
        mirror_dir=mirror_dir,
        name="fig_results_identifier_position_semantics_factorial",
    )
    plt.close(fig)
    row.update(
        {
            "section": "Results",
            "chart_type": "paired-difference lollipop plot and vocabulary-sensitivity lines",
            "source": (
                "main_interface_factorial_attribution_by_model.csv;"
                "main_interface_factorial_identifier_sensitivity.csv"
            ),
        }
    )
    return row

def _figure_layer_attribution_trajectory(
    aggregate: pd.DataFrame,
    by_model: pd.DataFrame,
    figure_dir: Path,
    mirror_dir: Path | None,
) -> dict[str, object] | None:
    required_aggregate = {
        "requested_layer_fraction",
        "metric",
        "mean",
    }
    required_model = {
        "model_label",
        "requested_layer_fraction",
        "mean_extraction_identifier_gain",
        "extraction_identifier_advantage",
    }
    if (
        aggregate.empty
        or by_model.empty
        or not required_aggregate.issubset(aggregate.columns)
        or not required_model.issubset(by_model.columns)
    ):
        return None

    plt, Line2D = _load_plotting()
    _configure_plot_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(8.45, 2.85))
    panels = [
        (
            "extraction_identifier_effect",
            "mean_extraction_identifier_gain",
            "A  Extraction-index effect",
            "Extraction-index effect",
        ),
        (
            "extraction_identifier_advantage",
            "extraction_identifier_advantage",
            "B  Extraction-index advantage",
            "Extraction-index advantage",
        ),
    ]
    depth_ticks = np.asarray([0.5, 0.625, 0.75, 0.875])
    for axis, (metric, model_value, label, y_label) in zip(axes, panels):
        for model_label in MODEL_ORDER:
            model = (
                by_model.loc[by_model["model_label"].eq(model_label)]
                .sort_values("requested_layer_fraction")
            )
            if model.empty:
                continue
            axis.plot(
                model["requested_layer_fraction"],
                model[model_value],
                color=MODEL_COLORS[model_label],
                linewidth=1.0,
                marker="o",
                markersize=3.2,
                alpha=0.62,
                zorder=2,
            )
        summary = (
            aggregate.loc[aggregate["metric"].eq(metric)]
            .sort_values("requested_layer_fraction")
        )
        x = summary["requested_layer_fraction"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        axis.plot(
            x,
            mean,
            color=FIGURE_COLORS["ink"],
            marker="D",
            markersize=3.6,
            linewidth=1.1,
            label="Pooled mean",
            zorder=4,
        )
        axis.axhline(
            0,
            color=FIGURE_COLORS["axis"],
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
        axis.set_xticks(
            depth_ticks,
            ["50%", "62.5%", "75%", "87.5%"],
        )
        axis.set_xlabel("Relative model depth")
        axis.set_ylabel(y_label)
        axis.grid(
            axis="y",
            color=FIGURE_COLORS["grid"],
            linestyle="--",
            linewidth=0.7,
            zorder=0,
        )
        axis.text(
            0.0,
            1.035,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=FIGURE_COLORS["ink"],
        )
        _style_axes(axis)

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[label],
            marker="o",
            linewidth=1.1,
            markersize=3.4,
            label=label,
            alpha=0.72,
        )
        for label in MODEL_ORDER
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=FIGURE_COLORS["ink"],
            marker="D",
            linewidth=1.8,
            markersize=4,
            label="Pooled mean",
        )
    )
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.79,
        bottom=0.23,
        wspace=0.27,
    )
    row = _save_matplotlib_figure(
        fig,
        figure_dir=figure_dir,
        mirror_dir=mirror_dir,
        name="fig_results_layer_attribution_trajectory",
    )
    plt.close(fig)
    row.update(
        {
            "section": "Results",
            "chart_type": "model layer trajectories with pooled means",
            "source": (
                "main_layer_attribution_trajectory.csv;"
                "main_layer_attribution_by_model.csv"
            ),
        }
    )
    return row

def _figure_mnli_cross_encoding_attribution(
    by_model_mapping: pd.DataFrame,
    baseline: pd.DataFrame,
    figure_dir: Path,
    mirror_dir: Path | None,
) -> dict[str, object] | None:
    required_effect = {
        "model_label",
        "interface",
        "mapping",
        "metric",
        "estimate",
    }
    required_baseline = {
        "model_label",
        "interface",
        "mapping",
        "estimate",
    }
    if (
        by_model_mapping.empty
        or baseline.empty
        or not required_effect.issubset(by_model_mapping.columns)
        or not required_baseline.issubset(baseline.columns)
    ):
        return None
    interfaces = [
        name
        for name in MNLI_MAPPING_LABELS
        if name != MNLI_EXTRACTION_INTERFACE
    ]
    labels = [MNLI_MAPPING_LABELS[name] for name in interfaces]
    effects = by_model_mapping.loc[
        by_model_mapping["metric"].eq("id_advantage")
        & by_model_mapping["interface"].isin(interfaces)
    ].copy()
    competence = baseline.loc[baseline["interface"].isin(interfaces)].copy()
    if effects.empty or competence.empty:
        return None

    plt, Line2D = _load_plotting()
    _configure_plot_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(8.45, 2.85))
    x = np.arange(len(interfaces), dtype=float)
    for model_label in MODEL_ORDER:
        effect_rows = (
            effects.loc[effects["model_label"].eq(model_label)]
            .set_index("interface")
            .reindex(interfaces)
        )
        baseline_rows = (
            competence.loc[competence["model_label"].eq(model_label)]
            .set_index("interface")
            .reindex(interfaces)
        )
        color = MODEL_COLORS[model_label]
        axes[0].plot(
            x,
            effect_rows["estimate"].to_numpy(dtype=float),
            color=color,
            marker="o",
            markersize=3.6,
            linewidth=1.25,
            label=model_label,
            zorder=3,
        )
        axes[1].plot(
            x,
            baseline_rows["estimate"].to_numpy(dtype=float),
            color=color,
            marker="o",
            markersize=3.6,
            linewidth=1.25,
            label=model_label,
            zorder=3,
        )

    axes[0].axhline(
        0,
        color=FIGURE_COLORS["axis"],
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )
    axes[0].set_ylabel("Extraction-index advantage")
    axes[0].text(
        0.0,
        1.035,
        "A  Frozen-direction attribution",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=FIGURE_COLORS["ink"],
    )
    axes[1].axhline(
        1.0 / 3.0,
        color=FIGURE_COLORS["axis"],
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )
    axes[1].set_ylabel("Unsteered accuracy")
    axes[1].set_ylim(0.30, 0.88)
    axes[1].text(
        0.0,
        1.035,
        "B  Mapping competence",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=FIGURE_COLORS["ink"],
    )
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("MNLI A/B/C mapping")
        axis.grid(
            axis="y",
            color=FIGURE_COLORS["grid"],
            linestyle="--",
            linewidth=0.7,
            zorder=0,
        )
        _style_axes(axis)
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[label],
            marker="o",
            linewidth=1.25,
            markersize=3.6,
            label=label,
        )
        for label in MODEL_ORDER
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.79,
        bottom=0.23,
        wspace=0.27,
    )
    row = _save_matplotlib_figure(
        fig,
        figure_dir=figure_dir,
        mirror_dir=mirror_dir,
        name="fig_appendix_mnli_cross_encoding_attribution",
    )
    plt.close(fig)
    row.update(
        {
            "section": "Appendix",
            "chart_type": "model mapping profiles and competence",
            "source": (
                "appendix_mnli_cross_encoding_by_model_mapping.csv;"
                "appendix_mnli_baseline_competence.csv"
            ),
        }
    )
    return row

def _read(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing paper-result table: {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the paper's four quantitative figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("paper_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results/figures"))
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = [
        _figure_interface_factorial_attribution(
            _read(results_dir, "main_interface_factorial_attribution_by_model.csv"),
            _read(results_dir, "main_interface_factorial_identifier_sensitivity.csv"),
            output_dir,
            None,
        ),
        _figure_layer_attribution_trajectory(
            _read(results_dir, "main_layer_attribution_trajectory.csv"),
            _read(results_dir, "main_layer_attribution_by_model.csv"),
            output_dir,
            None,
        ),
        _figure_mnli_cross_encoding_attribution(
            _read(results_dir, "appendix_mnli_cross_encoding_by_model_mapping.csv"),
            _read(results_dir, "appendix_mnli_baseline_competence.csv"),
            output_dir,
            None,
        ),
        _figure_mapping_balanced_factorial_shift(
            _read(results_dir, "main_mapping_balanced_factorial_by_model.csv"),
            output_dir,
            None,
        ),
    ]
    missing = [index for index, row in enumerate(rows, start=1) if row is None]
    if missing:
        raise RuntimeError(f"Could not render figure builders: {missing}")
    print(f"Rendered {len(rows)} figures in {output_dir}")


if __name__ == "__main__":
    main()
