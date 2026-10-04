"""Shared plotting helpers with one consistent visual style.

Every project renders charts through this module, which guarantees the same
palette, typography and sizing across all ten applications.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from shared.theme import CHART_COLORS, PALETTE

# --------------------------------------------------------------------------- #
# Matplotlib / Seaborn
# --------------------------------------------------------------------------- #

_STYLE_APPLIED = False


def apply_matplotlib_style() -> None:
    """Apply the shared matplotlib/seaborn theme once per process."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    import matplotlib

    matplotlib.use("Agg")  # Headless: required for Streamlit and CI environments.
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 4.2),
            "figure.dpi": 120,
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": PALETTE["line"],
            "grid.linewidth": 0.8,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["ink"],
            "axes.prop_cycle": matplotlib.cycler(color=CHART_COLORS),
            "figure.autolayout": True,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
    _STYLE_APPLIED = True


def new_figure(width: float = 9.0, height: float = 4.2):
    """Return ``(figure, axes)`` with the shared style applied."""
    apply_matplotlib_style()
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(width, height))


def line_chart(
    series: dict[str, Sequence[float]],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    x: Sequence[Any] | None = None,
    width: float = 9.0,
    height: float = 4.2,
):
    """Plot one or more labelled lines on a single axis."""
    fig, ax = new_figure(width, height)
    for index, (label, values) in enumerate(series.items()):
        ax.plot(
            x if x is not None else range(len(values)),
            values,
            label=label,
            color=CHART_COLORS[index % len(CHART_COLORS)],
            linewidth=1.6,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend()
    return fig


def scatter_chart(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    width: float = 5.6,
    height: float = 4.6,
):
    """Scatter plot with a dashed identity line, used for predicted vs actual."""
    fig, ax = new_figure(width, height)
    ax.scatter(x_values, y_values, s=14, alpha=0.55, color=CHART_COLORS[0], edgecolors="none")
    if len(x_values) > 0:
        low = float(np.min([np.min(x_values), np.min(y_values)]))
        high = float(np.max([np.max(x_values), np.max(y_values)]))
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2, color=PALETTE["muted"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    horizontal: bool = False,
    width: float = 8.0,
    height: float = 4.2,
):
    """Bar chart with automatic rotation for long category labels."""
    fig, ax = new_figure(width, height)
    positions = np.arange(len(labels))
    if horizontal:
        ax.barh(positions, values, color=CHART_COLORS[0], height=0.6)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
    else:
        ax.bar(positions, values, color=CHART_COLORS[0], width=0.6)
        ax.set_xticks(positions, labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def histogram(
    values: Sequence[float],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "Frequency",
    bins: int = 30,
    kde: bool = True,
    width: float = 8.0,
    height: float = 4.0,
):
    """Histogram with an optional kernel-density overlay."""
    apply_matplotlib_style()
    import seaborn as sns

    fig, ax = new_figure(width, height)
    sns.histplot(values, bins=bins, kde=kde, ax=ax, color=CHART_COLORS[0], edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def confusion_matrix_heatmap(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    *,
    title: str = "Confusion matrix",
    width: float = 5.6,
    height: float = 4.6,
    normalize: bool = False,
):
    """Annotated confusion-matrix heatmap; rows are true labels, columns predicted."""
    apply_matplotlib_style()
    import seaborn as sns

    raw = np.asarray(matrix)
    if normalize:
        # Row-normalise, guarding against rows whose true-label count is zero.
        as_float = raw.astype(float)
        row_sums = as_float.sum(axis=1, keepdims=True)
        array = np.divide(as_float, row_sums, out=np.zeros_like(as_float), where=row_sums != 0)
        fmt = ".2f"
    else:
        # Keep integer dtype so seaborn's "d" format specifier is valid.
        array = raw
        fmt = "d"

    fig, ax = new_figure(width, height)
    sns.heatmap(
        array,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.6,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def correlation_heatmap(frame, *, title: str = "Correlation matrix", width: float = 8.4, height: float = 6.6):
    """Correlation heatmap for the numeric columns of a DataFrame."""
    apply_matplotlib_style()
    import seaborn as sns

    numeric = frame.select_dtypes(include=[np.number])
    fig, ax = new_figure(width, height)
    if numeric.shape[1] < 2:
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
        ax.axis("off")
        return fig
    sns.heatmap(
        numeric.corr(numeric_only=True),
        annot=numeric.shape[1] <= 12,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(title)
    return fig


def training_curves(
    history: dict[str, Sequence[float]],
    *,
    title: str = "Training history",
    xlabel: str = "Epoch",
    width: float = 8.6,
    height: float = 4.0,
):
    """Two-panel loss/accuracy curves from a ``{"train_loss": [...], ...}`` dict."""
    apply_matplotlib_style()
    import matplotlib.pyplot as plt

    loss_keys = [key for key in history if "loss" in key]
    metric_keys = [key for key in history if "loss" not in key]

    panels = [keys for keys in (loss_keys, metric_keys) if keys]
    if not panels:
        raise ValueError("history must contain at least one series")

    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False)
    for panel_index, keys in enumerate(panels):
        ax = axes[0][panel_index]
        for key_index, key in enumerate(keys):
            ax.plot(
                range(1, len(history[key]) + 1),
                history[key],
                label=key.replace("_", " "),
                color=CHART_COLORS[key_index % len(CHART_COLORS)],
                linewidth=1.6,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss" if "loss" in keys[0] else "Score")
        ax.set_title("Loss" if "loss" in keys[0] else "Accuracy")
        ax.legend()
    fig.suptitle(title, fontsize=12, fontweight="600")
    return fig


def residual_plot(
    predictions: Sequence[float],
    residuals: Sequence[float],
    *,
    title: str = "Residuals vs predicted",
    width: float = 8.0,
    height: float = 4.0,
):
    """Residual scatter with a zero reference line."""
    fig, ax = new_figure(width, height)
    ax.scatter(predictions, residuals, s=14, alpha=0.55, color=CHART_COLORS[1], edgecolors="none")
    ax.axhline(0.0, color=PALETTE["negative"], linewidth=1.2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (actual - predicted)")
    return fig


def feature_importance_bar(
    names: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "Feature importance",
    top_n: int = 12,
    width: float = 8.0,
    height: float = 4.6,
):
    """Horizontal bar chart of the ``top_n`` most important features."""
    order = np.argsort(np.abs(np.asarray(values)))[::-1][:top_n]
    ordered_names = [str(names[i]) for i in order][::-1]
    ordered_values = [float(np.asarray(values)[i]) for i in order][::-1]

    fig, ax = new_figure(width, height)
    ax.barh(range(len(ordered_names)), ordered_values, color=CHART_COLORS[0], height=0.65)
    ax.set_yticks(range(len(ordered_names)), ordered_names)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return fig


# --------------------------------------------------------------------------- #
# Plotly (interactive)
# --------------------------------------------------------------------------- #


def interactive_line(
    series: dict[str, Sequence[float]],
    *,
    title: str,
    x: Sequence[Any] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    height: int = 380,
):
    """Return a Plotly line figure sharing the repository palette."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for index, (label, values) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=list(x) if x is not None else list(range(len(values))),
                y=list(values),
                mode="lines",
                name=label,
                line={"color": CHART_COLORS[index % len(CHART_COLORS)], "width": 2},
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    return fig


def interactive_scatter(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    text: Sequence[str] | None = None,
    height: int = 420,
):
    """Return a Plotly scatter figure with an identity reference line."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(x_values),
            y=list(y_values),
            mode="markers",
            marker={"size": 7, "color": CHART_COLORS[0], "opacity": 0.65},
            text=list(text) if text is not None else None,
            name="observations",
        )
    )
    if len(x_values) > 0:
        low = float(np.min([np.min(x_values), np.min(y_values)]))
        high = float(np.max([np.max(x_values), np.max(y_values)]))
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                line={"dash": "dash", "color": PALETTE["muted"], "width": 1.4},
                name="perfect prediction",
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        template="plotly_white",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    return fig


def interactive_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    height: int = 360,
):
    """Return a Plotly bar figure for class probabilities or importances."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=list(labels),
            y=list(values),
            marker={"color": [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(labels))]},
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        template="plotly_white",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=False,
    )
    return fig