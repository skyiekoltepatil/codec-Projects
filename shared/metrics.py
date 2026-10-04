"""Model-agnostic evaluation metrics with a single JSON-serialisable shape.

Every project writes its metrics through these helpers so that the dashboard,
the per-project READMEs and ``FINAL_PROJECT_REPORT.md`` all quote the same
numbers produced by the same code path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Return MAE, MSE, RMSE and R2 for a regression problem.

    Implemented directly on NumPy so the numbers are stable across
    scikit-learn versions and reproducible in tests.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true_arr.shape} and {y_pred_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute regression metrics on empty arrays")

    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    label_names: Sequence[str] | None = None,
    y_proba: Sequence[Sequence[float]] | np.ndarray | None = None,
    average: str = "macro",
    pos_label: Any | None = None,
) -> dict[str, Any]:
    """Return accuracy/precision/recall/F1, the confusion matrix and optional ROC-AUC.

    Args:
        y_true: ground-truth labels.
        y_pred: predicted labels.
        labels: explicit label order; inferred and sorted when omitted.
        label_names: Display names aligned with ``labels``. Needed when the model
            works on integer class indices but the interface must show names:
            scikit-learn rejects ``labels=["Apple", "Banana"]`` when ``y_true``
            contains ``0`` and ``1``.
        y_proba: ``(n_samples, n_classes)`` class probabilities. When supplied for a
            binary problem, ROC-AUC is computed from the positive-class column.
        average: averaging strategy for precision/recall/F1.
        pos_label: Required when ``average="binary"`` and the labels are strings.
            scikit-learn otherwise defaults to the integer ``1``, which raises
            ``ValueError: pos_label=1 is not a valid label`` for text labels.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute classification metrics on empty arrays")

    resolved_labels = list(labels) if labels is not None else sorted(set(y_true_arr) | set(y_pred_arr))

    binary_kwargs: dict[str, Any] = {}
    if average == "binary":
        if pos_label is None and resolved_labels and not isinstance(resolved_labels[0], (int, np.integer)):
            raise ValueError(
                "average='binary' requires pos_label when the labels are not integers; "
                f"resolved labels were {resolved_labels!r}"
            )
        if pos_label is not None:
            binary_kwargs["pos_label"] = pos_label

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(
                y_true_arr, y_pred_arr, labels=resolved_labels, average=average,
                zero_division=0, **binary_kwargs,
            )
        ),
        "recall": float(
            recall_score(
                y_true_arr, y_pred_arr, labels=resolved_labels, average=average,
                zero_division=0, **binary_kwargs,
            )
        ),
        "f1": float(
            f1_score(
                y_true_arr, y_pred_arr, labels=resolved_labels, average=average,
                zero_division=0, **binary_kwargs,
            )
        ),
        "average": average,
        "labels": [str(label) for label in resolved_labels],
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=resolved_labels).tolist(),
    }
    if label_names is not None:
        if len(label_names) != len(resolved_labels):
            raise ValueError(
                f"label_names has {len(label_names)} entries but {len(resolved_labels)} labels "
                "were supplied; they must correspond one-to-one."
            )
        metrics["labels"] = [str(name) for name in label_names]
    if pos_label is not None:
        metrics["pos_label"] = str(pos_label)

    if y_proba is not None:
        proba_arr = np.asarray(y_proba, dtype=float)
        try:
            if len(resolved_labels) == 2:
                positive_scores = proba_arr[:, -1] if proba_arr.ndim == 2 else proba_arr
                binary_truth = (y_true_arr == resolved_labels[-1]).astype(int)
                if len(set(binary_truth.tolist())) > 1:
                    metrics["roc_auc"] = float(roc_auc_score(binary_truth, positive_scores))
            elif proba_arr.ndim == 2 and proba_arr.shape[1] == len(resolved_labels):
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(
                        y_true_arr,
                        proba_arr,
                        multi_class="ovr",
                        average=average,
                        labels=resolved_labels,
                    )
                )
        except ValueError:
            # ROC-AUC is undefined when a split contains a single class.
            pass

    return metrics


def per_class_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    label_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return per-class precision/recall/F1/support rows for tables in the UI.

    Args:
        label_names: Optional display names aligned with ``labels``; see
            :func:`classification_metrics` for why integer class indices and
            string names cannot be passed interchangeably to scikit-learn.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    resolved = list(labels) if labels is not None else sorted(set(y_true_arr) | set(y_pred_arr))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=resolved, zero_division=0
    )
    names = [str(label) for label in resolved]
    if label_names is not None:
        if len(label_names) != len(resolved):
            raise ValueError(
                f"label_names has {len(label_names)} entries but {len(resolved)} labels "
                "were supplied; they must correspond one-to-one."
            )
        names = [str(name) for name in label_names]

    return [
        {
            "label": names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(resolved))
    ]


# --------------------------------------------------------------------------- #
# Ranking / recommendation
# --------------------------------------------------------------------------- #


def precision_at_k(
    recommended: Sequence[Sequence[Any]],
    relevant: Sequence[set[Any]],
    k: int,
) -> float:
    """Mean Precision@K across users.

    ``recommended[i]`` is the ranked list for user *i*; ``relevant[i]`` is that
    user's held-out relevant items.
    """
    if len(recommended) != len(relevant):
        raise ValueError("recommended and relevant must have equal length")
    if not recommended:
        return 0.0
    scores = []
    for recs, rel in zip(recommended, relevant):
        top_k = list(recs)[:k]
        if not top_k:
            scores.append(0.0)
            continue
        hits = sum(1 for item in top_k if item in rel)
        scores.append(hits / len(top_k))
    return float(np.mean(scores))


def recall_at_k(
    recommended: Sequence[Sequence[Any]],
    relevant: Sequence[set[Any]],
    k: int,
) -> float:
    """Mean Recall@K across users, ignoring users with no relevant items."""
    if len(recommended) != len(relevant):
        raise ValueError("recommended and relevant must have equal length")
    scores = []
    for recs, rel in zip(recommended, relevant):
        if not rel:
            continue
        top_k = list(recs)[:k]
        hits = sum(1 for item in top_k if item in rel)
        scores.append(hits / len(rel))
    return float(np.mean(scores)) if scores else 0.0


def coverage(recommended: Sequence[Sequence[Any]], catalogue_size: int) -> float:
    """Fraction of the catalogue that appears in at least one recommendation list."""
    if catalogue_size <= 0:
        return 0.0
    seen = {item for recs in recommended for item in recs}
    return len(seen) / catalogue_size


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Write metrics to ``path`` as indented JSON, creating parent directories."""
    from shared.paths import ensure_parent

    target = ensure_parent(Path(path))
    target.write_text(json.dumps(metrics, indent=2, sort_keys=False), encoding="utf-8")
    return target


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load a metrics JSON file previously written by :func:`save_metrics`."""
    from shared.errors import ModelNotFoundError

    target = Path(path)
    if not target.exists():
        raise ModelNotFoundError(
            f"Metrics file not found at '{target.name}'.",
            hint="Run the project's train.py to generate metrics and model artefacts.",
        )
    return json.loads(target.read_text(encoding="utf-8"))


def format_metric(value: float, digits: int = 4) -> str:
    """Format a metric for display, avoiding scientific notation for tiny values."""
    if value is None:
        return "n/a"
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"