#!/usr/bin/env python
"""Train, evaluate and save the customer-service chatbot's intent classifier.

Run from the ``06-customer-service-chatbot`` directory::

    python train.py

Trains TF-IDF + Logistic Regression on the intent corpus in ``src/data.py``,
reports per-intent accuracy, checks the confidence gate against held-out
paraphrases, and saves the model plus its metrics.

Flags let you see how the two thresholds change the bot's behaviour::

    python train.py --min-confidence 0.40
    python train.py --min-margin 0.05
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import pandas as pd  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.ml import Timer  # noqa: E402
from shared.paths import portable_display  # noqa: E402

from src.data import (  # noqa: E402
    CLASS_LABELS,
    FALLBACK_INTENT,
    GATE_PROBES,
    INTENT_LABELS,
    build_corpus,
    corpus_summary,
)
from src.model import (  # noqa: E402
    MIN_CONFIDENCE,
    MIN_MARGIN,
    build_training_report,
    load_model,
    save_metrics_report,
    save_model,
    train_model,
)

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the customer-service intent classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-confidence", type=float, default=MIN_CONFIDENCE,
        help="Decline when the top intent scores below this.",
    )
    parser.add_argument(
        "--min-margin", type=float, default=MIN_MARGIN,
        help="Decline when the top two intents are closer than this.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out fraction.")
    return parser.parse_args()


def _print_per_class(model) -> None:
    """Print the per-intent F1 table."""
    if not model.per_class:
        return
    frame = pd.DataFrame(model.per_class).rename(columns={"label": "intent"})
    frame["intent"] = frame["intent"].map(lambda name: INTENT_LABELS.get(name, name))
    print()
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def _print_gate_examples(model) -> dict:
    """Run the held-out probes and print how the gate behaved on each."""
    print()
    print("Confidence gate on held-out paraphrases")
    print("-" * 72)
    for text, true_intent in GATE_PROBES:
        response = model.classify(text)
        expected = INTENT_LABELS.get(true_intent, true_intent)
        if response.understood:
            verdict = "OK  " if response.intent == true_intent else "WRONG"
            print(f"[{verdict}] {response.intent_label:<20} ({response.confidence:.0%})  <- '{text}'")
        else:
            # Declining an off-topic message is correct; declining a real one is not.
            verdict = "OK  " if true_intent == FALLBACK_INTENT else "OVER-CAUTIOUS"
            print(f"[{verdict}] declined ({response.confidence:.0%})  <- '{text}' (expected {expected})")
    print()

    return model.confidence_gate_report(list(GATE_PROBES))


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    configure_logging()

    logger.info("=" * 72)
    logger.info("Customer Service Chatbot - training")
    logger.info("=" * 72)

    texts, labels = build_corpus()
    summary = corpus_summary()
    logger.info(
        "Built %d training examples from %d base utterances across %d intents",
        len(texts),
        summary["n_base_utterances"],
        summary["n_intents"],
    )
    logger.info(
        "Confidence gate: min_confidence=%.2f, min_margin=%.2f",
        args.min_confidence,
        args.min_margin,
    )

    with Timer() as timer:
        model = train_model(
            texts,
            labels,
            test_size=args.test_size,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
        )
    logger.info("Trained in %s", timer)

    import pandas as _pd

    model.trained_at = _pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")

    metrics = model.metrics
    print()
    print(f"{'Accuracy':<28}{format_metric(metrics['accuracy']):>12}")
    print(f"{'Macro F1':<28}{format_metric(metrics['f1']):>12}")
    print(f"{'Weighted F1':<28}{format_metric(metrics['f1']):>12}")
    print(f"{'ROC-AUC (one-vs-rest)':<28}{format_metric(metrics.get('roc_auc_ovr')):>12}")
    print(f"{'Training examples':<28}{metrics['n_train']:>12,}")
    print(f"{'Held-out examples':<28}{metrics['n_test']:>12,}")

    _print_per_class(model)
    gate_report = _print_gate_examples(model)

    print("Gate summary")
    print("-" * 72)
    print(f"{'Answered':<34}{gate_report['answered']}/{gate_report['n_messages']}")
    print(f"{'Declined':<34}{gate_report['declined']}/{gate_report['n_messages']}")
    print(f"{'Precision when answering':<34}{format_metric(gate_report['precision_when_answering'])}")
    print(f"{'Wrong answers given':<34}{gate_report['wrong_answers']}")
    print()

    model_path = save_model(model)
    logger.info("Saved model artefact -> %s", model_path)

    report = build_training_report(model, corpus=summary, gate_report=gate_report)
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    reloaded = load_model(model_path)
    probe = "what are your opening hours?"
    if reloaded.classify(probe).intent != model.classify(probe).intent:
        logger.error("Reloaded model disagrees with the trained model on '%s'.", probe)
        return 1

    print("Model saved successfully.")
    print(f"  Artefact : {portable_display(model_path)}")
    print(f"  Metrics  : {portable_display(metrics_path)}")
    print(f"  Accuracy : {format_metric(metrics['accuracy'])}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())