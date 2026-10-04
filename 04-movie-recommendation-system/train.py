#!/usr/bin/env python
"""Build, evaluate and save the MovieLens recommender models.

Run from the ``04-movie-recommendation-system`` directory::

    python train.py

Downloads MovieLens ml-latest-small (~1 MB) if necessary, fits the content-based
TF-IDF model and the collaborative SVD model, scores both with a leave-one-out
offline evaluation, prints the comparison table and saves:

* ``models/recommender.joblib``          - both fitted models
* ``models/training_metrics.json``       - evaluation results and settings

Useful flags::

    python train.py --eval-users 400     # more users in the offline evaluation
    python train.py --skip-collaborative  # content model only (faster)
    python train.py --force-download
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

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.ml import Timer  # noqa: E402
from shared.paths import portable_display  # noqa: E402

from src.data import load_dataset  # noqa: E402
from src.model import (  # noqa: E402
    DEFAULT_TOP_K,
    EVALUATION_KS,
    POSITIVE_RATING_THRESHOLD,
    MovieRecommender,
    build_collaborative_model,
    build_content_model,
    build_training_report,
    evaluate_offline,
    load_model,
    save_metrics_report,
    save_model,
)

logger = logging.getLogger("train")

METHOD_LABELS = {
    "content": "Content-based (TF-IDF + cosine)",
    "collaborative": "Collaborative (SVD)",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Build and evaluate the MovieLens recommender models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-users",
        type=int,
        default=200,
        help="Users sampled for the leave-one-out evaluation.",
    )
    parser.add_argument(
        "--skip-collaborative",
        action="store_true",
        help="Build only the content-based model (much faster, no SVD fit).",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=POSITIVE_RATING_THRESHOLD,
        help="A rating at or above this counts as a positive interaction.",
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download MovieLens even if cached."
    )
    return parser.parse_args()


def _print_evaluation(evaluation: dict) -> None:
    """Print the Precision@K / Recall@K comparison table."""
    ks = evaluation["ks"]
    header = f"{'Method':<34}" + "".join(f"{'P@' + str(k):>9}" for k in ks) + f"{'Coverage':>12}{'Mean rank':>12}"
    print()
    print(header)
    print("-" * len(header))
    for method, payload in evaluation["methods"].items():
        precision = payload["precision_at_k"]
        rank = payload.get("mean_rank_when_hit")
        rank_text = format_metric(rank, 1) if rank is not None else "n/a"
        print(
            f"{METHOD_LABELS.get(method, method):<34}"
            + "".join(f"{format_metric(precision[str(k)]):>9}" for k in ks)
            + f"{format_metric(payload['catalogue_coverage'], 4):>12}"
            + f"{rank_text:>12}"
        )
    print()
    print(f"Users evaluated: {evaluation['n_users_evaluated']}")
    print("A positive interaction is a rating >= " f"{evaluation['positive_threshold']}.")
    print("Recall@K equals Precision@K here because each user contributes exactly one held-out film.")


def _print_examples(recommender: MovieRecommender) -> None:
    """Print a few qualitative recommendations for a well-known seed film."""
    examples = [("Toy Story", "Toy Story (1995)"), ("The Matrix", "The Matrix (1999)")]
    for needle, description in examples:
        matches = recommender.search(needle, limit=5)
        if not matches:
            continue
        seed_id = matches[0]["movie_id"]
        print(f"\nFilms like {description}:")
        for item in recommender.recommend(seed_id, top_k=5):
            genres = ", ".join(item["genres"][:3]) or "no genres listed"
            print(f"  {item['title']:<45}{genres:<32}similarity {item['similarity']:.3f}")


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    configure_logging()

    logger.info("=" * 72)
    logger.info("Movie Recommendation System - building models")
    logger.info("=" * 72)

    with Timer() as data_timer:
        data = load_dataset(force_download=args.force_download)
    summary = data.summary()
    logger.info(
        "Loaded MovieLens in %s: %d movies, %d users, %d ratings",
        data_timer,
        summary["n_movies"],
        summary["n_users"],
        summary["n_ratings"],
    )

    with Timer() as content_timer:
        content = build_content_model(data)
    logger.info("Content model fitted in %s (%d TF-IDF features)", content_timer, content.n_features)

    collaborative = None
    if not args.skip_collaborative:
        with Timer() as collaborative_timer:
            collaborative = build_collaborative_model(data)
        logger.info("Collaborative model fitted in %s", collaborative_timer)

    recommender = MovieRecommender(content=content, collaborative=collaborative)

    with Timer() as eval_timer:
        evaluation = evaluate_offline(
            data,
            recommender,
            ks=EVALUATION_KS,
            n_users=args.eval_users,
            positive_threshold=args.positive_threshold,
        )
    logger.info("Offline evaluation finished in %s", eval_timer)

    import pandas as pd

    recommender.evaluation = evaluation
    recommender.metrics = {
        "n_movies": content.n_movies,
        "n_tfidf_features": content.n_features,
        "n_svd_factors": collaborative.n_factors if collaborative else 0,
        "svd_explained_variance": getattr(collaborative, "explained_variance", None),
        "default_top_k": DEFAULT_TOP_K,
    }
    recommender.dataset = summary
    recommender.trained_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")

    _print_evaluation(evaluation)
    _print_examples(recommender)

    model_path = save_model(recommender)
    logger.info("Saved model artefact -> %s", model_path)

    report = build_training_report(recommender, dataset=summary)
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    reloaded = load_model(model_path)
    if reloaded.content.n_movies != content.n_movies:
        logger.error("Reloaded model has %d movies, expected %d", reloaded.content.n_movies, content.n_movies)
        return 1

    print()
    print("Model saved successfully.")
    print(f"  Artefact : {portable_display(model_path)}")
    print(f"  Metrics  : {portable_display(metrics_path)}")
    print(f"  Movies   : {content.n_movies:,}")
    print(f"  TF-IDF features: {content.n_features:,}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())