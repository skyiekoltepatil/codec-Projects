"""Streamlit interface for the Movie Recommendation System.

Run from this project's directory::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.paths import models_dir, portable_display  # noqa: E402
from shared.plotting import bar_chart  # noqa: E402
from shared.theme import PALETTE, inject_theme, metric_row, note, page_header, result_card  # noqa: E402
from shared.ui import (  # noqa: E402
    model_info_panel,
    section,
    show_dataframe,
    show_error,
    show_figure,
)

from src.model import (  # noqa: E402
    DEFAULT_TOP_K,
    EVALUATION_KS,
    POSITIVE_RATING_THRESHOLD,
    MovieRecommender,
    load_model,
)

MODEL_PATH = models_dir("04-movie-recommendation-system") / "recommender.joblib"

METHOD_LABELS = {
    "content": "Content-based",
    "collaborative": "Collaborative (SVD)",
}

DISCLAIMER = (
    "MovieLens contains only 610 users and 9,742 films, of which only 1,572 carry any "
    "crowdsourced tag. Content-based similarity therefore has very little text to work "
    "with and scores near zero at K=10 in the offline evaluation; the collaborative model "
    "is far stronger here. Treat these as educational baselines, not production quality."
)


@st.cache_resource(show_spinner=False)
def _load_recommender(path_str: str, mtime: float) -> MovieRecommender:
    """Load the fitted recommenders once per artefact version."""
    return load_model(Path(path_str))


def _get_recommender() -> MovieRecommender | None:
    """Return the trained recommender, or ``None`` when it has not been built."""
    if not MODEL_PATH.exists():
        return None
    return _load_recommender(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def _suggested_titles(recommender: MovieRecommender) -> dict[str, int]:
    """Return a handful of well-known films as ready-made seeds."""
    candidates = [
        "Toy Story",
        "The Matrix",
        "Jurassic Park",
        "Titanic",
        "Star Wars",
        "The Godfather",
        "Finding Nemo",
        "Pulp Fiction",
    ]
    suggestions: dict[str, int] = {}
    for needle in candidates:
        matches = recommender.search(needle, limit=1)
        if matches:
            suggestions[f"{recommender.title_for(matches[0]['movie_id'])}"] = matches[0]["movie_id"]
    return suggestions


def _format_title(item: dict) -> str:
    """Return ``Title (Year)`` for a recommendation row."""
    year = item.get("year")
    return f"{item['title']} ({year})" if year else item["title"]


def render_recommendations(recommender: MovieRecommender) -> None:
    """Render the seed selection and recommendation list."""
    section(
        "Get recommendations",
        "Search for a film to find similar titles, or pick a user to see what the "
        "collaborative model predicts for their taste.",
    )

    method = st.radio(
        "Recommendation method",
        options=["content", "collaborative"],
        format_func=lambda key: METHOD_LABELS[key],
        horizontal=True,
        key="rec_method",
    )

    top_k = st.slider("Number of recommendations", min_value=3, max_value=30, value=DEFAULT_TOP_K, step=1)

    seed_id: int | None = None
    user_id: int | None = None

    if method == "content":
        suggestions = _suggested_titles(recommender)
        left, right = st.columns([2, 1])
        with left:
            query = st.text_input(
                "Search for a film",
                value="Toy Story",
                placeholder="Type part of a title, e.g. 'matrix'",
                key="rec_query",
            )
        with right:
            quick = st.selectbox(
                "Or pick a popular film",
                options=[""] + list(suggestions),
                key="rec_quick",
            )

        matches = recommender.search(quick or query, limit=300)
        if not matches:
            st.warning(
                f"No film in the catalogue matches '{(quick or query).strip()}'. "
                "Try a different title."
            )
            return

        labels = {f"{item['title']}": item["movie_id"] for item in matches[:300]}
        chosen = st.selectbox(
            f"{len(matches)} title(s) found",
            options=list(labels),
            index=0,
            key="rec_seed",
        )
        seed_id = labels[chosen]
    else:
        if recommender.collaborative is None:
            st.warning(
                "The collaborative model was not built. Re-run `python train.py` "
                "without `--skip-collaborative`."
            )
            return
        user_ids = recommender.collaborative.user_ids.tolist()
        user_id = st.selectbox(
            "User",
            options=[int(value) for value in user_ids],
            format_func=lambda value: f"User {value}",
            index=0,
            key="rec_user",
        )

    if st.button("Recommend", type="primary", width="stretch"):
        try:
            results = recommender.recommend(
                movie_id=seed_id,
                user_id=user_id,
                method=method,
                top_k=top_k,
            )
        except Exception as error:  # noqa: BLE001 - UI boundary
            show_error(error)
            return

        if not results:
            st.info(
                "No recommendations could be produced. For the collaborative model this "
                "usually means the user has already rated nearly everything in the catalogue."
            )
            return

        if method == "content" and seed_id is not None:
            seed_label = _format_title(
                {
                    "title": recommender.title_for(seed_id),
                    "year": recommender.content.years.get(int(seed_id)),
                }
            )
            result_card("Films similar to", seed_label, accent=PALETTE["accent"])
        else:
            result_card("Recommendations for", f"User {user_id}", accent=PALETTE["accent"])

        rows = []
        for rank, item in enumerate(results, start=1):
            row = {
                "rank": rank,
                "title": item["title"],
                "year": item.get("year") if item.get("year") else "",
                "genres": ", ".join(item.get("genres", [])[:4]) or "-",
            }
            if "predicted_rating" in item:
                row["predicted_rating"] = round(float(item["predicted_rating"]), 2)
            else:
                row["similarity"] = round(float(item["similarity"]), 4)
            rows.append(row)

        show_dataframe(
            rows,
            caption=(
                "Cosine similarity between the TF-IDF vectors of each film and the seed."
                if method == "content"
                else "Predicted rating is the film's mean rating plus its cosine similarity "
                "to the user in latent-factor space."
            ),
        )

        show_figure(
            bar_chart(
                [f"{item['title'][:26]}" for item in results],
                [float(item["similarity"]) for item in results],
                title="Similarity to seed" if method == "content" else "Affinity in latent space",
                ylabel="Cosine similarity",
                width=8.4,
                height=4.0,
            )
        )


def render_evaluation(recommender: MovieRecommender) -> None:
    """Render the offline evaluation table."""
    evaluation = recommender.evaluation or {}
    methods = evaluation.get("methods") or {}
    if not methods:
        return

    section(
        "Offline evaluation",
        "Leave-one-out: one highly rated film per user is hidden, the model ranks the "
        "catalogue, and a hit counts only if that film lands in the top K.",
    )

    rows = []
    for method, payload in methods.items():
        row = {"method": METHOD_LABELS.get(method, method), "n_users": payload.get("n_scored_users")}
        for k in evaluation.get("ks", EVALUATION_KS):
            row[f"P@{k}"] = round(float(payload["precision_at_k"].get(str(k), 0.0)), 4)
        row["hit_rate"] = round(float(payload["hit_rate"]), 4)
        rank = payload.get("mean_rank_when_hit")
        row["mean_rank_when_hit"] = round(float(rank), 1) if rank is not None else None
        row["catalogue_coverage"] = round(float(payload["catalogue_coverage"]), 4)
        rows.append(row)

    show_dataframe(rows, caption="Precision@K for each method on the same held-out task.")

    chart_ks = [str(k) for k in evaluation.get("ks", EVALUATION_KS)]
    for method, payload in methods.items():
        show_figure(
            bar_chart(
                [f"K={k}" for k in chart_ks],
                [float(payload["precision_at_k"].get(k, 0.0)) for k in chart_ks],
                title=f"{METHOD_LABELS.get(method, method)} - Precision@K",
                ylabel="Precision@K",
                width=6.0,
                height=3.2,
            )
        )

    st.caption(
        f"A positive interaction is a rating of {evaluation.get('positive_threshold', POSITIVE_RATING_THRESHOLD)} "
        f"or higher, and {evaluation.get('n_users_evaluated')} users were sampled with seed "
        f"{evaluation.get('seed')}."
    )
    st.caption(
        "Recall@K is equal to Precision@K in this setup because each user contributes "
        "exactly one held-out film."
    )


def render_dataset_panel(recommender: MovieRecommender) -> None:
    """Show dataset provenance and the popularity distribution."""
    with st.expander("Dataset details"):
        summary = recommender.dataset or {}
        metric_row(
            [
                ("Films", f"{summary.get('n_movies', 0):,}", "MovieLens catalogue"),
                ("Users", f"{summary.get('n_users', 0):,}", "rated at least once"),
                ("Ratings", f"{summary.get('n_ratings', 0):,}", "0.5 to 5.0 stars"),
                ("Tags", f"{summary.get('n_tags', 0):,}", f"{summary.get('movies_with_tags', 0):,} films"),
            ]
        )
        st.markdown(f"**Source:** `{summary.get('source', 'n/a')}`")
        st.caption(
            "MovieLens does not redistribute plot overviews because they are licensed "
            "from IMDb, so the text half of the content model uses crowdsourced tags. "
            "This is documented in the project README."
        )


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Movie Recommendation System", page_icon="04", layout="wide")
    inject_theme()

    page_header(
        "Movie Recommendation System",
        "Two recommenders over the MovieLens catalogue: content-based similarity on "
        "genres and tags, and latent-factor collaborative filtering, compared on the "
        "same offline evaluation.",
        eyebrow="Project 04",
        chips=["Recommender system", "TF-IDF + cosine", "Truncated SVD", "MovieLens"],
    )

    recommender = _get_recommender()
    if recommender is None:
        st.warning(
            f"**No trained model found.** The app expects `{portable_display(MODEL_PATH)}`.\n\n"
            "Build it by running:\n\n"
            "```bash\npython train.py\n```"
        )
        note(DISCLAIMER)
        return

    render_recommendations(recommender)
    render_evaluation(recommender)
    render_dataset_panel(recommender)

    metrics = recommender.metrics or {}
    model_info_panel(
        algorithm="TF-IDF cosine similarity + truncated SVD (64 latent factors)",
        trained_on=f"{metrics.get('n_movies', 0):,} films from MovieLens ml-latest-small",
        metrics={
            "precision@10 (content)": (recommender.evaluation.get("methods", {})
                                       .get("content", {})
                                       .get("precision_at_k", {})
                                       .get("10", 0.0)),
            "precision@10 (collaborative)": (recommender.evaluation.get("methods", {})
                                             .get("collaborative", {})
                                             .get("precision_at_k", {})
                                             .get("10", 0.0)),
            "tfidf_features": float(metrics.get("n_tfidf_features", 0)),
            "svd_factors": float(metrics.get("n_svd_factors", 0)),
        },
        extra={
            "Trained at": recommender.trained_at,
            "SVD variance explained": (
                format_metric(metrics["svd_explained_variance"])
                if metrics.get("svd_explained_variance") is not None
                else "n/a"
            ),
            "Positive threshold": f"rating >= {POSITIVE_RATING_THRESHOLD}",
            "Evaluation": "leave-one-out over sampled users",
        },
    )

    note(DISCLAIMER)


if __name__ == "__main__":
    main()