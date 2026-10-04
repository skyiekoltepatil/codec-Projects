"""Two recommender models built on MovieLens, plus an offline evaluation harness.

Content-based (primary)
-----------------------
Each movie becomes a TF-IDF document built from its genres and crowdsourced tags
(see :meth:`src.data.MovieLensData.content_text`). Pairwise cosine similarity
between those documents yields "movies like this one". This works immediately for
every film in the catalogue, including ones nobody has rated, which is why it is
the model the interface leads with.

Collaborative filtering (secondary)
----------------------------------
Latent-factor SVD over a user-by-item rating matrix. A user is represented in the
same reduced space as the items they rated, and their nearest neighbours in that
space become their recommendations. This captures taste patterns that metadata
cannot express - "people who like this also liked that" - but it can only
recommend items the user or their neighbours have already interacted with, so it
needs interaction history and a fallback for cold-start users.

Evaluation
----------
Both models are scored on a leave-some-out split of the ratings: for each user a
portion of their ratings is held out, the model ranks the rest of the catalogue,
and Precision@K / Recall@K are averaged over users. Because a movie the user has
already rated is not a useful recommendation, every candidate is filtered against
the user's own history. Content-based filtering has no natural train/test split
(it fits no user parameters), so it is evaluated on the same task for
comparability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from shared.errors import InvalidInputError, ModelNotFoundError
from shared.metrics import coverage
from shared.ml import RANDOM_SEED, load_joblib, save_joblib, save_json, set_global_seed
from shared.paths import models_dir

from src.data import MovieLensData

logger = logging.getLogger(__name__)

PROJECT_SLUG = "04-movie-recommendation-system"

#: TF-IDF settings for the content model.
#:
#: ``min_df=1`` keeps every tag term. Tags are the only free-text signal in
#: MovieLens and are already rare (1,572 of 9,742 films carry any tag at all),
#: so pruning rare terms throws away most of the discriminative vocabulary.
TFIDF_PARAMS: dict[str, Any] = {
    "min_df": 1,
    "max_df": 0.6,
    "sublinear_tf": True,
    "ngram_range": (1, 1),
    "norm": "l2",
}

#: Latent-factor settings for the collaborative model. 64 factors is a common
#: upper bound for a 610-user dataset; the number actually kept is chosen at fit
#: time from the explained-variance curve (see :data:`VARIANCE_TARGET`), because
#: surplus factors inject noise into the cosine similarity.
SVD_PARAMS: dict[str, Any] = {
    "n_components": 64,
    "n_iter": 20,
    "random_state": RANDOM_SEED,
}

#: Cumulative explained variance that the retained latent factors should reach.
VARIANCE_TARGET = 0.95

#: Ratings at or above this count as a positive (the user liked the film).
POSITIVE_RATING_THRESHOLD = 4.0

#: Default number of recommendations returned to the user.
DEFAULT_TOP_K = 10

#: K values reported in the evaluation table.
EVALUATION_KS: tuple[int, ...] = (5, 10, 20)


# --------------------------------------------------------------------------- #
# Content-based model
# --------------------------------------------------------------------------- #


@dataclass
class ContentRecommender:
    """TF-IDF over movie metadata, matched by cosine similarity."""

    vectorizer: Any
    matrix: Any
    movie_ids: np.ndarray
    titles: dict[int, str]
    genres: dict[int, list[str]]
    years: dict[int, int | None]
    n_features: int = 0

    @property
    def n_movies(self) -> int:
        """Number of films in the similarity matrix."""
        return int(len(self.movie_ids))

    def _index_of(self, movie_id: int) -> int:
        position = np.searchsorted(self.movie_ids, movie_id)
        if position >= len(self.movie_ids) or int(self.movie_ids[position]) != int(movie_id):
            raise InvalidInputError(
                f"Movie {movie_id} is not in the catalogue.",
                hint="Pick a film from the search box in the interface.",
            )
        return int(position)

    def similar_items(self, movie_id: int, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Return the ``top_k`` films most similar to ``movie_id``.

        The query film itself is excluded, and films with no similarity at all
        (an all-zero row) are dropped rather than returned with a score of 0.0.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        index = self._index_of(movie_id)
        query = self.matrix[index]
        scores = cosine_similarity(query, self.matrix).ravel()

        order = np.argsort(scores)[::-1]
        results: list[dict[str, Any]] = []
        for position in order:
            if int(position) == index:
                continue
            score = float(scores[position])
            if score <= 0.0:
                break
            results.append(self._describe(int(self.movie_ids[position]), score))
            if len(results) >= max(1, int(top_k)):
                break
        return results

    def _describe(self, movie_id: int, score: float) -> dict[str, Any]:
        """Bundle a movie's identity and metadata with a similarity score."""
        return {
            "movie_id": int(movie_id),
            "title": self.titles.get(int(movie_id), f"Movie {movie_id}"),
            "genres": self.genres.get(int(movie_id), []),
            "year": self.years.get(int(movie_id)),
            "similarity": float(score),
        }


# --------------------------------------------------------------------------- #
# Collaborative filtering model
# --------------------------------------------------------------------------- #


@dataclass
class CollaborativeRecommender:
    """Truncated-SVD latent factors over the user-by-item rating matrix."""

    item_factors: np.ndarray
    movie_ids: np.ndarray
    user_factors: np.ndarray
    user_ids: np.ndarray
    titles: dict[int, str]
    genres: dict[int, list[str]]
    years: dict[int, int | None]
    mean_rating: dict[int, float]
    n_factors: int = 0
    #: Films each user positively rated, keyed by user id. Needed at inference
    #: time to avoid recommending something the user has already seen.
    seen: dict[int, set[int]] = field(default_factory=dict)

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = DEFAULT_TOP_K,
        *,
        exclude: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` highest-predicted unseen films for ``user_id``.

        Args:
            user_id: The user to recommend for.
            top_k: Number of results.
            exclude: Films to withhold. Defaults to everything the user has
                positively rated. The offline evaluation passes a narrower set so
                that the held-out film stays reachable.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        position = np.searchsorted(self.user_ids, user_id)
        if position >= len(self.user_ids) or int(self.user_ids[position]) != int(user_id):
            raise InvalidInputError(
                f"User {user_id} has no rating history.",
                hint="This dataset contains 610 users; pick one from the interface.",
            )

        blocked = self._seen_by_user(user_id) if exclude is None else {int(item) for item in exclude}

        scores = cosine_similarity(
            self.user_factors[position].reshape(1, -1), self.item_factors
        ).ravel()

        order = np.argsort(scores)[::-1]
        results: list[dict[str, Any]] = []
        for item_position in order:
            movie_id = int(self.movie_ids[item_position])
            if movie_id in blocked:
                continue
            results.append(
                {
                    "movie_id": movie_id,
                    "title": self.titles.get(movie_id, f"Movie {movie_id}"),
                    "genres": self.genres.get(movie_id, []),
                    "year": self.years.get(movie_id),
                    "similarity": float(scores[item_position]),
                    "predicted_rating": float(
                        self.mean_rating.get(movie_id, 0.0) + scores[item_position]
                    ),
                }
            )
            if len(results) >= max(1, int(top_k)):
                break
        return results

    def _seen_by_user(self, user_id: int) -> set[int]:
        """Return the films a user has already positively rated."""
        return set(self.seen.get(int(user_id), ()))


@dataclass
class MovieRecommender:
    """The fitted pair of models plus the catalogue needed to serve results."""

    content: ContentRecommender
    collaborative: CollaborativeRecommender | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    trained_at: str = ""

    def title_for(self, movie_id: int) -> str:
        """Return a film's display title."""
        return self.content.titles.get(int(movie_id), f"Movie {movie_id}")

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return catalogue entries whose title contains ``query``, case-insensitively."""
        needle = str(query).strip().lower()
        if not needle:
            return []
        matches = [
            {"movie_id": movie_id, "title": title}
            for movie_id, title in self.content.titles.items()
            if needle in title.lower()
        ]
        matches.sort(key=lambda item: item["title"])
        return matches[:limit]

    def recommend(
        self,
        movie_id: int | None = None,
        *,
        user_id: int | None = None,
        method: str = "content",
        top_k: int = DEFAULT_TOP_K,
        exclude: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return recommendations from the requested method.

        Args:
            movie_id: Seed film, required for ``method="content"``.
            user_id: Seed user, required for ``method="collaborative"``.
            method: ``"content"`` or ``"collaborative"``.
            top_k: Number of recommendations.
            exclude: Films to withhold; only supported by ``"collaborative"``.

        Raises:
            InvalidInputError: When the seed is missing or the method is unknown.
        """
        if method == "content":
            if movie_id is None:
                raise InvalidInputError(
                    "Choose a film to get similar recommendations.",
                    hint="Use the search box above to select a seed movie.",
                )
            return self.content.similar_items(movie_id, top_k)

        if method == "collaborative":
            if self.collaborative is None:
                raise InvalidInputError(
                    "The collaborative model was not built.",
                    hint="Re-run `python train.py` to rebuild both models.",
                )
            if user_id is None:
                raise InvalidInputError(
                    "Choose a user to get personalised recommendations.",
                    hint="Use the user selector above.",
                )
            return self.collaborative.recommend_for_user(user_id, top_k, exclude=exclude)

        raise InvalidInputError(
            f"Unknown recommendation method '{method}'.",
            hint="Choose either 'content' or 'collaborative'.",
        )


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _catalogue_maps(data: MovieLensData) -> tuple[dict[int, str], dict[int, list[str]], dict[int, int | None]]:
    """Return title, genre and year lookups keyed by movie id."""
    genres = data.genre_lists()
    titles: dict[int, str] = {}
    years: dict[int, int | None] = {}
    genre_map: dict[int, list[str]] = {}
    for row, genre_list in zip(data.movies.itertuples(), genres, strict=True):
        movie_id = int(row.movieId)
        titles[movie_id] = str(row.title_clean)
        year = getattr(row, "year", None)
        years[movie_id] = int(year) if pd.notna(year) else None
        genre_map[movie_id] = genre_list
    return titles, genre_map, years


def build_content_model(data: MovieLensData) -> ContentRecommender:
    """Fit TF-IDF on movie metadata and build the cosine similarity matrix."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = data.content_text()
    if len(documents) < 2:
        raise InvalidInputError(
            f"Only {len(documents)} content document(s) available; at least 2 are needed.",
            hint="Re-download MovieLens; the genres and tags columns look incomplete.",
        )

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError as error:
        # The realistic cause is that every document is empty, which means the
        # genres column was not parsed. scikit-learn's own message ("empty
        # vocabulary") is accurate but does not tell the user what to do.
        raise InvalidInputError(
            "Could not build any TF-IDF features from the movie metadata.",
            hint=(
                "This happens when the genres and tags columns are empty or unreadable. "
                f"Re-download MovieLens and check that genres are pipe-separated. ({error})"
            ),
        ) from error

    titles, genres, years = _catalogue_maps(data)
    return ContentRecommender(
        vectorizer=vectorizer,
        matrix=matrix,
        movie_ids=data.movies["movieId"].to_numpy(),
        titles=titles,
        genres=genres,
        years=years,
        n_features=int(len(vectorizer.vocabulary_)),
    )


def build_collaborative_model(
    data: MovieLensData,
    *,
    positive_threshold: float = POSITIVE_RATING_THRESHOLD,
) -> CollaborativeRecommender:
    """Fit truncated SVD on the implicit user-by-item matrix.

    Ratings below ``positive_threshold`` are treated as *no signal* rather than as
    a negative preference. With a 0-filled matrix a 1-star and a 5-star entry are
    both simply "non-zero", so a user who disliked a film still pulls their latent
    factor towards the cluster of people who loved it. Binarising to the positive
    interactions removes that failure mode and matches how the recommender is
    actually used: what you liked, not what you disliked.
    """
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    ratings = data.ratings
    if len(ratings) < 4:
        raise InvalidInputError(
            f"Only {len(ratings)} ratings available for collaborative filtering.",
            hint="Re-download MovieLens; the ratings table looks incomplete.",
        )

    # Keep only positive interactions. Everything else becomes an implicit zero,
    # which is the standard "what you liked, not what you disliked" formulation.
    interactions = ratings[ratings["rating"] >= positive_threshold][
        ["userId", "movieId"]
    ].assign(weight=1.0)

    if interactions.empty:
        raise InvalidInputError(
            f"No ratings at or above {positive_threshold}.",
            hint="Lower positive_threshold and re-run.",
        )

    # Unrated pairs stay 0: a user who has not rated a film has expressed no
    # opinion about it, which is what "missing" means in the implicit view.
    pivot = interactions.pivot_table(
        index="userId", columns="movieId", values="weight", fill_value=0.0
    )
    user_ids = pivot.index.to_numpy()
    movie_ids = pivot.columns.to_numpy()

    # Truncated SVD cannot produce more components than the smaller matrix
    # dimension, and scikit-learn requires strictly fewer. Clamping keeps the
    # model buildable on small or heavily-filtered datasets instead of failing
    # deep inside the estimator.
    max_components = max(1, min(len(user_ids), len(movie_ids)) - 1)
    n_components = min(int(SVD_PARAMS["n_components"]), max_components)
    if n_components < int(SVD_PARAMS["n_components"]):
        logger.warning(
            "Reduced SVD components from %d to %d: the user-by-item matrix is only %dx%d.",
            int(SVD_PARAMS["n_components"]),
            n_components,
            len(user_ids),
            len(movie_ids),
        )

    matrix = csr_matrix(pivot.to_numpy(dtype=float))
    svd = TruncatedSVD(n_components=n_components, n_iter=int(SVD_PARAMS["n_iter"]),
                       random_state=int(SVD_PARAMS["random_state"]))
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_.T

    # When more components are requested than the matrix effectively has rank,
    # the surplus singular values are numerically tiny and their directions are
    # arbitrary. Cosine similarity in that subspace is dominated by noise, which
    # visibly degrades recommendations - unseen films end up all tied at ~0.
    # Retaining the smallest number of components that reaches
    # ``variance_target`` of explained variance fixes this, and on a well
    # conditioned dataset like MovieLens it simply keeps all the useful factors.
    explained_ratio = np.asarray(svd.explained_variance_ratio_, dtype=float)
    if explained_ratio.size:
        cumulative = np.cumsum(explained_ratio)
        target = float(VARIANCE_TARGET)
        at_target = int(np.searchsorted(cumulative, target) + 1)
        informative = max(1, min(at_target, int(explained_ratio.size)))
    else:
        informative = 1

    singular_values = np.asarray(svd.singular_values_, dtype=float)
    if singular_values.size:
        # Keep the second guard: drop factors whose singular value is
        # indistinguishable from zero relative to the largest one.
        tolerance = float(singular_values.max()) * 1e-6
        informative = min(informative, max(1, int((singular_values > tolerance).sum())))

    if informative < user_factors.shape[1]:
        logger.info(
            "Using %d of %d latent factors (%.1f%% cumulative explained variance).",
            informative,
            user_factors.shape[1],
            float(cumulative[informative - 1]) * 100 if explained_ratio.size else 0.0,
        )
        user_factors = user_factors[:, :informative]
        item_factors = item_factors[:, :informative]

    titles, genres, years = _catalogue_maps(data)
    mean_rating = ratings.groupby("movieId")["rating"].mean().to_dict()

    positive = ratings[ratings["rating"] >= positive_threshold]
    seen: dict[int, set[int]] = {
        int(user_id): set(group["movieId"].astype(int)) for user_id, group in positive.groupby("userId")
    }

    explained = float(svd.explained_variance_ratio_.sum())
    logger.info(
        "Collaborative model: %d users x %d movies, %d factors, %.1f%% variance explained",
        len(user_ids),
        len(movie_ids),
        int(user_factors.shape[1]),
        explained * 100,
    )

    model = CollaborativeRecommender(
        item_factors=item_factors,
        movie_ids=movie_ids,
        user_factors=user_factors,
        user_ids=user_ids,
        titles=titles,
        genres=genres,
        years=years,
        mean_rating={int(key): float(value) for key, value in mean_rating.items()},
        n_factors=int(user_factors.shape[1]),
        seen=seen,
    )
    model.explained_variance = explained
    return model


# --------------------------------------------------------------------------- #
# Offline evaluation
# --------------------------------------------------------------------------- #


def evaluate_offline(
    data: MovieLensData,
    recommender: MovieRecommender,
    *,
    ks: Sequence[int] = EVALUATION_KS,
    n_users: int = 200,
    positive_threshold: float = POSITIVE_RATING_THRESHOLD,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Score both recommenders with leave-one-out Precision@K and Recall@K.

    For each sampled user one liked film is held out and the model must rank it
    without ever being told it is the target:

    * The **collaborative** model is seeded with the user id and simply filters
      out films that user has already rated.
    * The **content** model cannot be seeded with the held-out film, because a
      model never recommends the query back to itself. It is instead seeded with
      a *different* film the same user liked, which is the honest analogue of
      "I liked A, what else is like A?".

    Any other film the user already rated is filtered out of both lists, so the
    two methods face an identical candidate set and their numbers are comparable.
    """
    set_global_seed(seed)
    positive = data.ratings[data.ratings["rating"] >= positive_threshold]
    if positive.empty:
        raise InvalidInputError(
            f"No ratings at or above {positive_threshold}.",
            hint="Lower POSITIVE_RATING_THRESHOLD and re-run.",
        )

    rng = np.random.default_rng(seed)
    positive_by_user = {
        int(user): set(group["movieId"].astype(int)) for user, group in positive.groupby("userId")
    }

    # A user needs at least two liked films: one to hold out and one to seed from.
    eligible = [user for user, films in positive_by_user.items() if len(films) >= 2]
    if not eligible:
        raise InvalidInputError(
            "No user has two or more films rated above the threshold.",
            hint="Lower POSITIVE_RATING_THRESHOLD and re-run.",
        )

    chosen = [int(user) for user in rng.choice(sorted(eligible), size=min(int(n_users), len(eligible)), replace=False)]

    # A held-out film must be in the catalogue, otherwise no method can ever
    # retrieve it and the measurement would be meaningless.
    catalogue = set(recommender.content.movie_ids.tolist())

    max_k = max(int(k) for k in ks)
    evaluation: dict[str, Any] = {
        "n_users_evaluated": len(chosen),
        "positive_threshold": positive_threshold,
        "ks": [int(k) for k in ks],
        "seed": int(seed),
        "methods": {},
    }

    for method in ("content", "collaborative"):
        if method == "collaborative" and recommender.collaborative is None:
            continue

        hits_by_k = {int(k): 0 for k in ks}
        hit_positions: list[int] = []
        recommended_lists: list[list[int]] = []
        scored_users = 0
        skipped = 0

        for user_id in chosen:
            seen = positive_by_user.get(user_id, set()) & catalogue
            if len(seen) < 2:
                continue
            candidates = sorted(seen)
            held_out = int(candidates[int(rng.integers(len(candidates)))])
            seed_pool = [movie_id for movie_id in candidates if movie_id != held_out]

            if method == "content":
                seed_movie = int(seed_pool[int(rng.integers(len(seed_pool)))])
                try:
                    # Fetch extra candidates because the "already seen" filter
                    # below removes a variable number of them.
                    raw = recommender.content.similar_items(seed_movie, top_k=max_k * 4)
                except InvalidInputError as error:
                    logger.warning("Skipping user %s (content): %s", user_id, error)
                    skipped += 1
                    continue
            else:
                try:
                    raw = recommender.recommend(
                        user_id=user_id,
                        method="collaborative",
                        top_k=max_k * 4,
                        # Everything the user liked except the target: the target
                        # must stay reachable or the method could never score a hit.
                        exclude=seen - {held_out},
                    )
                except InvalidInputError as error:
                    logger.warning("Skipping user %s (collaborative): %s", user_id, error)
                    skipped += 1
                    continue

            # Everything the user liked other than the target is a non-recommendation.
            ranked = [
                int(item["movie_id"])
                for item in raw
                if int(item["movie_id"]) not in (seen - {held_out})
            ]

            scored_users += 1
            rank = ranked.index(held_out) + 1 if held_out in ranked else None
            if rank is not None:
                hit_positions.append(rank)
            for k in ks:
                if held_out in ranked[: int(k)]:
                    hits_by_k[int(k)] += 1
            recommended_lists.append(ranked[:max_k])

        if skipped:
            logger.warning("Method '%s' skipped %d of %d users.", method, skipped, len(chosen))

        if scored_users == 0:
            raise InvalidInputError(
                f"No user could be evaluated with the '{method}' model.",
                hint=(
                    "Every sampled user was skipped. This usually means the collaborative "
                    "model does not cover the sampled user ids, or the catalogue and "
                    "rating tables disagree."
                ),
            )

        # Each user contributes exactly one relevant held-out film, so
        # Precision@K and Recall@K coincide numerically. Both are reported so the
        # table reads the way a reviewer expects.
        scores = {int(k): hits_by_k[int(k)] / scored_users for k in ks}
        evaluation["methods"][method] = {
            "precision_at_k": {str(k): scores[int(k)] for k in ks},
            "recall_at_k": {str(k): scores[int(k)] for k in ks},
            "hit_rate": len(hit_positions) / scored_users,
            "n_scored_users": scored_users,
            "mean_rank_when_hit": float(np.mean(hit_positions)) if hit_positions else None,
            "catalogue_coverage": coverage(recommended_lists, recommender.content.n_movies),
        }

    if not evaluation["methods"]:
        raise InvalidInputError(
            "No method could be evaluated.",
            hint="Check that the ratings table loaded correctly.",
        )
    return evaluation


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_model(recommender: MovieRecommender, *, directory: Path | None = None) -> Path:
    """Persist the fitted recommenders and their metrics."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "recommender.joblib"
    save_joblib(recommender, path)
    return path


def load_model(path: Path | None = None) -> MovieRecommender:
    """Load a previously trained :class:`MovieRecommender`."""
    target = Path(path) if path else models_dir(PROJECT_SLUG) / "recommender.joblib"
    loaded = load_joblib(
        target,
        description="movie recommender",
        train_command="python train.py",
    )
    if not isinstance(loaded, MovieRecommender):
        raise ModelNotFoundError(
            f"'{target.name}' does not contain a MovieRecommender.",
            hint="Delete the file and re-run `python train.py`.",
        )
    return loaded


def save_metrics_report(report: dict[str, Any], *, directory: Path | None = None) -> Path:
    """Write the training report to ``models/training_metrics.json``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_json(report, target_dir / "training_metrics.json")


def build_training_report(
    recommender: MovieRecommender,
    *,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON report written next to the model."""
    return {
        "project": PROJECT_SLUG,
        "trained_at": recommender.trained_at,
        "algorithms": {
            "content_based": "TF-IDF over genres and tags + cosine similarity",
            "collaborative": "Truncated SVD latent factors on the implicit rating matrix",
        },
        "tfidf": {key: str(value) for key, value in TFIDF_PARAMS.items()},
        "svd": {key: str(value) for key, value in SVD_PARAMS.items()},
        "positive_rating_threshold": POSITIVE_RATING_THRESHOLD,
        "dataset": dataset,
        "evaluation": recommender.evaluation,
        "metrics": recommender.metrics,
        "random_seed": RANDOM_SEED,
        "notes": (
            "MovieLens does not redistribute plot overviews, so the text half of the "
            "content model uses crowdsourced tags instead. Collaborative filtering is "
            "evaluated on the same leave-one-out task as the content model so the two "
            "columns are directly comparable."
        ),
    }