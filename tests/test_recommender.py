"""Tests for Project 04 - Movie Recommendation System.

Tests that need the full MovieLens catalogue skip when the dataset has not been
downloaded. Everything else runs against a small in-memory fixture so the suite
stays fast and does not require the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from project_env import REPO_ROOT, use_project

use_project("04-movie-recommendation-system")

sys.path.insert(0, str(REPO_ROOT))

from shared.errors import InvalidInputError  # noqa: E402
from src.data import MovieLensData, load_dataset  # noqa: E402
from src.model import (  # noqa: E402
    MovieRecommender,
    build_collaborative_model,
    build_content_model,
    build_training_report,
    evaluate_offline,
    load_model,
    save_model,
)


# --------------------------------------------------------------------------- #
# Fixture: a tiny synthetic "MovieLens" with known structure
# --------------------------------------------------------------------------- #


#: Genres used by the synthetic fixture, and the film ids belonging to each.
FIXTURE_GENRES = ("Science Fiction", "Comedy", "Drama")

#: Films per genre in the synthetic fixture.
FIXTURE_FILMS_PER_GENRE = 20

#: Users per taste group in the synthetic fixture.
FIXTURE_USERS_PER_GROUP = 10


@pytest.fixture()
def tiny_movielens() -> MovieLensData:
    """Return a synthetic dataset with three clearly separated taste groups.

    The fixture is larger than a hand-written table on purpose. A 4x6 rating
    matrix has rank 2, so truncated SVD has no room to express three distinct
    tastes and would return arbitrary neighbours. With 60 films and 30 users the
    latent structure is real, which makes "a sci-fi fan gets sci-fi films" an
    assertion the model has to genuinely satisfy.

    Users are split into three groups; each group rates 12 films of its own
    genre highly (4.0-5.0) and 8 films of other genres poorly (0.5-2.0).

    Within a group, user *j* rates genre films ``[j, j+1, ..., j+11] mod 20``. That
    sliding window matters: every film is rated by several group members while
    each individual user still leaves 8 same-genre films unseen. Those unseen
    films are exactly what collaborative filtering is supposed to surface, and
    they have to be rated by somebody or no model could ever recommend them.
    """
    movie_rows = []
    by_genre: dict[str, list[int]] = {genre: [] for genre in FIXTURE_GENRES}
    movie_id = 0
    for genre in FIXTURE_GENRES:
        for _ in range(FIXTURE_FILMS_PER_GENRE):
            movie_id += 1
            by_genre[genre].append(movie_id)
            movie_rows.append(
                {
                    "movieId": movie_id,
                    "title": f"{genre} film {len(by_genre[genre])} (2000)",
                    "genres": genre,
                }
            )
    movies = pd.DataFrame(movie_rows)
    movies["title_clean"] = [title.split(" (")[0] for title in movies["title"]]
    movies["year"] = pd.array([2000] * len(movies), dtype="Int64")

    rating_rows = []
    for index in range(len(FIXTURE_GENRES) * FIXTURE_USERS_PER_GROUP):
        group = index // FIXTURE_USERS_PER_GROUP
        position_in_group = index % FIXTURE_USERS_PER_GROUP
        user_id = index + 1
        preferred = FIXTURE_GENRES[group % len(FIXTURE_GENRES)]

        genre_films = by_genre[preferred]
        liked = [genre_films[(position_in_group + offset) % FIXTURE_FILMS_PER_GENRE] for offset in range(12)]

        others = [film for genre in FIXTURE_GENRES if genre != preferred for film in by_genre[genre]]
        disliked = [others[(position_in_group + offset) % len(others)] for offset in range(8)]

        for slot, film in enumerate(liked):
            rating_rows.append(
                {"userId": user_id, "movieId": film, "rating": 4.0 + 0.5 * (slot % 2), "timestamp": 0}
            )
        for slot, film in enumerate(disliked):
            rating_rows.append(
                {"userId": user_id, "movieId": film, "rating": 0.5 + 0.5 * (slot % 4), "timestamp": 0}
            )

    tag_rows = []
    for genre in FIXTURE_GENRES:
        for film in by_genre[genre][:2]:
            tag_rows.append({"userId": 1, "movieId": film, "tag": f"{genre.lower()}-tag", "timestamp": 0})

    return MovieLensData(
        movies=movies,
        ratings=pd.DataFrame(rating_rows),
        tags=pd.DataFrame(tag_rows),
    )


def _genre_of(fixture: MovieLensData, movie_id: int) -> str:
    """Return the single genre of a fixture film."""
    row = fixture.movies[fixture.movies["movieId"] == movie_id]
    return str(row["genres"].iloc[0])


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #


def test_content_text_combines_genres_and_tags(tiny_movielens):
    """Each movie's document contains its genre and any tags, lowercased."""
    documents = tiny_movielens.content_text()
    assert len(documents) == len(tiny_movielens.movies)

    tagged = tiny_movielens.movies.loc[1, "movieId"]
    first = documents.loc[tagged]
    assert "science fiction" in first
    assert "science fiction-tag" in first
    # Genres are repeated so they are not drowned out by long tag lists.
    assert first.count("science fiction") > 1


def test_content_text_handles_movie_without_tags(tiny_movielens):
    """A film with no tags still produces a genre-only document."""
    documents = tiny_movielens.content_text()
    untagged = tiny_movielens.movies.loc[5, "movieId"]
    assert documents.loc[untagged].strip() != ""
    assert "-tag" not in documents.loc[untagged]


def test_genre_lists_exclude_the_no_genres_placeholder(tiny_movielens):
    """MovieLens' '(no genres listed)' marker is not surfaced as a genre."""
    extended = tiny_movielens.movies.copy()
    extended.loc[extended.index[0], "genres"] = "Action|(no genres listed)"
    placeholder = MovieLensData(movies=extended, ratings=tiny_movielens.ratings, tags=tiny_movielens.tags)
    assert "(no genres listed)" not in placeholder.genre_lists()[0]


def test_summary_counts_are_consistent(tiny_movielens):
    """Summary counts match the underlying tables."""
    summary = tiny_movielens.summary()
    assert summary["n_movies"] == len(tiny_movielens.movies)
    assert summary["n_users"] == tiny_movielens.ratings["userId"].nunique()
    assert summary["n_ratings"] == len(tiny_movielens.ratings)


# --------------------------------------------------------------------------- #
# Content-based model
# --------------------------------------------------------------------------- #


def test_content_model_finds_same_genre_neighbours(tiny_movielens):
    """A seed retrieves other films of the same genre first."""
    recommender = build_content_model(tiny_movielens)
    seed = int(tiny_movielens.movies.loc[0, "movieId"])
    results = recommender.similar_items(seed, top_k=2)
    assert len(results) == 2
    assert _genre_of(tiny_movielens, seed) == "Science Fiction"
    for item in results:
        assert item["similarity"] > 0
        assert item["genres"] == ["Science Fiction"]


def test_content_model_excludes_the_seed_itself(tiny_movielens):
    """A film is never recommended back to itself."""
    recommender = build_content_model(tiny_movielens)
    seed = int(tiny_movielens.movies.loc[0, "movieId"])
    results = recommender.similar_items(seed, top_k=5)
    assert seed not in {item["movie_id"] for item in results}


def test_content_model_rejects_unknown_movie(tiny_movielens):
    """An unknown movie id raises an actionable error."""
    recommender = build_content_model(tiny_movielens)
    with pytest.raises(InvalidInputError) as error:
        recommender.similar_items(9999)
    assert "9999" in str(error.value)


def test_content_model_reports_only_positive_similarity(tiny_movielens):
    """Films with no overlap are dropped rather than returned at score 0."""
    recommender = build_content_model(tiny_movielens)
    seed = int(tiny_movielens.movies.loc[0, "movieId"])
    results = recommender.similar_items(seed, top_k=100)
    assert results, "expected at least some neighbours"
    assert all(item["similarity"] > 0 for item in results)


def test_search_matches_case_insensitively(tiny_movielens):
    """Title search ignores case and tolerates an empty query."""
    recommender = MovieRecommender(content=build_content_model(tiny_movielens))
    matches = recommender.search("COMEDY film 1")
    assert matches, "expected at least one match"
    # Every hit contains the needle, and alphabetical ordering puts the exact
    # title before its longer siblings such as "Comedy film 10".
    assert all("comedy film 1" in item["title"].lower() for item in matches)
    assert matches[0]["title"] == "Comedy film 1"
    assert recommender.search("") == []
    assert recommender.search("zzz") == []


# --------------------------------------------------------------------------- #
# Collaborative model
# --------------------------------------------------------------------------- #


def test_collaborative_model_separates_tastes(tiny_movielens):
    """A user's recommendations are mostly their own taste group.

    Every user in the fixture rates 12 films of one genre highly and 8 films of
    other genres poorly, so a model that has learned anything should return
    mostly same-genre films. With three genres, chance would score about 1.7 of
    5; the measured mean is above 3, and every user clears 2. Perfect purity is
    deliberately not demanded - this is a 60-film fixture with three synthetic
    clusters, and asserting perfection would only make the test brittle.
    """
    model = build_collaborative_model(tiny_movielens)
    positive = tiny_movielens.ratings[tiny_movielens.ratings["rating"] >= 4.0]

    purities: list[int] = []
    for user_id in model.user_ids.tolist():
        preferred = _genre_of(
            tiny_movielens,
            int(positive[positive["userId"] == user_id]["movieId"].iloc[0]),
        )
        results = model.recommend_for_user(int(user_id), top_k=5)
        assert results, f"no recommendations for user {user_id}"
        purities.append(
            sum(1 for item in results if _genre_of(tiny_movielens, item["movie_id"]) == preferred)
        )

    assert len(purities) == len(FIXTURE_GENRES) * FIXTURE_USERS_PER_GROUP
    assert min(purities) >= 2, f"a user got only {min(purities)}/5 same-genre films"
    assert sum(purities) / len(purities) >= 3.0, (
        f"mean same-genre purity was {sum(purities) / len(purities):.2f}/5, "
        "which is barely above the ~1.67 chance level"
    )


def test_collaborative_model_excludes_already_rated_films(tiny_movielens):
    """By default, films the user already rated highly are never recommended."""
    model = build_collaborative_model(tiny_movielens)
    positive = tiny_movielens.ratings[tiny_movielens.ratings["rating"] >= 4.0]
    user_id = int(model.user_ids[0])
    liked = set(positive[positive["userId"] == user_id]["movieId"].astype(int))

    results = model.recommend_for_user(user_id, top_k=20)
    assert not ({item["movie_id"] for item in results} & liked)


def test_collaborative_exclude_argument_can_narrow_the_filter(tiny_movielens):
    """An explicit exclusion set overrides the default 'already seen' filter.

    With every film the user rated blocked except one, that film becomes
    reachable - which is exactly what the offline evaluation depends on.
    """
    model = build_collaborative_model(tiny_movielens)
    positive = tiny_movielens.ratings[tiny_movielens.ratings["rating"] >= 4.0]
    user_id = int(model.user_ids[0])
    liked = sorted(positive[positive["userId"] == user_id]["movieId"].astype(int))
    target = liked[0]

    results = model.recommend_for_user(user_id, top_k=5, exclude=set(liked) - {target})
    assert results[0]["movie_id"] == target


def test_collaborative_model_rejects_unknown_user(tiny_movielens):
    """An unknown user id raises an actionable error."""
    model = build_collaborative_model(tiny_movielens)
    with pytest.raises(InvalidInputError):
        model.recommend_for_user(4242)


# --------------------------------------------------------------------------- #
# Recommendation dispatch
# --------------------------------------------------------------------------- #


def test_recommend_requires_the_right_seed(tiny_movielens):
    """Each method needs its own seed and rejects unknown method names."""
    recommender = MovieRecommender(
        content=build_content_model(tiny_movielens),
        collaborative=build_collaborative_model(tiny_movielens),
    )
    with pytest.raises(InvalidInputError):
        recommender.recommend(None, method="content")
    with pytest.raises(InvalidInputError):
        recommender.recommend(None, method="collaborative")
    with pytest.raises(InvalidInputError):
        recommender.recommend(
            int(tiny_movielens.movies.loc[0, "movieId"]), method="telepathy"
        )


def test_recommend_collaborative_without_model_is_reported(tiny_movielens):
    """Asking for collaborative results when only content was built is explained."""
    recommender = MovieRecommender(content=build_content_model(tiny_movielens), collaborative=None)
    with pytest.raises(InvalidInputError) as error:
        recommender.recommend(user_id=1, method="collaborative")
    assert "train.py" in (error.value.hint or "")


def test_top_k_is_respected(tiny_movielens):
    """Exactly ``top_k`` results are returned when enough similar films exist.

    The fixture has 20 films per genre, so a science-fiction seed has 19
    genuine neighbours and no padding is ever needed.
    """
    recommender = MovieRecommender(content=build_content_model(tiny_movielens))
    seed = int(tiny_movielens.movies.loc[0, "movieId"])
    assert len(recommender.recommend(seed, method="content", top_k=1)) == 1
    assert len(recommender.recommend(seed, method="content", top_k=3)) == 3


# --------------------------------------------------------------------------- #
# Offline evaluation
# --------------------------------------------------------------------------- #


def test_evaluation_reports_both_methods(tiny_movielens):
    """Both methods are scored on the same held-out task."""
    recommender = MovieRecommender(
        content=build_content_model(tiny_movielens),
        collaborative=build_collaborative_model(tiny_movielens),
    )
    evaluation = evaluate_offline(tiny_movielens, recommender, ks=(2, 5), n_users=4)

    assert set(evaluation["methods"]) == {"content", "collaborative"}
    for payload in evaluation["methods"].values():
        assert set(payload["precision_at_k"]) == {"2", "5"}
        assert 0.0 <= payload["hit_rate"] <= 1.0
        assert 0.0 <= payload["catalogue_coverage"] <= 1.0


def test_evaluation_is_deterministic(tiny_movielens):
    """The same seed produces the same evaluation numbers."""
    recommender = MovieRecommender(
        content=build_content_model(tiny_movielens),
        collaborative=build_collaborative_model(tiny_movielens),
    )
    first = evaluate_offline(tiny_movielens, recommender, ks=(3,), n_users=4, seed=7)
    second = evaluate_offline(tiny_movielens, recommender, ks=(3,), n_users=4, seed=7)
    assert first["methods"] == second["methods"]


def test_evaluation_rejects_users_without_enough_history(tiny_movielens):
    """A dataset where nobody has two liked films fails clearly."""
    # Keep only the first positive rating per user, so no user can have an item
    # held out while still having a seed.
    positives = tiny_movielens.ratings[tiny_movielens.ratings["rating"] >= 4.0]
    one_each = positives.groupby("userId", as_index=False).head(1)
    thin = MovieLensData(
        movies=tiny_movielens.movies,
        ratings=one_each,
        tags=tiny_movielens.tags,
    )
    recommender = MovieRecommender(content=build_content_model(tiny_movielens))
    with pytest.raises(InvalidInputError):
        evaluate_offline(thin, recommender, ks=(5,), n_users=4)


# --------------------------------------------------------------------------- #
# Persistence and report
# --------------------------------------------------------------------------- #


def test_save_load_round_trip(tiny_movielens, tmp_path):
    """A saved recommender reloads and returns identical results."""
    content = build_content_model(tiny_movielens)
    recommender = MovieRecommender(content=content, collaborative=build_collaborative_model(tiny_movielens))
    recommender.trained_at = "2026-01-01T00:00:00+00:00"

    path = save_model(recommender, directory=tmp_path)
    assert path.exists()
    reloaded = load_model(path)

    assert reloaded.trained_at == recommender.trained_at
    assert reloaded.content.titles == recommender.content.titles
    assert reloaded.recommend(1, method="content", top_k=2) == recommender.recommend(1, method="content", top_k=2)


def test_load_rejects_foreign_artefact(tmp_path, tiny_movielens):
    """A joblib file holding the wrong object is reported clearly."""
    from shared.errors import ModelNotFoundError
    from shared.ml import save_joblib

    save_joblib({"not": "a recommender"}, tmp_path / "recommender.joblib")
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "recommender.joblib")


def test_load_missing_artefact_explains_remedy(tmp_path):
    """A missing artefact tells the user which command to run."""
    with pytest.raises(Exception) as error:
        load_model(tmp_path / "nope.joblib")
    assert "train.py" in (error.value.hint or "")


def test_training_report_is_json_serialisable(tiny_movielens):
    """The report written to disk contains real measured numbers."""
    import json

    recommender = MovieRecommender(
        content=build_content_model(tiny_movielens),
        collaborative=build_collaborative_model(tiny_movielens),
    )
    recommender.evaluation = evaluate_offline(tiny_movielens, recommender, ks=(3,), n_users=4)
    recommender.metrics = {"n_movies": recommender.content.n_movies}
    recommender.trained_at = "2026-01-01T00:00:00+00:00"

    report = build_training_report(recommender, dataset=tiny_movielens.summary())
    assert json.loads(json.dumps(report))["project"] == "04-movie-recommendation-system"
    assert "collaborative" in report["algorithms"]


# --------------------------------------------------------------------------- #
# Real-dataset smoke test
# --------------------------------------------------------------------------- #


def _real_dataset_available() -> bool:
    """Return ``True`` when the MovieLens CSVs are already extracted."""
    return (REPO_ROOT / "04-movie-recommendation-system" / "data" / "ml-latest-small" / "movies.csv").exists()


@pytest.mark.skipif(
    not _real_dataset_available(),
    reason="MovieLens not downloaded. Run `python train.py` in project 04 once.",
)
def test_real_dataset_builds_and_recommends():
    """The genuine MovieLens data yields a working content model."""
    data = load_dataset()
    recommender = MovieRecommender(content=build_content_model(data))
    assert recommender.content.n_movies > 9_000

    matches = recommender.search("matrix", limit=5)
    assert matches
    results = recommender.recommend(matches[0]["movie_id"], method="content", top_k=5)
    assert len(results) == 5
    assert all(item["similarity"] > 0 for item in results)


@pytest.mark.skipif(
    not (REPO_ROOT / "04-movie-recommendation-system" / "models" / "recommender.joblib").exists(),
    reason="Trained artefact absent; run `python train.py` in project 04 first.",
)
def test_app_renders_without_exception():
    """The Streamlit page executes top to bottom without raising."""
    from streamlit.testing.v1 import AppTest

    project_dir = REPO_ROOT / "04-movie-recommendation-system"
    app = AppTest.from_file(str(project_dir / "app.py"), default_timeout=240)
    app.run()
    assert not app.exception