"""Weather data acquisition and loading for the Weather Analysis project.

Data source
-----------
Daily observations come from the **Open-Meteo historical archive** (ERA5
reanalysis). The API is free, needs no API key and requires no registration:

    https://archive-api.open-meteo.com/v1/archive

Five daily variables are requested for one city at a time: maximum, minimum and
mean 2 m temperature, total precipitation and maximum wind speed. The response is
cached as a small CSV in this project's ``data/`` directory, so training and the
interface work offline after the first download.

Why one city at a time
----------------------
Weather is strongly location-specific: pooling cities would let the model learn
from a different climate than the one it is asked to forecast. Caching per city
mirrors the per-ticker design of project 01 and keeps the leakage analysis
simple, because a single city forms one chronological series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shared.datasets import fetch_json
from shared.errors import DataDownloadError, DataNotFoundError, InvalidInputError
from shared.paths import data_dir

PROJECT_SLUG = "09-weather-analysis-prediction"

#: Open-Meteo historical archive endpoint (ERA5 reanalysis, no API key).
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

#: Daily variables requested from the archive, in the order they are stored.
DAILY_VARIABLES: tuple[str, ...] = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
)

#: Mapping from the API variable names to the column names used in this project.
COLUMN_NAMES: dict[str, str] = {
    "time": "date",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "precipitation",
    "wind_speed_10m_max": "wind_max",
}

#: Data columns produced by :func:`load_weather`, after the date.
DATA_COLUMNS: list[str] = [COLUMN_NAMES[name] for name in DAILY_VARIABLES]

#: Default analysis window; long enough to cover several full annual cycles.
DEFAULT_START = "2000-01-01"
DEFAULT_END = "2023-12-31"


@dataclass(frozen=True)
class City:
    """A location the archive can be queried for."""

    name: str
    country: str
    latitude: float
    longitude: float


#: The cities offered in the interface, keyed by the label shown to the user.
#: A small, fixed list keeps the download fast and the comparison meaningful.
CITY_CATALOGUE: dict[str, City] = {
    "London, United Kingdom": City("London", "United Kingdom", 51.5074, -0.1278),
    "New York, United States": City("New York", "United States", 40.7128, -74.0060),
    "Tokyo, Japan": City("Tokyo", "Japan", 35.6762, 139.6503),
    "Sydney, Australia": City("Sydney", "Australia", -33.8688, 151.2093),
    "Mumbai, India": City("Mumbai", "India", 19.0760, 72.8777),
    "Cairo, Egypt": City("Cairo", "Egypt", 30.0444, 31.2357),
}

#: City selected when the interface first loads.
DEFAULT_CITY = "London, United Kingdom"


def get_data_directory() -> Path:
    """Return (and create) this project's ``data/`` directory."""
    return data_dir(PROJECT_SLUG)


def _resolve_city(city_label: str) -> City:
    """Return the :class:`City` for a catalogue label.

    Raises:
        InvalidInputError: When the label is not in :data:`CITY_CATALOGUE`.
    """
    try:
        return CITY_CATALOGUE[city_label]
    except KeyError:
        raise InvalidInputError(
            f"Unknown city '{city_label}'.",
            hint=f"Choose one of: {', '.join(CITY_CATALOGUE)}.",
        ) from None


def _slug(city_label: str) -> str:
    """Return a filesystem-safe slug for a city label."""
    return _resolve_city(city_label).name.lower().replace(" ", "_")


def cache_path(city_label: str) -> Path:
    """Return the CSV cache path for a city."""
    return get_data_directory() / f"weather_{_slug(city_label)}.csv"


def _build_request_url(city: City, start: str, end: str) -> str:
    """Return the full archive URL for one city and date range."""
    variables = ",".join(DAILY_VARIABLES)
    return (
        f"{OPEN_METEO_URL}?latitude={city.latitude}&longitude={city.longitude}"
        f"&start_date={start}&end_date={end}&daily={variables}&timezone=UTC"
    )


def download_city(
    city_label: str,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force: bool = False,
) -> Path:
    """Download one city's daily history and cache it as CSV.

    The write is atomic (temporary file plus rename), so an interrupted download
    can never leave a half-written CSV behind for training to read.

    Args:
        city_label: A key from :data:`CITY_CATALOGUE`.
        start: Inclusive ISO start date.
        end: Inclusive ISO end date.
        force: Re-download even when a cached copy already exists.

    Returns:
        The path to the cached CSV.

    Raises:
        InvalidInputError: When the city label is unknown.
        DataDownloadError: When the archive response is missing expected fields.
    """
    path = cache_path(city_label)
    if path.exists() and not force:
        return path

    city = _resolve_city(city_label)
    url = _build_request_url(city, start, end)
    payload = fetch_json(url, description=f"Open-Meteo archive for {city.name}")

    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not isinstance(daily, dict) or "time" not in daily:
        raise DataDownloadError(
            f"The archive response for {city.name} had no daily data.",
            hint="The coordinates or date range may be outside the archive's coverage.",
        )

    missing = [name for name in DAILY_VARIABLES if name not in daily]
    if missing:
        raise DataDownloadError(
            f"The archive response for {city.name} is missing: {', '.join(missing)}.",
            hint="The upstream variable names may have changed; see the README.",
        )

    frame = pd.DataFrame(
        {
            COLUMN_NAMES["time"]: daily["time"],
            **{COLUMN_NAMES[name]: daily[name] for name in DAILY_VARIABLES},
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    temporary = path.with_name(f".{path.name}.part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def load_weather(
    city_label: str = DEFAULT_CITY,
    *,
    start: str | None = None,
    end: str | None = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load one city's daily weather as a clean, chronological frame.

    Args:
        city_label: A key from :data:`CITY_CATALOGUE`.
        start: Optional inclusive ISO start date filter.
        end: Optional inclusive ISO end date filter.
        auto_download: Download the CSV when it is not cached yet.

    Returns:
        DataFrame with columns ``["date", "temp_max", "temp_min", "temp_mean",
        "precipitation", "wind_max"]``, sorted by date.

    Raises:
        DataNotFoundError: When the cache is missing and ``auto_download`` is off.
        DataDownloadError: When the cached file is missing required columns.
        InvalidInputError: When the filters leave no rows.
    """
    path = cache_path(city_label)
    if not path.exists():
        if not auto_download:
            raise DataNotFoundError(
                f"Weather data for '{city_label}' is not cached.",
                hint="Call download_city() first, or enable auto_download.",
            )
        path = download_city(city_label)

    frame = pd.read_csv(path)
    required = ["date", *DATA_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataDownloadError(
            f"Cached weather file is missing column(s): {', '.join(missing)}.",
            hint="Delete the CSV and re-download it; it may be truncated.",
        )

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in DATA_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", "temp_mean"])
    frame = frame.drop_duplicates(subset=["date"]).sort_values("date")

    if start is not None:
        frame = frame[frame["date"] >= _parse_date(start, "start")]
    if end is not None:
        frame = frame[frame["date"] <= _parse_date(end, "end")]

    if frame.empty:
        raise InvalidInputError(
            "No weather rows match the selected city and date range.",
            hint="Widen the date range or choose a different city.",
        )

    return frame.reset_index(drop=True)


def _parse_date(value: str, label: str) -> pd.Timestamp:
    """Parse a date string, raising a clear error when it is invalid."""
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as error:
        raise InvalidInputError(
            f"Invalid {label} date: '{value}'.",
            hint="Use the ISO format YYYY-MM-DD, for example 2020-06-01.",
        ) from error


def dataset_summary(city_label: str = DEFAULT_CITY) -> dict[str, object]:
    """Return a compact description of a city's data for the sidebar."""
    frame = load_weather(city_label)
    return {
        "city": city_label,
        "rows": int(len(frame)),
        "start": frame["date"].min().strftime("%Y-%m-%d"),
        "end": frame["date"].max().strftime("%Y-%m-%d"),
        "mean_temp": float(frame["temp_mean"].mean()),
        "source": OPEN_METEO_URL,
    }


if __name__ == "__main__":  # pragma: no cover - manual helper
    from shared.config import configure_logging

    configure_logging()
    cached = download_city(DEFAULT_CITY)
    print(f"Cached weather data: {cached}")
    info = dataset_summary()
    print(
        f"Loaded {info['rows']} days for {info['city']} "
        f"({info['start']} to {info['end']})."
    )

