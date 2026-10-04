"""Dataset acquisition and loading for the Stock Price Predictor.

Data source decision (documented for the reviewer)
--------------------------------------------------
The original plan was to fetch live prices from Yahoo Finance via ``yfinance``.
During development that endpoint returned HTTP 429 ("Too Many Requests") for
every request, and Stooq — the usual fallback — now serves a JavaScript
anti-bot challenge instead of CSV data. Both are therefore unusable from a
scripted, reproducible setup.

Instead this project uses two small, stable, publicly hosted CSV datasets that
require no API key and download in under a second:

1. ``stockdata.csv`` (Plotly datasets mirror, originally from Quandl).
   Daily *closing* prices for five instruments between 2007-01-03 and
   2016-03-01: MSFT, IBM, SBUX, AAPL and the S&P 500 index (GSPC).
   Used as the primary multi-ticker dataset.

2. ``finance-charts-apple.csv`` (Plotly datasets mirror).
   Daily OHLCV bars for AAPL between 2015-02-17 and 2017-02-16, including
   volume. Used for the volume and intraday-range analysis, because the primary
   dataset has no volume column.

Both are real market data. Because they are historical snapshots, the project
makes no claim about current prices — see the README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.datasets import DatasetSpec, fetch_dataset
from shared.errors import DataNotFoundError, InvalidInputError
from shared.paths import data_dir

PROJECT_SLUG = "01-stock-price-predictor"

#: Primary multi-ticker closing-price dataset.
STOCK_PRICE_SPEC = DatasetSpec(
    name="Daily closing prices for five instruments (2007-2016)",
    url="https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv",
    filename="stockdata.csv",
    mirrors=[
        "https://cdn.jsdelivr.net/gh/plotly/datasets@master/stockdata.csv",
    ],
    approx_size_mb=0.15,
    notes="Wide format: one column per ticker plus a Date column.",
)

#: Secondary AAPL OHLCV dataset, used for volume and intraday-range analysis.
AAPL_OHLCV_SPEC = DatasetSpec(
    name="Apple Inc. daily OHLCV bars (2015-2017)",
    url="https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
    filename="aapl_ohlcv.csv",
    mirrors=[
        "https://cdn.jsdelivr.net/gh/plotly/datasets@master/finance-charts-apple.csv",
    ],
    approx_size_mb=0.06,
    notes="Includes Open, High, Low, Close, Volume and Adjusted close.",
)

#: Display names for the tickers present in the primary dataset.
TICKER_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "IBM": "IBM Corporation",
    "SBUX": "Starbucks Corporation",
    "GSPC": "S&P 500 Index",
}

#: Column in the primary CSV that holds the date.
DATE_COLUMN = "Date"

#: The available tickers, in the order shown in the interface.
TICKERS: list[str] = list(TICKER_NAMES)


def get_data_directory() -> Path:
    """Return (and create) this project's ``data/`` directory."""
    return data_dir(PROJECT_SLUG)


def download_datasets(*, force: bool = False) -> tuple[Path, Path]:
    """Download both datasets into ``data/`` and return their paths.

    Args:
        force: Re-download even when a verified cached copy exists.
    """
    directory = get_data_directory()
    prices = fetch_dataset(STOCK_PRICE_SPEC, directory, force=force)
    ohlcv = fetch_dataset(AAPL_OHLCV_SPEC, directory, force=force)
    return prices, ohlcv


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a dataset CSV, translating low-level failures into clear errors."""
    if not path.exists():
        raise DataNotFoundError(
            f"Dataset '{path.name}' not found.",
            hint=(
                "Run `python src/data.py` from the project directory, or call "
                "`python -c \"from src.data import download_datasets; download_datasets()\"`, "
                "to download it."
            ),
        )
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
        raise DataNotFoundError(
            f"Dataset '{path.name}' could not be parsed as CSV.",
            hint=f"Delete the file and re-download it. ({type(error).__name__}: {error})",
        ) from error


def load_price_history(
    tickers: list[str] | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load daily closing prices in tidy (long) format.

    Args:
        tickers: Subset of :data:`TICKERS` to keep. Defaults to all of them.
        start: Inclusive start date, ``"YYYY-MM-DD"``.
        end: Inclusive end date, ``"YYYY-MM-DD"``.
        auto_download: Download the CSV when it is not present locally.

    Returns:
        DataFrame with columns ``["date", "ticker", "close"]``, sorted by ticker
        then date, with duplicate dates removed.

    Raises:
        InvalidInputError: For unknown tickers or a malformed date range.
        DataNotFoundError: When the dataset is unavailable.
    """
    directory = get_data_directory()
    path = directory / STOCK_PRICE_SPEC.filename
    if not path.exists() and auto_download:
        path = fetch_dataset(STOCK_PRICE_SPEC, directory)

    frame = _read_csv(path)

    if DATE_COLUMN not in frame.columns:
        raise DataNotFoundError(
            f"Dataset '{path.name}' is missing the expected '{DATE_COLUMN}' column.",
            hint="Delete the CSV and re-download it; the upstream file may have changed.",
        )

    selected = list(tickers) if tickers else list(TICKERS)
    unknown = [ticker for ticker in selected if ticker not in frame.columns]
    if unknown:
        raise InvalidInputError(
            f"Unknown ticker(s): {', '.join(unknown)}.",
            hint=f"Available tickers: {', '.join(TICKERS)}.",
        )
    if not selected:
        raise InvalidInputError("Select at least one ticker.", hint=f"Available: {', '.join(TICKERS)}.")

    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[DATE_COLUMN])

    tidy = (
        frame.melt(
            id_vars=[DATE_COLUMN],
            value_vars=selected,
            var_name="ticker",
            value_name="close",
        )
        .rename(columns={DATE_COLUMN: "date"})
        .dropna(subset=["close"])
    )
    tidy["close"] = pd.to_numeric(tidy["close"], errors="coerce")
    tidy = tidy.dropna(subset=["close"])
    tidy = tidy.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])

    if start is not None:
        tidy = tidy[tidy["date"] >= _parse_date(start, "start")]
    if end is not None:
        tidy = tidy[tidy["date"] <= _parse_date(end, "end")]

    if tidy.empty:
        raise InvalidInputError(
            "No price rows match the selected tickers and date range.",
            hint="Widen the date range or choose different tickers.",
        )

    return tidy.reset_index(drop=True)


def _parse_date(value: str, label: str) -> pd.Timestamp:
    """Parse a date string, raising a clear error when it is invalid."""
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as error:
        raise InvalidInputError(
            f"Invalid {label} date: '{value}'.",
            hint="Use the ISO format YYYY-MM-DD, for example 2014-01-31.",
        ) from error


def load_aapl_ohlcv(*, auto_download: bool = True) -> pd.DataFrame:
    """Load the AAPL OHLCV dataset used for volume analysis.

    Returns:
        DataFrame with columns ``["date", "open", "high", "low", "close",
        "volume", "adjusted_close"]``.
    """
    directory = get_data_directory()
    path = directory / AAPL_OHLCV_SPEC.filename
    if not path.exists() and auto_download:
        path = fetch_dataset(AAPL_OHLCV_SPEC, directory)

    frame = _read_csv(path)
    required = ["Date", "AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataNotFoundError(
            f"OHLCV dataset is missing expected column(s): {', '.join(missing)}.",
            hint="Delete the CSV and re-download it; the upstream file may have changed.",
        )

    renamed = frame.rename(
        columns={
            "Date": "date",
            "AAPL.Open": "open",
            "AAPL.High": "high",
            "AAPL.Low": "low",
            "AAPL.Close": "close",
            "AAPL.Volume": "volume",
            "AAPL.Adjusted": "adjusted_close",
        }
    )
    renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce")
    keep = ["date", "open", "high", "low", "close", "volume"]
    if "adjusted_close" in renamed.columns:
        keep.append("adjusted_close")

    cleaned = renamed[keep].dropna(subset=["date", "close"])
    return cleaned.sort_values("date").reset_index(drop=True)


def dataset_summary() -> dict[str, object]:
    """Return a compact description of the primary dataset for the UI sidebar."""
    frame = load_price_history()
    return {
        "rows": int(len(frame)),
        "tickers": sorted(frame["ticker"].unique().tolist()),
        "start": frame["date"].min().strftime("%Y-%m-%d"),
        "end": frame["date"].max().strftime("%Y-%m-%d"),
        "source": STOCK_PRICE_SPEC.url,
    }


if __name__ == "__main__":  # pragma: no cover - manual helper
    from shared.config import configure_logging

    configure_logging()
    prices_path, ohlcv_path = download_datasets()
    print(f"Prices dataset: {prices_path}")
    print(f"OHLCV dataset:  {ohlcv_path}")
    summary = dataset_summary()
    print(
        f"Loaded {summary['rows']} rows for {len(summary['tickers'])} tickers "
        f"({summary['start']} to {summary['end']})."
    )