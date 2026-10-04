"""Leakage-free feature engineering for next-day close prediction.

Every feature for day *t* is computed from information available at the end of
day *t*; the target is the close of day *t + horizon*. The shifted target is the
only forward-looking column, and it is dropped before model fitting, so no
future observation can influence a training row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shared.errors import InvalidInputError

#: Feature columns produced by :func:`build_features`, in a fixed order.
#: The order is persisted with the model and re-checked at inference time.
FEATURE_COLUMNS: list[str] = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_10",
    "return_1",
    "return_5",
    "volatility_5",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_12",
    "ema_26",
    "macd",
    "rsi_14",
    "price_vs_sma_10",
    "price_vs_sma_20",
    "high_20",
    "low_20",
    "position_in_range_20",
    "day_of_week",
    "month",
]

#: Name of the supervised target column.
TARGET_COLUMN = "target"

#: Supported target definitions.
#:
#: ``delta`` predicts the *change* in close over the horizon and reconstructs the
#: price as ``close + delta``. This is the default because it is the only target
#: that works for every estimator: tree ensembles cannot extrapolate beyond the
#: range of values seen in training, and a trending price series leaves most test
#: prices outside that range. An earlier version predicted the raw level and the
#: tree models scored R2 of -5 or worse purely from that limitation.
#:
#: ``level`` predicts the raw close. It is retained so the experiment can be
#: reproduced and documented, but it is not the default.
TARGET_MODES: tuple[str, ...] = ("delta", "level")
DEFAULT_TARGET_MODE = "delta"

#: Default simple and exponential moving-average windows.
SMA_WINDOWS: tuple[int, ...] = (5, 10, 20)
EMA_WINDOWS: tuple[int, ...] = (12, 26)

#: Default RSI look-back period.
RSI_WINDOW = 14

#: Look-back used for the 20-day high/low channel.
CHANNEL_WINDOW = 20

#: Minimum usable rows per ticker before a model is worth training.
MIN_TRAINING_ROWS = 120

#: Number of warm-up rows consumed by the longest look-back (channel window).
WARMUP_ROWS = CHANNEL_WINDOW


@dataclass
class FeatureConfig:
    """Configuration for feature construction.

    Attributes:
        sma_windows: Simple moving-average windows.
        ema_windows: Exponential moving-average windows.
        rsi_window: Look-back period for the RSI oscillator.
        channel_window: Look-back for the rolling high/low channel.
        lags: Lag offsets for raw price levels.
        horizon: Number of trading days ahead to predict (1 = next trading day).
        target_mode: Either ``"delta"`` (default) or ``"level"``; see
            :data:`TARGET_MODES` for why ``delta`` is the default.
        feature_columns: Feature order used as model input; must stay stable
            between training and inference.
    """

    sma_windows: tuple[int, ...] = SMA_WINDOWS
    ema_windows: tuple[int, ...] = EMA_WINDOWS
    rsi_window: int = RSI_WINDOW
    channel_window: int = CHANNEL_WINDOW
    lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    horizon: int = 1
    target_mode: str = DEFAULT_TARGET_MODE
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def __post_init__(self) -> None:
        if self.target_mode not in TARGET_MODES:
            raise InvalidInputError(
                f"Unknown target mode '{self.target_mode}'.",
                hint=f"Choose one of: {', '.join(TARGET_MODES)}.",
            )
        if self.horizon < 1:
            raise InvalidInputError(
                f"horizon must be at least 1, got {self.horizon}.",
                hint="Use 1 for a next-trading-day forecast.",
            )


def add_moving_averages(
    frame: pd.DataFrame,
    *,
    sma_windows: tuple[int, ...] = SMA_WINDOWS,
    ema_windows: tuple[int, ...] = EMA_WINDOWS,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``sma_<n>`` and ``ema_<n>`` columns added."""
    result = frame.copy()
    close = result["close"].astype(float)
    for window in sma_windows:
        result[f"sma_{window}"] = close.rolling(window=window, min_periods=window).mean()
    for window in ema_windows:
        result[f"ema_{window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()
    return result


def compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Return Wilder's Relative Strength Index for ``close``.

    Wilder's smoothing is an exponential moving average with ``alpha = 1/window``,
    which is the standard definition used by charting platforms.

    Args:
        close: Series of closing prices in chronological order.
        window: Look-back period; 14 is the conventional default.

    Returns:
        Series of RSI values in the range 0-100, aligned to ``close``'s index.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    # A window with no losses means the RSI is conventionally pinned at 100.
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    # No gains and no losses carries no directional information.
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return rsi


def engineer_ticker(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Build the engineered table for a single ticker, sorted chronologically."""
    result = frame.sort_values("date").reset_index(drop=True).copy()
    close = result["close"].astype(float)

    # Lagged price levels: the model sees recent history, never the future.
    for lag in config.lags:
        result[f"lag_{lag}"] = close.shift(lag)

    # Returns and short-horizon volatility.
    result["return_1"] = close.pct_change(1)
    result["return_5"] = close.pct_change(5)
    result["volatility_5"] = result["return_1"].rolling(5, min_periods=5).std()

    # Trend indicators.
    result = add_moving_averages(result, sma_windows=config.sma_windows, ema_windows=config.ema_windows)
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["rsi_14"] = compute_rsi(close, config.rsi_window)

    # Where price sits relative to its own trend.
    result["price_vs_sma_10"] = close / result["sma_10"] - 1.0
    result["price_vs_sma_20"] = close / result["sma_20"] - 1.0

    # Rolling price channel and the position inside it. This is a distinct
    # signal from the moving averages: it says where price sits between its
    # recent high and low, not merely above or below an average.
    window = config.channel_window
    result["high_20"] = close.rolling(window=window, min_periods=window).max()
    result["low_20"] = close.rolling(window=window, min_periods=window).min()
    span = (result["high_20"] - result["low_20"]).replace(0.0, np.nan)
    result["position_in_range_20"] = ((close - result["low_20"]) / span).fillna(0.5)

    # Calendar effects.
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month

    # Supervised target.
    future_close = close.shift(-config.horizon)
    if config.target_mode == "level":
        result[TARGET_COLUMN] = future_close
    else:
        # The change in price over the horizon, in the same units as the price.
        # Reconstructing the price is then `close + delta`.
        result[TARGET_COLUMN] = future_close - close
    return result


class FeatureFrame:
    """Container pairing an engineered table with its feature/target layout."""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = TARGET_COLUMN,
        target_mode: str = DEFAULT_TARGET_MODE,
    ) -> None:
        self.data = data
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.data)

    @property
    def empty(self) -> bool:
        """Whether the frame holds no rows."""
        return self.data.empty

    def matrix(self) -> np.ndarray:
        """Return the ``(n_samples, n_features)`` float feature matrix."""
        return self.data[self.feature_columns].to_numpy(dtype=float)

    def target(self) -> np.ndarray:
        """Return the ``(n_samples,)`` float target vector."""
        return self.data[self.target_column].to_numpy(dtype=float)

    def dates(self) -> np.ndarray:
        """Return the ``(n_samples,)`` array of dates."""
        return self.data["date"].to_numpy()

    def for_ticker(self, ticker: str) -> "FeatureFrame":
        """Return a new :class:`FeatureFrame` restricted to one ticker."""
        subset = self.data[self.data["ticker"] == ticker].reset_index(drop=True)
        return FeatureFrame(subset, self.feature_columns, self.target_column, self.target_mode)

    def reconstruct_price(self, base_close: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Convert a model prediction into a price in the same units as ``base_close``.

        For the ``delta`` target mode the prediction is a price *change*, so it is
        added to the close at the base date. For ``level`` the prediction is
        already a price and is returned unchanged.
        """
        if self.target_mode == "level":
            return np.asarray(predicted, dtype=float)
        return np.asarray(base_close, dtype=float) + np.asarray(predicted, dtype=float)

    def to_price(self, base_close: float, predicted: float) -> float:
        """Scalar counterpart of :meth:`reconstruct_price`."""
        if self.target_mode == "level":
            return float(predicted)
        return float(base_close) + float(predicted)

    def tickers(self) -> list[str]:
        """Return the distinct tickers present, sorted."""
        return sorted(self.data["ticker"].unique().tolist())


def build_features(
    prices: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
) -> FeatureFrame:
    """Build the supervised modelling table for every ticker in ``prices``.

    Args:
        prices: Long-format frame from :func:`~src.data.load_price_history`.
            Tickers are processed independently so that price levels never mix.
        config: Feature configuration; defaults to :class:`FeatureConfig`.

    Returns:
        A :class:`FeatureFrame` whose ``data`` holds ``date``, ``ticker``,
        ``close``, the engineered features and the shifted ``target``. Warm-up
        rows and the final ``horizon`` rows are retained in ``data`` but marked
        with ``NaN`` features so the UI can still plot the most recent close.

    Raises:
        InvalidInputError: When required columns are missing or no usable rows
            remain.
    """
    config = config or FeatureConfig()

    if prices is None or prices.empty:
        raise InvalidInputError(
            "No price data supplied.",
            hint="Select at least one ticker and a date range before training.",
        )

    missing = [column for column in ("date", "ticker", "close") if column not in prices.columns]
    if missing:
        raise InvalidInputError(
            f"Price data is missing column(s): {', '.join(missing)}.",
            hint="Reload the dataset; it may have been modified.",
        )

    engineered = [engineer_ticker(group, config) for _, group in prices.groupby("ticker", sort=False)]
    combined = pd.concat(engineered, ignore_index=True)

    missing_features = [column for column in config.feature_columns if column not in combined.columns]
    if missing_features:
        raise InvalidInputError(
            f"Feature construction did not produce: {', '.join(missing_features)}.",
            hint="This indicates a bug in feature engineering.",
        )

    if combined.empty:
        raise InvalidInputError(
            "Not enough history to build any rows.",
            hint=f"At least {MIN_TRAINING_ROWS} rows per ticker are needed.",
        )

    return FeatureFrame(
        combined,
        feature_columns=list(config.feature_columns),
        target_mode=config.target_mode,
    )


def ready_rows(frame: FeatureFrame) -> pd.DataFrame:
    """Return only the rows that are complete enough to train or predict on.

    A row is usable when every feature is present and the forward-looking target
    exists. This is what training uses; the UI uses :meth:`FeatureFrame.data`
    when it needs to plot the most recent close.
    """
    columns = [*frame.feature_columns, frame.target_column]
    usable = frame.data.dropna(subset=columns).reset_index(drop=True)
    return usable


def make_feature_frame(prices: pd.DataFrame, *, config: FeatureConfig | None = None) -> FeatureFrame:
    """Convenience wrapper around :func:`build_features`."""
    return build_features(prices, config=config)


def summarise_features(frame: FeatureFrame) -> dict[str, object]:
    """Return a short description of an engineered frame for logs and the UI."""
    usable = ready_rows(frame)
    return {
        "total_rows": int(len(frame.data)),
        "usable_rows": int(len(usable)),
        "features": len(frame.feature_columns),
        "tickers": frame.tickers(),
        "start": frame.data["date"].min().strftime("%Y-%m-%d") if len(frame.data) else "n/a",
        "end": frame.data["date"].max().strftime("%Y-%m-%d") if len(frame.data) else "n/a",
    }