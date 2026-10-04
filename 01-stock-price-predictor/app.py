"""Streamlit interface for the Stock Price Predictor.

Run from this project's directory:

    streamlit run app.py

The app loads the trained per-ticker model bundle and never retraining on page
load. When the artefact is missing it explains how to create it and offers a
one-click training action instead of failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.errors import InvalidInputError, ModelNotFoundError  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.paths import models_dir, portable_display  # noqa: E402
from shared.plotting import (  # noqa: E402
    feature_importance_bar,
    histogram,
    interactive_line,
    interactive_scatter,
    line_chart,
    residual_plot,
)
from shared.theme import PALETTE, inject_theme, metric_row, note, page_header, result_card  # noqa: E402
from shared.ui import (  # noqa: E402
    disclaimer,
    download_bytes_button,
    dataframe_to_csv_bytes,
    model_info_panel,
    section,
    show_dataframe,
    show_error,
    show_figure,
)

from src.data import (  # noqa: E402
    STOCK_PRICE_SPEC,
    TICKER_NAMES,
    TICKERS,
    dataset_summary,
    load_aapl_ohlcv,
    load_price_history,
)
from src.features import FeatureConfig, make_feature_frame, ready_rows  # noqa: E402
from src.model import (  # noqa: E402
    MODEL_LABELS,
    TRAIN_FRACTION,
    VAL_FRACTION,
    build_prediction_table,
    feature_importance,
    load_bundle,
    predict_next_day,
    save_bundle,
    save_metrics_report,
    build_training_report,
    test_block_predictions,
    train_bundle,
)

MODEL_PATH = models_dir("01-stock-price-predictor") / "stock_model.joblib"
METRICS_PATH = models_dir("01-stock-price-predictor") / "training_metrics.json"

DISCLAIMER = (
    "This is an educational regression model fitted to historical price data. "
    "Short-horizon prices are strongly autocorrelated, so the model mostly learns that "
    "tomorrow's price resembles today's; it beats a persistence baseline only modestly. "
    "It does <strong>not</strong> predict markets and is not investment advice."
)


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def _load_prices(tickers: tuple[str, ...], start: str | None, end: str | None) -> pd.DataFrame:
    """Load price history with Streamlit caching keyed on the arguments."""
    return load_price_history(list(tickers), start=start, end=end)


@st.cache_data(show_spinner=False)
def _dataset_summary() -> dict:
    """Return the cached dataset summary."""
    return dataset_summary()


@st.cache_resource(show_spinner=False)
def _load_bundle_cached(path_str: str, mtime: float):
    """Load the model bundle once per artefact version.

    ``mtime`` is part of the cache key so that retraining in another process
    invalidates the cached object automatically.
    """
    return load_bundle(Path(path_str))


def _get_bundle():
    """Return the model bundle, or ``None`` when it has not been trained yet."""
    if not MODEL_PATH.exists():
        return None
    return _load_bundle_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def render_sidebar() -> dict:
    """Render the sidebar controls and return the selected options."""
    with st.sidebar:
        st.markdown("### Configuration")

        tickers = st.multiselect(
            "Tickers",
            options=TICKERS,
            default=["AAPL"],
            format_func=lambda ticker: f"{ticker} - {TICKER_NAMES.get(ticker, ticker)}",
            help="Train and predict one model per ticker.",
        )

        st.markdown("**Date range**")
        default_start, default_end = pd.Timestamp("2010-01-01"), pd.Timestamp("2016-02-29")
        date_range = st.date_input(
            "Historical window",
            value=(default_start.date(), default_end.date()),
            min_value=pd.Timestamp("2007-01-03").date(),
            max_value=pd.Timestamp("2016-03-01").date(),
            help="The bundled dataset covers 2007-01-03 to 2016-03-01.",
        )

        horizon = st.slider(
            "Forecast horizon (trading days)",
            min_value=1,
            max_value=10,
            value=1,
            help="How many trading days ahead the target is shifted.",
        )

        confidence = st.select_slider(
            "Prediction interval",
            options=[0.80, 0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda value: f"{int(value * 100)}%",
            help="Empirical interval from the spread of test-set residuals.",
        )

        st.markdown("---")
        st.markdown("### Actions")
        train_clicked = st.button("Train / retrain models", width="stretch")
        st.caption(
            "Training compares Linear Regression, Random Forest and Gradient Boosting "
            "per ticker and keeps the best on test RMSE."
        )

    start = end = None
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start = str(date_range[0]), 
        start = str(date_range[0])
        end = str(date_range[1])

    return {
        "tickers": tickers or ["AAPL"],
        "start": start,
        "end": end,
        "horizon": horizon,
        "confidence": confidence,
        "train_clicked": train_clicked,
    }


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def render_dataset_overview(prices: pd.DataFrame) -> None:
    """Show a compact summary of the loaded price data."""
    section("Historical data", "Closing prices for the selected tickers and window.")
    metric_row(
        [
            ("Rows", f"{len(prices):,}", ""),
            ("Tickers", str(prices["ticker"].nunique()), ", ".join(sorted(prices["ticker"].unique()))),
            ("From", prices["date"].min().strftime("%Y-%m-%d"), ""),
            ("To", prices["date"].max().strftime("%Y-%m-%d"), ""),
        ]
    )
    show_dataframe(
        prices.tail(8).assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d")),
        caption="Most recent eight rows of the loaded window.",
    )


def render_price_charts(prices: pd.DataFrame) -> None:
    """Render the closing-price and moving-average charts."""
    section("Price and moving averages", "Interactive; hover for exact values.")

    pivot = prices.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    series = {column: pivot[column].dropna().tolist() for column in pivot.columns}
    x_values = pivot.index.strftime("%Y-%m-%d").tolist()
    st.plotly_chart(
        interactive_line(series, title="Closing price", x=x_values, ylabel="Price"),
        width="stretch",
    )

    single = st.selectbox(
        "Ticker for moving-average detail",
        options=sorted(prices["ticker"].unique()),
        key="ma_ticker",
    )
    ticker_prices = prices[prices["ticker"] == single].sort_values("date")
    close = ticker_prices.set_index("date")["close"].astype(float)
    moving = {
        "Close": close.tolist(),
        "SMA 10": close.rolling(10, min_periods=10).mean().tolist(),
        "SMA 20": close.rolling(20, min_periods=20).mean().tolist(),
        "SMA 50": close.rolling(50, min_periods=50).mean().tolist(),
    }
    st.plotly_chart(
        interactive_line(
            moving,
            title=f"{single} - close vs moving averages",
            x=[timestamp.strftime("%Y-%m-%d") for timestamp in close.index],
            ylabel="Price",
        ),
        width="stretch",
    )


def render_prediction(bundle, feature_frame, ticker: str, confidence: float) -> None:
    """Render the next-day forecast for one ticker."""
    section("Next-step forecast", "Model output for the most recent available observation.")

    ticker_model = bundle.get(ticker)
    forecast = predict_next_day(ticker_model, feature_frame, confidence=confidence)

    direction_color = {
        "up": PALETTE["positive"],
        "down": PALETTE["negative"],
        "flat": PALETTE["muted"],
    }[forecast["direction"]]

    left, right = st.columns([1, 1])
    with left:
        result_card(
            f"Predicted close on the next trading day (as of {forecast['as_of_date']})",
            f"{forecast['prediction']:.2f}",
            accent=direction_color,
        )
        st.caption(
            f"{int(forecast['confidence_level'] * 100)}% empirical interval: "
            f"{forecast['lower_bound']:.2f} to {forecast['upper_bound']:.2f}"
        )
    with right:
        metric_row(
            [
                ("Last close", f"{forecast['last_close']:.2f}", ""),
                (
                    "Expected change",
                    f"{forecast['expected_change']:+.2f}",
                    f"{forecast['expected_change_pct']:+.2f}%",
                ),
                ("Model", MODEL_LABELS.get(forecast["model_name"], forecast["model_name"]), ticker),
            ]
        )

    note(
        "The interval is derived from the standard deviation of residuals on the held-out "
        "test block, scaled by a normal quantile. It describes historical model error spread, "
        "not market risk.",
        accent=PALETTE["accent"],
    )


def render_evaluation(bundle, feature_frame, ticker: str) -> None:
    """Render held-out metrics, actual-vs-predicted and residual diagnostics."""
    section(
        "Held-out evaluation",
        "Metrics are computed on the most recent 10% of observations, never used for training.",
    )

    ticker_model = bundle.get(ticker)

    metric_row(
        [
            ("MAE", format_metric(ticker_model.metrics["mae"]), "mean absolute error"),
            ("RMSE", format_metric(ticker_model.metrics["rmse"]), "root mean squared error"),
            ("R2", format_metric(ticker_model.metrics["r2"]), "coefficient of determination"),
            (
                "Baseline RMSE",
                format_metric(ticker_model.baseline_metrics["rmse"]),
                "persistence: tomorrow = today",
            ),
        ]
    )

    if ticker_model.beats_baseline:
        st.success(
            f"The model beats the persistence baseline on RMSE "
            f"({ticker_model.metrics['rmse']:.3f} vs {ticker_model.baseline_metrics['rmse']:.3f})."
        )
    else:
        st.warning(
            "The model does **not** beat the persistence baseline on this ticker. "
            "Report this honestly rather than presenting the model as skilful."
        )

    test_predictions = test_block_predictions(ticker_model, feature_frame)
    if test_predictions.empty:
        st.info("Not enough held-out rows to plot."
                " Widen the date range to produce a larger test block.")
        return

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            interactive_scatter(
                test_predictions["actual"].tolist(),
                test_predictions["predicted"].tolist(),
                title=f"{ticker} - actual vs predicted (test block)",
                xlabel="Actual close",
                ylabel="Predicted close",
            ),
            width="stretch",
        )
    with right:
        show_figure(
            residual_plot(
                test_predictions["predicted"].tolist(),
                test_predictions["residual"].tolist(),
                title=f"{ticker} - residuals vs predicted",
            )
        )

    show_dataframe(
        test_predictions.assign(
            date=lambda frame: pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d"),
            actual=lambda frame: frame["actual"].round(3),
            predicted=lambda frame: frame["predicted"].round(3),
            residual=lambda frame: frame["residual"].round(3),
        ).tail(15),
        caption="Most recent held-out predictions.",
    )
    download_bytes_button(
        dataframe_to_csv_bytes(test_predictions),
        filename=f"{ticker}_test_predictions.csv",
        label="Download test-block predictions (CSV)",
        mime="text/csv",
    )


def render_feature_importance(bundle, ticker: str) -> None:
    """Render the feature importance chart for the selected ticker."""
    section("What the model relies on", "Feature importance for the selected ticker.")
    importance = feature_importance(bundle.get(ticker))
    if importance is None:
        st.info("This estimator does not expose feature importances.")
        return
    show_figure(
        feature_importance_bar(
            importance["feature"].tolist(),
            importance["importance"].tolist(),
            title=f"{ticker} - top features ({importance['kind'].iloc[0]})",
        )
    )


def render_model_comparison(bundle) -> None:
    """Show the per-ticker metrics table across all trained tickers."""
    section("Trained models", "One estimator per ticker, selected on lowest test RMSE.")
    table = bundle.summary_table()
    show_dataframe(
        table.assign(
            mae=lambda frame: frame["mae"].round(4),
            rmse=lambda frame: frame["rmse"].round(4),
            r2=lambda frame: frame["r2"].round(4),
            baseline_rmse=lambda frame: frame["baseline_rmse"].round(4),
        ),
        caption=f"{len(table)} ticker model(s). 'beats_baseline' compares RMSE against persistence.",
    )


def render_volume_analysis() -> None:
    """Render the AAPL OHLCV volume and range analysis from the second dataset."""
    with st.expander("Supplementary analysis: Apple OHLCV volume and daily range (2015-2017)"):
        try:
            ohlcv = load_volume_data()
        except Exception as error:  # noqa: BLE001 - optional panel
            show_error(error)
            return

        ohlcv = ohlcv.copy()
        ohlcv["daily_range"] = ohlcv["high"] - ohlcv["low"]
        left, right = st.columns(2)
        with left:
            show_figure(
                histogram(
                    ohlcv["volume"].tolist(),
                    title="Distribution of daily volume",
                    xlabel="Shares traded",
                    bins=40,
                )
            )
        with right:
            show_figure(
                line_chart(
                    {"Daily high-low range": ohlcv["daily_range"].tolist()},
                    title="Daily trading range over time",
                    xlabel="Trading day index",
                    ylabel="Price range",
                )
            )
        st.caption(
            "This second dataset provides volume, which the primary closing-price dataset "
            "does not include."
        )


@st.cache_data(show_spinner=False)
def load_volume_data() -> pd.DataFrame:
    """Load the cached AAPL OHLCV dataset."""
    return load_aapl_ohlcv()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def run_training(options: dict) -> bool:
    """Train and persist the model bundle. Returns ``True`` on success."""
    config = FeatureConfig(horizon=options["horizon"])
    try:
        with st.spinner("Downloading data if needed and training models..."):
            prices = _load_prices(tuple(options["tickers"]), options["start"], options["end"])
            feature_frame = make_feature_frame(prices, config=config)
            bundle, comparison = train_bundle(feature_frame, config=config)
            save_bundle(bundle)

            dataset_info = {
                "rows": int(len(prices)),
                "tickers": sorted(prices["ticker"].unique().tolist()),
                "start": prices["date"].min().strftime("%Y-%m-%d"),
                "end": prices["date"].max().strftime("%Y-%m-%d"),
            }
            from src.features import summarise_features

            report = build_training_report(
                bundle, comparison, dataset=dataset_info, feature_summary=summarise_features(feature_frame)
            )
            save_metrics_report(report)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return False

    st.success(
        f"Trained {len(bundle)} model(s) for {', '.join(bundle.tickers)}. "
        f"Saved to {portable_display(MODEL_PATH)}."
    )
    st.cache_resource.clear()
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Stock Price Predictor", page_icon="01", layout="wide")
    inject_theme()

    page_header(
        "Stock Price Predictor",
        "Regression on historical prices to forecast the next trading-day close, with an "
        "explicit persistence baseline and chronological validation.",
        eyebrow="Project 01",
        chips=["Regression / Time series", "scikit-learn", "Streamlit", "Chronological split"],
    )

    options = render_sidebar()

    try:
        prices = _load_prices(tuple(options["tickers"]), options["start"], options["end"])
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        st.stop()

    bundle = _get_bundle()

    if options["train_clicked"]:
        if run_training(options):
            bundle = _get_bundle()
            st.rerun()

    if bundle is None:
        st.warning(
            f"**No trained model found.** The app expects `{portable_display(MODEL_PATH)}`.\n\n"
            "Click **Train / retrain models** in the sidebar, or run:\n\n"
            "```bash\npython train.py\n```"
        )
        render_dataset_overview(prices)
        render_price_charts(prices)
        disclaimer(DISCLAIMER)
        return

    available = [ticker for ticker in options["tickers"] if ticker in bundle.tickers]
    if not available:
        st.error(
            f"The trained model covers {', '.join(bundle.tickers)}, which does not include your "
            f"selection ({', '.join(options['tickers'])})."
        )
        st.info("Select a ticker from the trained set, or retrain with your selection.")
        return

    selected_ticker = st.radio(
        "Ticker to inspect",
        options=available,
        horizontal=True,
        key="selected_ticker",
    )

    # Refit features with the configuration recorded at training time so the
    # feature layout always matches the saved model.
    config = FeatureConfig(horizon=bundle.horizon, target_mode=bundle.target_mode)
    try:
        feature_frame = make_feature_frame(prices, config=config)
    except InvalidInputError as error:
        show_error(error)
        st.stop()

    render_dataset_overview(prices)
    render_price_charts(prices)
    render_prediction(bundle, feature_frame, selected_ticker, options["confidence"])
    render_evaluation(bundle, feature_frame, selected_ticker)
    render_feature_importance(bundle, selected_ticker)
    render_model_comparison(bundle)
    render_volume_analysis()

    with st.sidebar:
        ticker_model = bundle.get(selected_ticker)
        model_info_panel(
            algorithm=MODEL_LABELS.get(ticker_model.model_name, ticker_model.model_name),
            trained_on=(
                f"{ticker_model.n_train:,} rows, {ticker_model.train_start} to {ticker_model.train_end}"
            ),
            metrics=ticker_model.metrics,
            extra={
                "Trained at": bundle.trained_at,
                "Forecast horizon": f"{bundle.horizon} trading day(s)",
                "Target definition": (
                    "price change" if bundle.target_mode == "delta" else "price level"
                ),
                "Dataset": f"{STOCK_PRICE_SPEC.name}",
            },
        )

    disclaimer(DISCLAIMER)


if __name__ == "__main__":
    main()