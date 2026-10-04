"""Streamlit interface for the Weather Analysis & Prediction app.

Run from this project's directory::

    streamlit run app.py

Explore a city's historical weather and forecast tomorrow's mean temperature
with a model trained on the archive. The trained artefact is produced by
``train.py`` and loaded here without retraining on page load.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import streamlit as st  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.plotting import interactive_bar, interactive_line, interactive_scatter  # noqa: E402
from shared.theme import (  # noqa: E402
    PALETTE,
    inject_theme,
    metric_row,
    note,
    page_header,
    require_trained_model,
    result_card,
)
from shared.ui import (  # noqa: E402
    dataframe_to_csv_bytes,
    download_bytes_button,
    model_info_panel,
    section,
    show_dataframe,
    show_error,
    show_figure,
)

from src.data import (  # noqa: E402
    CITY_CATALOGUE,
    DEFAULT_CITY,
    DEFAULT_END,
    DEFAULT_START,
    download_city,
    load_weather,
)
from src.features import FeatureConfig, make_feature_frame, summarise_features  # noqa: E402
from src.model import (  # noqa: E402
    MODEL_FILENAME,
    load_bundle,
    model_directory,
    predict_next,
    predict_series,
)

DISCLAIMER = (
    "This is an educational next-day temperature forecast, not a meteorological "
    "product. It is trained on a reanalysis archive and must not be used for "
    "safety-critical decisions."
)


@st.cache_data(show_spinner="Downloading archive data...")
def _features_for(city: str, start: str, end: str, horizon: int):
    """Load and engineer one city's data, cached across reruns."""
    frame = load_weather(city, start=start, end=end)
    config = FeatureConfig(horizon=horizon)
    return frame, make_feature_frame(frame, config=config)


@st.cache_resource(show_spinner=False)
def _load_saved_bundle():
    """Load the trained bundle once per server process."""
    return load_bundle()


def render_sidebar() -> dict:
    """Render the configuration controls and return the selected options."""
    with st.sidebar:
        st.markdown("### Configuration")
        city = st.selectbox(
            "City",
            options=list(CITY_CATALOGUE),
            index=list(CITY_CATALOGUE).index(DEFAULT_CITY),
            key="weather_city",
        )
        st.caption("Source: Open-Meteo historical archive (ERA5 reanalysis).")

        start = st.date_input(
            "From",
            value=date.fromisoformat(DEFAULT_START),
            min_value=date(1940, 1, 1),
            max_value=date(2025, 12, 31),
            key="weather_start",
        )
        end = st.date_input(
            "To",
            value=date.fromisoformat(DEFAULT_END),
            min_value=date(1940, 1, 1),
            max_value=date(2025, 12, 31),
            key="weather_end",
        )
        horizon = st.slider("Forecast horizon (days ahead)", 1, 7, 1, key="weather_horizon")
        refresh = st.checkbox("Force re-download", value=False, key="weather_force")

    return {
        "city": city,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "horizon": horizon,
        "refresh": refresh,
    }


def _temp(value: float) -> str:
    """Format a temperature in degrees Celsius."""
    return f"{value:.1f} \u00b0C"


def render_forecast(bundle, feature_frame) -> None:
    """Render the headline next-day forecast and its headline metrics."""
    section("Next-day forecast", "Based on the most recent day with a complete feature vector.")
    try:
        forecast = predict_next(bundle, feature_frame)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return

    direction = "warmer" if forecast["change"] >= 0 else "cooler"
    result_card(
        f"Predicted mean temperature for {forecast['date']}",
        _temp(forecast["predicted"]),
        accent=PALETTE["accent"],
    )
    metric_row(
        [
            ("Today", _temp(forecast["current"]), f"observed {forecast['based_on']}"),
            ("Change", f"{forecast['change']:+.1f} \u00b0C", direction),
            ("Model", str(forecast["model"]), f"horizon {bundle.horizon} day(s)"),
            (
                "Test RMSE",
                f"{format_metric(bundle.metrics['rmse'])} \u00b0C",
                f"baseline {format_metric(bundle.baseline_metrics['rmse'])} \u00b0C",
            ),
        ]
    )
    if not bundle.beats_baseline:
        st.warning(
            "The model does **not** beat the persistence baseline on this city. "
            "Report this honestly rather than presenting the model as skillful."
        )


def render_charts(bundle, feature_frame, city: str) -> None:
    """Render the history, the parity plot and the feature importances."""
    section("History and evaluation", "Predicted versus observed mean temperature.")
    series = predict_series(bundle, feature_frame)
    if series.empty:
        st.info("Not enough complete rows to plot the evaluation.")
        return

    recent = series.tail(365)
    show_figure(
        interactive_line(
            {
                "Actual": recent["actual"].tolist(),
                "Predicted": recent["predicted"].tolist(),
                "Persistence baseline": recent["baseline"].tolist(),
            },
            x=recent["date"].dt.strftime("%Y-%m-%d").tolist(),
            title=f"{city}: last {len(recent)} days (expected next-day temperature)",
            xlabel="Date",
            ylabel="Temperature (\u00b0C)",
        ),
        caption="Predicted values are for the following day, so the series is shifted by the horizon.",
    )

    left, right = st.columns(2)
    with left:
        show_figure(
            interactive_scatter(
                series["actual"].tolist(),
                series["predicted"].tolist(),
                title="Predicted vs actual",
                xlabel="Actual (\u00b0C)",
                ylabel="Predicted (\u00b0C)",
            )
        )
    with right:
        importances = bundle.feature_importances
        if importances:
            items = sorted(importances.items(), key=lambda pair: pair[1], reverse=True)[:12]
            show_figure(
                interactive_bar(
                    [name for name, _ in items],
                    [value for _, value in items],
                    title="Feature importance",
                    xlabel="Feature",
                    ylabel="Importance",
                )
            )
        else:
            st.caption("This model does not expose feature importances.")

    with st.expander("Download the evaluation table"):
        show_dataframe(series.tail(120).rename(columns={"date": "Date"}))
        download_bytes_button(
            dataframe_to_csv_bytes(series),
            filename=f"weather_evaluation_{city.split(',')[0].lower()}.csv",
            label="Download CSV",
            mime="text/csv",
        )



def render_about() -> None:
    """Explain how the forecast is produced and how it is evaluated."""
    with st.expander("How this works"):
        st.markdown(
            "1. Daily observations come from the Open-Meteo historical archive "
            "(ERA5 reanalysis) for the selected city.\n"
            "2. Features for each day are lagged temperatures, trailing rolling "
            "statistics of the mean temperature, and calendar terms.\n"
            "3. The target is the mean temperature the chosen number of days ahead, "
            "predicted as a *change* from today so tree ensembles generalise.\n"
            "4. Linear Regression, Random Forest and Gradient Boosting are compared "
            "on one chronological split and judged against a persistence baseline "
            "(`tomorrow = today`).\n\n"
            "Metrics are reported in degrees Celsius on the held-out final block of "
            "the series; the split never shuffles, so no future day leaks into training."
        )


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Weather Analysis & Prediction", page_icon="09", layout="wide")
    inject_theme()

    page_header(
        "Weather Data Analysis & Prediction",
        "Explore a city's historical weather and forecast the next-day mean "
        "temperature with a model trained on the Open-Meteo reanalysis archive.",
        eyebrow="Project 09",
        chips=["Regression", "Time series", "scikit-learn", "Plotly"],
    )

    options = render_sidebar()

    model_path = model_directory() / MODEL_FILENAME
    if not require_trained_model(model_path, "python train.py", label="weather model"):
        return

    if options["refresh"]:
        try:
            download_city(options["city"], force=True)
            _features_for.clear()
            st.success(f"Re-downloaded the archive for {options['city']}.")
        except Exception as error:  # noqa: BLE001 - UI boundary
            show_error(error)
            return

    try:
        _frame, feature_frame = _features_for(
            options["city"], options["start"], options["end"], options["horizon"]
        )
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return

    summary = summarise_features(feature_frame)
    bundle = _load_saved_bundle()

    if bundle.city != options["city"]:
        note(
            f"The saved model was trained for <strong>{bundle.city}</strong>, but you are "
            f"viewing <strong>{options['city']}</strong>. Re-run "
            f"<code>python train.py --city \"{options['city']}\"</code> to forecast this city.",
            accent=PALETTE["warning"],
        )

    render_forecast(bundle, feature_frame)
    render_charts(bundle, feature_frame, options["city"])
    render_about()

    model_info_panel(
        algorithm=bundle.algorithm,
        trained_on=f"Open-Meteo archive for {bundle.city}",
        metrics=bundle.metrics,
        extra={
            "Target": f"{bundle.target_mode} change over {bundle.horizon} day(s)",
            "Features": str(len(bundle.feature_columns)),
            "Usable days": f"{summary['usable_rows']:,}",
            "Source": "Open-Meteo: ERA5 archive",
        },
    )

    note(DISCLAIMER, accent=PALETTE["accent"])


if __name__ == "__main__":
    main()

