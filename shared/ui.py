"""Streamlit-specific rendering helpers shared by all ten applications.

Keeping these in one place means every project presents results, errors and
model metadata identically, and that app.py files stay small and readable.
"""

from __future__ import annotations

import io
from typing import Any, Callable, Sequence

from shared.errors import friendly_error

# --------------------------------------------------------------------------- #
# Error presentation
# --------------------------------------------------------------------------- #


def show_error(error: BaseException) -> None:
    """Render an exception as a curated message instead of a raw traceback.

    Known :class:`~shared.errors.ProjectError` subclasses keep their wording and
    hint. Anything else is shown with its type and message so the app stays
    usable, while the full traceback is written to the server log only.
    """
    import logging

    import streamlit as st

    title, message = friendly_error(error)
    st.error(f"**{title}**\n\n{message}")
    logging.getLogger("app").exception("Unhandled error surfaced to the UI")


def guard(render: Callable[[], None]) -> None:
    """Run ``render`` and convert any exception into a friendly UI message."""
    try:
        render()
    except Exception as error:  # noqa: BLE001 - the UI boundary must not crash
        show_error(error)


# --------------------------------------------------------------------------- #
# Charts and tables
# --------------------------------------------------------------------------- #


def show_figure(figure: Any, *, caption: str | None = None) -> None:
    """Render a matplotlib or Plotly figure, closing matplotlib handles after use."""
    import streamlit as st

    try:
        import matplotlib.figure

        is_matplotlib = isinstance(figure, matplotlib.figure.Figure)
    except ImportError:  # pragma: no cover
        is_matplotlib = False

    if is_matplotlib:
        st.pyplot(figure, width="stretch")
        import matplotlib.pyplot as plt

        plt.close(figure)
    else:
        st.plotly_chart(figure, width="stretch")

    if caption:
        st.caption(caption)


def show_dataframe(
    frame: Any,
    *,
    caption: str | None = None,
    height: int | None = None,
    hide_index: bool = True,
) -> None:
    """Render a DataFrame with consistent sizing."""
    import streamlit as st

    if frame is None or len(frame) == 0:
        st.info("No rows to display.")
        return
    # Streamlit rejects an explicit None height, so the argument is omitted
    # entirely when the caller did not request a fixed height.
    optional_height = {"height": height} if height is not None else {}
    st.dataframe(
        frame,
        width="stretch",
        hide_index=hide_index,
        **optional_height,
    )
    if caption:
        st.caption(caption)


def show_metrics_dict(
    metrics: dict[str, Any],
    *,
    keys: Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    digits: int = 4,
) -> None:
    """Render selected scalar entries of a metrics dict as metric cards."""
    from shared.metrics import format_metric
    from shared.theme import metric_row

    labels = labels or {}
    selected = list(keys) if keys is not None else [
        key for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]

    cards: list[tuple[str, str, str]] = []
    for key in selected:
        if key not in metrics or not isinstance(metrics[key], (int, float)):
            continue
        cards.append(
            (
                labels.get(key, key.replace("_", " ").upper()),
                format_metric(float(metrics[key]), digits),
                "",
            )
        )
    metric_row(cards)


# --------------------------------------------------------------------------- #
# Sidebar metadata
# --------------------------------------------------------------------------- #


def model_info_panel(
    *,
    algorithm: str,
    trained_on: str,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, str] | None = None,
) -> None:
    """Render a compact 'Model information' block in the sidebar."""
    import streamlit as st

    with st.sidebar:
        st.markdown("#### Model information")
        st.markdown(f"**Algorithm**  \n{algorithm}")
        st.markdown(f"**Training data**  \n{trained_on}")
        if extra:
            for key, value in extra.items():
                st.markdown(f"**{key}**  \n{value}")
        if metrics:
            scalar = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if scalar:
                st.markdown("---")
                st.markdown("#### Held-out metrics")
                from shared.metrics import format_metric

                for key, value in scalar.items():
                    st.markdown(f"- **{key.upper()}**: {format_metric(float(value))}")


def download_text_button(
    text: str,
    *,
    filename: str,
    label: str = "Download",
    key: str | None = None,
    mime: str = "text/plain",
) -> None:
    """Offer a text result as a file download."""
    import streamlit as st

    st.download_button(
        label=label,
        data=text.encode("utf-8"),
        file_name=filename,
        mime=mime,
        key=key,
        width="content",
    )


def download_bytes_button(
    data: bytes,
    *,
    filename: str,
    label: str = "Download",
    mime: str = "application/octet-stream",
    key: str | None = None,
) -> None:
    """Offer a binary artefact (image, csv) as a file download."""
    import streamlit as st

    st.download_button(label=label, data=data, file_name=filename, mime=mime, key=key, width="content")


def dataframe_to_csv_bytes(frame: Any) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes for a download button."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def section(title: str, description: str | None = None) -> None:
    """Render a section heading with an optional explanatory line."""
    import streamlit as st

    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def disclaimer(text: str) -> None:
    """Render a prominent limitation notice; used by the forecasting projects."""
    import streamlit as st

    from shared.theme import note

    note(f"<strong>Limitation.</strong> {text}")