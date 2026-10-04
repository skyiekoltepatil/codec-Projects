"""A small, consistent visual system shared by every Streamlit app.

Design direction is deliberately restrained: a neutral slate palette, generous
whitespace, and typography that reads well in a project review. No gradients,
no decorative illustration, no animation.
"""

from __future__ import annotations

import html
from typing import Iterable

#: Palette. Kept small on purpose so every project looks like one product.
PALETTE: dict[str, str] = {
    "ink": "#000000",
    "muted": "#444444",
    "line": "#d0d0d0",
    "surface": "#ffffff",
    "canvas": "#f5f5f5",
    "accent": "#000000",
    "accent_soft": "#eeeeee",
    "positive": "#000000",
    "warning": "#555555",
    "negative": "#000000",
}

#: Chart colours, ordered for categorical series.
CHART_COLORS: list[str] = ["#000000", "#666666", "#999999", "#333333", "#bbbbbb", "#555555"]

#: Order used when a project renders a fixed set of sentiment/risk classes.
RISK_COLORS: dict[str, str] = {
    "LOW": PALETTE["positive"],
    "MEDIUM": PALETTE["warning"],
    "HIGH": PALETTE["negative"],
}


def _css() -> str:
    """Return the global stylesheet injected into every app."""
    p = PALETTE
    return f"""
<style>
  /* Force a single, predictable black-and-white light appearance. The browser's
     OS-level dark mode must never leak in and make text unreadable, so the
     colour scheme, Streamlit theme variables and every container are pinned. */
  :root, html, body, .stApp {{
    color-scheme: light !important;
    --ink: {p['ink']};
    --muted: {p['muted']};
    --line: {p['line']};
    --surface: {p['surface']};
    --canvas: {p['canvas']};
    --accent: {p['accent']};
    --accent-soft: {p['accent_soft']};
    --background-color: {p['surface']};
    --secondary-background-color: {p['canvas']};
    --text-color: {p['ink']};
    --primary-color: {p['accent']};
  }}
  html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }}
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
  [data-testid="stMain"], [data-testid="stMainBlockContainer"], [data-testid="stToolbar"] {{
    background: #ffffff !important; color: #000000 !important;
  }}
  [data-testid="stSidebar"], [data-testid="stSidebar"] > div,
  [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {{
    background: #f5f5f5 !important;
  }}
  .stApp, .stApp label, .stApp p, .stApp span, .stApp small, .stApp li, .stApp strong,
  .stApp div[data-testid="stMarkdownContainer"], .stApp h1, .stApp h2, .stApp h3, .stApp h4,
  [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] *,
  [data-testid="stSidebar"] * {{ color: #000000; }}
  .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div,
  .stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"] {{
    background: #ffffff !important; color: #000000 !important; border-color: #000000 !important;
  }}
  .stApp [data-baseweb="popover"], .stApp [data-baseweb="menu"], .stApp ul[role="listbox"],
  .stApp [role="option"] {{
    background: #ffffff !important; color: #000000 !important;
  }}
  .stApp [role="option"]:hover, .stApp [aria-selected="true"] {{ background: #eeeeee !important; }}
  .stApp button, .stApp [data-testid="stBaseButton-secondary"],
  .stApp [data-testid="stBaseButton-primary"] {{ background: #ffffff; color: #000000; border: 1px solid #000000; }}
  .stApp button:hover {{ background: #000000; color: #ffffff; }}
  .stApp button:hover * {{ color: #ffffff; }}
  .stApp code, .stApp pre {{ background: #f0f0f0 !important; color: #000000 !important; }}
  .stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"] {{
    background: #ffffff !important; color: #000000 !important;
  }}
  .stApp hr, .stApp [data-testid="stDivider"] {{ border-color: {p['line']} !important; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.85rem; font-weight: 650; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.25rem; font-weight: 600; margin-top: 1.6rem; }}
  h3 {{ font-size: 1.0rem; font-weight: 600; }}
  p, li {{ color: var(--ink); }}
  @media (prefers-color-scheme: dark) {{
    :root, html, body {{ color-scheme: light !important; }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
      background: #ffffff !important; color: #000000 !important;
    }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {{
      background: #f5f5f5 !important;
    }}
  }}

  .app-eyebrow {{
    font-size: 0.72rem; font-weight: 650; letter-spacing: 0.10em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.35rem;
  }}
  .app-subtitle {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 1.1rem; max-width: 72ch; }}

  .meta-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.6rem 0 0.2rem 0; }}
  .chip {{
    display: inline-block; padding: 0.16rem 0.55rem; border-radius: 999px;
    border: 1px solid var(--line); background: var(--surface);
    font-size: 0.74rem; color: var(--muted); font-weight: 550; white-space: nowrap;
  }}
  .chip-accent {{ background: var(--accent-soft); border-color: #c7d7fe; color: var(--accent); }}

  .metric-card {{
    border: 1px solid var(--line); background: var(--surface);
    border-radius: 10px; padding: 0.85rem 1rem; height: 100%;
  }}
  .metric-label {{ font-size: 0.8rem; font-size: 0.8rem; font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-value {{ font-size: 1.45rem; font-weight: 650; color: var(--ink); margin-top: 0.15rem; }}
  .metric-note {{ font-size: 0.78rem; color: var(--muted); margin-top: 0.1rem; }}

  .result-card {{
    border: 1px solid var(--line); border-left: 4px solid var(--accent);
    background: var(--surface); border-radius: 10px; padding: 1rem 1.15rem;
  }}
  .result-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
  .result-value {{ font-size: 1.7rem; font-weight: 680; color: var(--ink); line-height: 1.2; }}

  .proj-card {{
    border: 1px solid var(--line); background: var(--surface);
    border-radius: 12px; padding: 1rem 1.1rem 1.1rem 1.1rem; height: 100%;
  }}
  .proj-num {{ font-size: 0.74rem; font-weight: 700; color: var(--accent); letter-spacing: 0.08em; }}
  .proj-title {{ font-size: 1.02rem; font-weight: 640; color: var(--ink); margin: 0.15rem 0 0.3rem 0; }}
  .proj-desc {{ font-size: 0.85rem; color: var(--muted); min-height: 3.2rem; }}

  .note {{ border: 1px solid var(--line); border-left: 4px solid var(--muted);
           background: var(--canvas); border-radius: 8px; padding: 0.7rem 0.9rem;
           font-size: 0.85rem; color: var(--ink); }}

  .chat-user {{ background: var(--accent-soft); border: 1px solid #c7d7fe; border-radius: 10px;
                padding: 0.6rem 0.85rem; margin: 0.35rem 0; }}
  .chat-bot {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
               padding: 0.6rem 0.85rem; margin: 0.35rem 0; }}
  .chat-role {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); }}

  footer, #MainMenu {{ visibility: hidden; }}
  .stButton > button {{ border-radius: 8px; font-weight: 560; }}
</style>
"""


def inject_theme() -> None:
    """Apply the shared stylesheet to the current Streamlit page.

    Safe to call on every rerun; Streamlit replaces the previous ``<style>`` tag.
    """
    import streamlit as st

    st.markdown(_css(), unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str, chips: Iterable[str] = ()) -> None:
    """Render the standard page header: eyebrow, title, subtitle and metadata chips."""
    import streamlit as st

    chip_html = "".join(
        f'<span class="chip">{html.escape(str(chip))}</span>' for chip in chips
    )
    st.markdown(
        f"""
        <div class="app-eyebrow">{html.escape(eyebrow)}</div>
        <h1>{html.escape(title)}</h1>
        <div class="app-subtitle">{html.escape(subtitle)}</div>
        <div class="meta-row">{chip_html}</div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> str:
    """Return HTML for a single metric card (caller decides the layout)."""
    note_html = f'<div class="metric-note">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>{note_html}</div>'
    )


def metric_row(metrics: list[tuple[str, str, str]]) -> None:
    """Render a responsive grid of metric cards.

    Args:
        metrics: list of ``(label, value, note)`` triples.
    """
    import streamlit as st

    if not metrics:
        return
    columns = st.columns(min(len(metrics), 4))
    for index, (label, value, note) in enumerate(metrics):
        with columns[index % len(columns)]:
            st.markdown(metric_card(label, value, note), unsafe_allow_html=True)


def result_card(label: str, value: str, accent: str | None = None) -> None:
    """Render the primary prediction/answer block."""
    import streamlit as st

    border = accent or PALETTE["accent"]
    st.markdown(
        f'<div class="result-card" style="border-left-color:{border};">'
        f'<div class="result-label">{html.escape(label)}</div>'
        f'<div class="result-value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def note(text: str, accent: str | None = None) -> None:
    """Render a low-emphasis informational block."""
    import streamlit as st

    border = accent or PALETTE["muted"]
    st.markdown(
        f'<div class="note" style="border-left-color:{border};">{text}</div>',
        unsafe_allow_html=True,
    )


def project_card_html(
    number: str,
    title: str,
    description: str,
    ml_type: str,
    technology: str,
) -> str:
    """Return HTML for one dashboard project card."""
    return (
        f'<div class="proj-card">'
        f'<div class="proj-num">{html.escape(number)}</div>'
        f'<div class="proj-title">{html.escape(title)}</div>'
        f'<div class="proj-desc">{html.escape(description)}</div>'
        f'<div class="meta-row">'
        f'<span class="chip chip-accent">{html.escape(ml_type)}</span>'
        f'<span class="chip">{html.escape(project_short_tech(technology))}</span>'
        f"</div></div>"
    )


def project_short_tech(technology: str) -> str:
    """Shorten a technology string so it fits a chip comfortably."""
    first = technology.split(",")[0].strip()
    return first if len(first) <= 26 else first[:24].rstrip() + "\u2026"


def require_trained_model(
    path,
    train_command: str,
    *,
    label: str = "model",
) -> bool:
    """Render a consistent 'train me first' panel when an artefact is missing.

    Returns ``True`` when the artefact exists, otherwise renders guidance and
    returns ``False`` so the caller can stop cleanly.
    """
    import streamlit as st

    from shared.paths import portable_display

    from pathlib import Path

    path = Path(path)
    if path.exists():
        return True
    st.error(f"**Trained {label} not found.**")
    st.markdown(
        f"The app expects `{portable_display(path)}`, which is generated by the "
        f"training script rather than committed to Git.\n\n"
        f"```bash\n{train_command}\n```"
    )
    return False