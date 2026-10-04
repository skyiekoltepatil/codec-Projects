"""Root dashboard for the AI/ML Internship Projects suite.

Run from the repository root:

    streamlit run app.py

The dashboard lists all ten projects and starts the selected one as a separate
Streamlit server on its own port. A separate process per project is required
because every project ships a package named ``src``; see ``shared/launcher.py``
for the full explanation of that constraint.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.errors import friendly_error  # noqa: E402
from shared.launcher import get_manager  # noqa: E402
from shared.paths import portable_display  # noqa: E402
from shared.projects import PROJECTS, ProjectInfo  # noqa: E402
from shared.theme import PALETTE, inject_theme, metric_row, note, page_header, project_card_html  # noqa: E402
from shared.ui import show_error  # noqa: E402

SUITE_TITLE = "AI/ML Internship Projects"
SUITE_SUBTITLE = (
    "Ten independently runnable machine-learning applications with real training, "
    "honest evaluation and a consistent interface. Each project can be launched "
    "from here, or run on its own with `streamlit run app.py`."
)


def readiness(project: ProjectInfo) -> tuple[bool, str]:
    """Return ``(ready, description)`` describing a project's training state.

    A project is ready when its trained artefact exists, or when it needs no
    training at all (project 10 uses a pretrained model).
    """
    if not project.trainable:
        return True, "Pretrained model, downloads on first use"

    if not project.model_dir.is_dir():
        return False, "Not trained yet"

    artefacts = [
        path
        for path in project.model_dir.iterdir()
        if path.suffix in {".joblib", ".pt", ".pth"} and not path.name.endswith(".json")
    ]
    if not artefacts:
        return False, "Not trained yet"

    newest = max(artefacts, key=lambda path: path.stat().st_mtime)
    size_mb = newest.stat().st_size / (1024 * 1024)
    return True, f"Trained ({newest.name}, {size_mb:.1f} MB)"


def render_overview() -> None:
    """Render the summary strip at the top of the dashboard."""
    trained = sum(1 for project in PROJECTS if readiness(project)[0])
    total_apps = sum(1 for project in PROJECTS if project.app_file.exists())
    metric_row(
        [
            ("Projects", str(len(PROJECTS)), "independently runnable"),
            ("Apps present", f"{total_apps}/{len(PROJECTS)}", "Streamlit entry points"),
            ("Trained", f"{trained}/{len(PROJECTS)}", "artefacts on disk"),
            ("Python", f"{sys.version_info.major}.{sys.version_info.minor}", "runtime"),
        ]
    )


def render_project_grid() -> None:
    """Render the ten project cards with a launch control each."""
    st.markdown("### Projects")

    manager = get_manager()
    # Two columns keeps the cards readable and matches the review context.
    columns = st.columns(2)

    for index, project in enumerate(PROJECTS):
        with columns[index % 2]:
            st.markdown(
                project_card_html(
                    project.number,
                    project.title,
                    project.short,
                    project.ml_type,
                    project.technology,
                ),
                unsafe_allow_html=True,
            )

            ready, description = readiness(project)
            server = manager.get(project.slug)

            status_column, action_column = st.columns([3, 2])
            with status_column:
                if server is not None:
                    st.caption(f"Running on port {server.port}")
                elif ready:
                    st.caption(description)
                else:
                    st.caption(f"{description} - train it first")

            with action_column:
                label = "Open" if server is not None else "Launch"
                clicked = st.button(
                    label,
                    key=f"launch_{project.slug}",
                    width="stretch",
                    type="primary" if ready else "secondary",
                )

            if clicked:
                if server is None:
                    with st.spinner(f"Starting {project.title}..."):
                        try:
                            server = manager.start(
                                project.slug,
                                project.app_file,
                                project.title,
                                cwd=project.directory,
                            )
                        except Exception as error:  # noqa: BLE001 - UI boundary
                            title, message = friendly_error(error)
                            st.error(f"**{title}**\n\n{message}")
                            server = None
                if server is not None:
                    st.success(f"Running at {server.url}")
                    st.markdown(f"[Open **{project.title}** in a new tab]({server.url})")

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)


def render_running_servers() -> None:
    """List running servers with links and stop controls."""
    manager = get_manager()
    servers = manager.servers

    st.markdown("### Running applications")
    if not servers:
        st.caption(
            "No project is running. Launch one above, or run a project directly with "
            "`streamlit run app.py` inside its directory."
        )
        return

    for slug, server in sorted(servers.items()):
        project = next((item for item in PROJECTS if item.slug == slug), None)
        name = project.title if project else slug
        left, middle, right = st.columns([4, 3, 2])
        with left:
            st.markdown(f"**{name}**")
            st.caption(f"Port {server.port} - up {server.uptime_seconds():.0f}s")
        with middle:
            if server.health_ok():
                st.markdown(f"[Open {name}]({server.url})")
            else:
                st.caption("Server is not responding to health checks")
        with right:
            if st.button("Stop", key=f"stop_{slug}", width="stretch"):
                if manager.stop(slug):
                    st.success(f"Stopped {name}.")
                    st.rerun()


def render_sidebar() -> None:
    """Render the sidebar with environment details and bulk actions."""
    manager = get_manager()
    with st.sidebar:
        st.markdown("### Suite status")
        st.markdown(f"**Projects:** {len(PROJECTS)}")
        st.markdown(f"**Running:** {len(manager.servers)}")
        st.markdown(f"**Root:** `{PROJECT_ROOT.name}`")

        st.markdown("---")
        st.markdown("### Actions")
        if st.button("Stop all running apps", width="stretch"):
            stopped = manager.stop_all()
            st.success(f"Stopped {stopped} application(s).")
            st.rerun()

        st.markdown("---")
        st.markdown("### Getting started")
        st.markdown(
            "1. `scripts/setup.sh` - create the environment\n"
            "2. `scripts/train_all.sh` - train every model\n"
            "3. `scripts/run_tests.sh` - run the test suite\n"
            "4. Launch a project from the cards"
        )


def render_instructions() -> None:
    """Explain how to train projects and run them directly."""
    with st.expander("Training and running projects directly"):
        st.markdown(
            "Each project is fully self-contained. From a project directory:\n\n"
            "```bash\n"
            "cd 03-handwritten-digit-recognizer\n"
            "python train.py          # creates models/*.pt\n"
            "streamlit run app.py     # launches just this project\n"
            "```\n\n"
            "To train everything at once from the repository root:\n\n"
            "```bash\n"
            "./scripts/train_all.sh\n"
            "```\n\n"
            "Some projects download their dataset on first training run; the README of each "
            "project documents the source, size and licence."
        )
        st.markdown("**Trained artefacts on disk**")
        for project in PROJECTS:
            ready, description = readiness(project)
            mark = "yes" if ready else "no"
            st.markdown(f"- `{project.number}` {project.title} - trained: **{mark}** ({description})")


def main() -> None:
    """Render the dashboard."""
    configure_logging()
    st.set_page_config(page_title=SUITE_TITLE, page_icon="AI", layout="wide")
    inject_theme()

    page_header(
        SUITE_TITLE,
        SUITE_SUBTITLE,
        eyebrow="Project suite",
        chips=[
            "Python 3.11+",
            "scikit-learn",
            "PyTorch",
            "Streamlit",
            "10 projects",
        ],
    )

    render_sidebar()

    try:
        render_overview()
        render_project_grid()
        render_running_servers()
        render_instructions()
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)

    note(
        "Every model in this suite is trained from real data with a documented evaluation "
        "protocol. Metrics shown inside each project come from a held-out split, and the "
        "forecasting projects are explicitly labelled as educational rather than "
        "production-grade predictions.",
        accent=PALETTE["accent"],
    )


if __name__ == "__main__":
    main()
