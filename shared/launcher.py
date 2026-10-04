"""Launch and manage each project's Streamlit app as a child process.

Why a process launcher rather than in-process imports
----------------------------------------------------
Every project ships a package named ``src``, which is the required repository
layout. Python resolves modules by name through a global ``sys.modules`` cache,
so importing two projects' ``src`` packages into one interpreter is impossible:
the second import silently returns the first project's modules. Verified
directly during development:

    ImportError: cannot import name 'CLASS_LABELS' from 'src.data'
    (.../01-stock-price-predictor/src/data.py)

The same collision also breaks ``pickle``, because a class pickled as
``src.model.SentimentModel`` cannot be re-imported once ``src`` points elsewhere.
Running each app in its own process with its own working directory removes the
ambiguity entirely and is how Streamlit multi-page deployments normally work.

This module therefore starts a real ``streamlit run`` server per project, waits
for it to become healthy, and tracks it so it can be stopped again.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shared.errors import ProjectError

logger = logging.getLogger(__name__)

#: Ports are searched in this range, starting just above Streamlit's default.
PORT_RANGE_START = 8601

#: How many consecutive ports to probe before giving up.
PORT_RANGE_SIZE = 60

#: Seconds to wait for a freshly started server to answer its health endpoint.
STARTUP_TIMEOUT_SECONDS = 90

#: Interval between health checks.
HEALTH_POLL_SECONDS = 0.5

#: File recording live child processes, so a restarted dashboard can clean up.
REGISTRY_FILE = Path(__file__).resolve().parent.parent / ".streamlit_servers.json"


@dataclass
class RunningServer:
    """A Streamlit server started by the dashboard."""

    slug: str
    title: str
    port: int
    pid: int
    url: str
    started_at: float = field(default_factory=time.time)
    log_path: str = ""

    def is_alive(self) -> bool:
        """Whether the child process is still running."""
        try:
            os.kill(self.pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def uptime_seconds(self) -> float:
        """Seconds since the server was started."""
        return time.time() - self.started_at

    def health_ok(self, timeout: float = 2.0) -> bool:
        """Whether the server's health endpoint responds."""
        return _health_check(self.port, timeout=timeout)


# --------------------------------------------------------------------------- #
# Port and process helpers
# --------------------------------------------------------------------------- #


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Whether a TCP port can be bound on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(start: int = PORT_RANGE_START, count: int = PORT_RANGE_SIZE) -> int:
    """Return the first free port at or after ``start``.

    Raises:
        ProjectError: When no port in the scanned range is available.
    """
    for port in range(start, start + count):
        if _port_is_free(port):
            return port
    raise ProjectError(
        f"No free port found in the range {start}-{start + count - 1}.",
        hint="Close some of the running project servers and try again.",
    )


def _health_check(port: int, timeout: float = 2.0) -> bool:
    """Query Streamlit's health endpoint on ``port``."""
    url = f"http://127.0.0.1:{port}/_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200 and response.read().strip().lower() == b"ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _registry_path() -> Path:
    return REGISTRY_FILE


def _save_registry(servers: dict[str, RunningServer]) -> None:
    """Persist live server details so a restart can adopt or clean them up."""
    try:
        payload = {slug: asdict(server) for slug, server in servers.items() if server.is_alive()}
        _registry_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as error:  # pragma: no cover - disk-level failure
        logger.warning("Could not persist the server registry: %s", error)


def _load_registry() -> dict[str, RunningServer]:
    """Read previously started servers, dropping any that are no longer alive."""
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    servers: dict[str, RunningServer] = {}
    for slug, record in payload.items():
        try:
            server = RunningServer(**record)
        except TypeError:
            continue
        if server.is_alive():
            servers[slug] = server
    return servers


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


class ServerManager:
    """Tracks the Streamlit servers started from the dashboard."""

    def __init__(self) -> None:
        # Adopt servers that survived a dashboard restart.
        self._servers: dict[str, RunningServer] = _load_registry()
        atexit.register(self.stop_all)

    @property
    def servers(self) -> dict[str, RunningServer]:
        """Currently tracked servers, pruned of dead processes."""
        self._servers = {
            slug: server for slug, server in self._servers.items() if server.is_alive()
        }
        return self._servers

    def get(self, slug: str) -> RunningServer | None:
        """Return the running server for a project, if any."""
        server = self.servers.get(slug)
        return server if server is not None and server.is_alive() else None

    def start(self, slug: str, app_file: Path, title: str, *, cwd: Path) -> RunningServer:
        """Start ``app_file`` in a new process and wait until it is healthy.

        Args:
            slug: Project directory name, used as the registry key.
            app_file: Path to the project's ``app.py``.
            title: Display name, stored for the dashboard listing.
            cwd: Working directory for the child process. This must be the
                project directory so its ``src`` package resolves correctly.

        Returns:
            The :class:`RunningServer` handle.

        Raises:
            ProjectError: When the app file is missing, the server exits early,
                or it does not become healthy within the timeout.
        """
        existing = self.get(slug)
        if existing is not None and existing.health_ok():
            return existing

        if not app_file.exists():
            raise ProjectError(
                f"No app.py found for '{title}'.",
                hint=f"Expected the file at {app_file}.",
            )

        port = find_free_port()
        log_path = cwd / "models" / f"streamlit_{port}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_file),
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]

        logger.info("Starting %s on port %d (cwd=%s)", title, port, cwd)
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603 - fixed argument list, no shell
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        deadline = time.time() + STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if process.poll() is not None:
                handle.close()
                tail = _read_tail(log_path)
                raise ProjectError(
                    f"'{title}' exited immediately (code {process.returncode}).",
                    hint=f"Check the log for details:\n\n```\n{tail}\n```",
                )
            if _health_check(port):
                break
            time.sleep(HEALTH_POLL_SECONDS)
        else:
            process.terminate()
            handle.close()
            raise ProjectError(
                f"'{title}' did not become ready within {STARTUP_TIMEOUT_SECONDS} seconds.",
                hint=f"It may still be starting. Check the log: {log_path}",
            )

        server = RunningServer(
            slug=slug,
            title=title,
            port=port,
            pid=process.pid,
            url=f"http://127.0.0.1:{port}",
            log_path=str(log_path),
        )
        self._servers[slug] = server
        _save_registry(self._servers)
        logger.info("Started %s at %s", title, server.url)
        return server

    def stop(self, slug: str) -> bool:
        """Terminate a project's server. Returns ``True`` when one was stopped."""
        server = self._servers.pop(slug, None)
        if server is None:
            return False
        _terminate(server.pid)
        _save_registry(self._servers)
        return True

    def stop_all(self) -> int:
        """Terminate every tracked server. Returns the number stopped."""
        stopped = 0
        for slug in list(self._servers):
            if self.stop(slug):
                stopped += 1
        return stopped


def _terminate(pid: int, *, timeout: float = 10.0) -> None:
    """Ask a process to stop, escalating to SIGKILL if it does not exit."""
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return
        time.sleep(0.2)

    try:  # pragma: no cover - only when a process ignores SIGTERM
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _read_tail(path: Path, lines: int = 20) -> str:
    """Return the last ``lines`` of a log file, or a placeholder on failure."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(log unavailable)"
    return "\n".join(content[-lines:]) or "(log is empty)"


def get_manager() -> ServerManager:
    """Return the process-wide :class:`ServerManager`.

    Streamlit re-executes the whole script on every rerun, so the manager is
    cached on the module to avoid starting duplicate servers.
    """
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ServerManager()
    return _MANAGER


_MANAGER: ServerManager | None = None