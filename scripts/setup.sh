#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create the virtual environment and install dependencies.
#
# Usage:
#   scripts/setup.sh
#
# The script is idempotent: re-running it reuses the existing .venv.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No suitable Python interpreter found. Install Python 3.11 or newer." >&2
  exit 1
fi

VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using $PYTHON_BIN (Python $VERSION)"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required, found {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "Reusing existing .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Activate the environment with:"
echo "    source .venv/bin/activate        # Windows: .venv\\Scripts\\activate"
echo ""
echo "Then launch the dashboard:"
echo "    streamlit run app.py"
