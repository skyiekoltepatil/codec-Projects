#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the test suite.
#
# Why each test module gets its own pytest process
# ------------------------------------------------
# Every project in this repository ships a package named `src`, which is the
# required layout. Python caches modules by name in `sys.modules`, so a single
# interpreter can only ever hold one project's `src` package. Running the whole
# suite in one process would make later modules silently import the first
# project's code, and would also break pickling of model classes.
#
# Running one pytest process per test module is the standard, honest fix: it
# keeps the required directory structure and makes each project's tests
# genuinely independent.
#
# Usage:
#   scripts/run_tests.sh                 # run everything
#   scripts/run_tests.sh -m "not network"  # skip tests needing the internet
#   scripts/run_tests.sh -k stock        # extra pytest args are forwarded
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

if [[ ! -d tests ]]; then
  echo "No tests/ directory found. Run this script from the repository root." >&2
  exit 1
fi

# Collect test modules, excluding conftest and helper modules.
# A while-read loop is used instead of `mapfile`, which is unavailable in the
# bash 3.2 that ships with macOS.
MODULES=()
while IFS= read -r module; do
  MODULES+=("$module")
done < <(find tests -maxdepth 1 -name 'test_*.py' | sort)

if [[ ${#MODULES[@]} -eq 0 ]]; then
  echo "No test modules found in tests/." >&2
  exit 1
fi

FAILED=0
TOTAL_PASSED=0

for module in "${MODULES[@]}"; do
  echo ""
  echo "=============================================================="
  echo " $module"
  echo "=============================================================="
  if "$PYTHON" -m pytest "$module" "$@"; then
    :
  else
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=============================================================="
if [[ $FAILED -eq 0 ]]; then
  echo "All ${#MODULES[@]} test modules passed."
  exit 0
else
  echo "${FAILED} of ${#MODULES[@]} test modules failed."
  exit 1
fi
