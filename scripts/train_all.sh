#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train every project's model in sequence.
#
# Each project's train.py is run from its own directory so that its `src`
# package resolves correctly. Artefacts land in <project>/models/.
#
# Usage:
#   scripts/train_all.sh
#   scripts/train_all.sh 01 07        # train only the listed project numbers
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

# Projects that require a training step. Project 10 uses a pretrained speech
# model that downloads on first use, so it has no train.py.
TRAINABLE=(
  "01-stock-price-predictor"
  "02-twitter-sentiment-analysis"
  "03-handwritten-digit-recognizer"
  "04-movie-recommendation-system"
  "05-customer-churn-prediction"
  "06-customer-service-chatbot"
  "07-spam-email-classifier"
  "08-fruit-image-classifier"
  "09-weather-analysis-prediction"
)

selected=()
if [[ $# -gt 0 ]]; then
  for number in "$@"; do
    for project in "${TRAINABLE[@]}"; do
      if [[ "$project" == "$number"* ]]; then
        selected+=("$project")
      fi
    done
  done
else
  selected=("${TRAINABLE[@]}")
fi

failures=()
for project in "${selected[@]}"; do
  echo ""
  echo "=============================================================="
  echo " Training $project"
  echo "=============================================================="
  if [[ ! -f "$project/train.py" ]]; then
    echo "  no train.py - skipping"
    continue
  fi
  if (cd "$project" && "$PYTHON" train.py); then
    echo "  OK: $project"
  else
    echo "  FAILED: $project"
    failures+=("$project")
  fi
done

echo ""
echo "=============================================================="
if [[ ${#failures[@]} -eq 0 ]]; then
  echo "All selected projects trained successfully."
  exit 0
fi
echo "These projects failed: ${failures[*]}"
exit 1
