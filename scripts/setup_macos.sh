#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3.12}"
VENV_DIR="${2:-.venv.macos}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN"
  echo "Install Python 3.12 (e.g. brew install python@3.12), then rerun."
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"
