#!/usr/bin/env bash
set -e

if [ ! -f "pyproject.toml" ]; then
  echo "[Bootstrap] ERROR: pyproject.toml not found."
  echo "[Bootstrap] Please run this script from the repository root."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[Bootstrap] ERROR: python3 is not available in PATH."
  echo "[Bootstrap] Please install Python 3 and try again."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[Bootstrap] .venv not found. Creating virtual environment..."
  python3 -m venv .venv
  echo "[Bootstrap] Virtual environment created at .venv"
else
  echo "[Bootstrap] Reusing existing virtual environment: .venv"
fi

echo "[Bootstrap] Activating .venv..."
source .venv/bin/activate

echo "[Bootstrap] Installing with python -m pip inside .venv (avoids system pip/PEP 668 issues)..."
echo "[Bootstrap] Upgrading pip..."
python -m pip install --upgrade pip

echo "[Bootstrap] Installing project in editable mode..."
python -m pip install -e .

echo "[Bootstrap] Bootstrap completed successfully."
echo "[Bootstrap] Next command:"
echo "python scripts/smoke_test.py"
