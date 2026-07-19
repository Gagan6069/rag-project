#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"$PYTHON_BIN" --version

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

DEPENDENCY_FILE="requirements-lock.txt"

if [ ! -f "$DEPENDENCY_FILE" ]; then
  DEPENDENCY_FILE="requirements.txt"
fi

python -m pip install -r "$DEPENDENCY_FILE"

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env. Add your GROQ_API_KEY."
fi

echo
echo "Installation completed."
echo "Run:"
echo "  python -m src.vectordb"
echo "  python -m src.rag"
echo "  python -m src.run_eval"