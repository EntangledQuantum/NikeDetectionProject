#!/usr/bin/env bash
# Create an isolated Nike Detection environment OUTSIDE the repo (macOS / Linux).
# Usage:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh
#   ENV_DIR=$HOME/.venvs/nike-detection ./scripts/setup_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/nike-detection}"
OS="$(uname -s)"

echo "==> Nike Detection — environment setup ($OS)"
echo "    Project: $PROJECT_ROOT"
echo "    Env dir: $ENV_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi

if [[ "$OS" == "Darwin" ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "==> macOS: creating venv with uv"
    uv venv "$ENV_DIR" --python python3
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    uv pip install -r "$PROJECT_ROOT/requirements.txt"
  else
    echo "==> macOS: uv not found; using python3 -m venv"
    python3 -m venv "$ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    python -m pip install --upgrade pip wheel
    pip install -r "$PROJECT_ROOT/requirements.txt"
  fi
elif [[ "$OS" == "Linux" ]]; then
  if command -v conda >/dev/null 2>&1; then
    echo "==> Linux: creating conda env 'nike-detection'"
    conda create -y -p "$ENV_DIR" python=3.11 pip
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate" 2>/dev/null || conda activate "$ENV_DIR"
    pip install -r "$PROJECT_ROOT/requirements.txt"
  elif command -v uv >/dev/null 2>&1; then
    echo "==> Linux: creating venv with uv"
    uv venv "$ENV_DIR" --python python3
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    uv pip install -r "$PROJECT_ROOT/requirements.txt"
  else
    echo "==> Linux: using python3 -m venv"
    python3 -m venv "$ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    python -m pip install --upgrade pip wheel
    pip install -r "$PROJECT_ROOT/requirements.txt"
  fi
else
  echo "ERROR: Unsupported OS: $OS" >&2
  exit 1
fi

echo ""
echo "Done. Activate and run:"
echo "  source \"$ENV_DIR/bin/activate\""
echo "  cd \"$PROJECT_ROOT\""
echo "  python -m nike_detection -i data/blackStripe.tiff --only stripe_misalignment --no-vis"
