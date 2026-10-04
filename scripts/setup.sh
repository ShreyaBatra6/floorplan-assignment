#!/usr/bin/env bash
# groundplan setup for macOS / Linux. Usage: bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

case "$(pwd)" in
  *OneDrive*|*Dropbox*|*"Mobile Documents"*)
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/groundplan}"
    echo "environment placed at $UV_PROJECT_ENVIRONMENT (outside the synced folder)";;
esac

uv sync --extra models --extra video
uv run python scripts/fetch_models.py
uv run pytest -q -m "not slow and not sample"
echo "ready: uv run groundplan run <capture>"
