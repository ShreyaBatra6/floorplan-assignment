"""Download the pretrained weights groundplan uses (public Hugging Face hub) into the local HF cache.

Running this once ahead of time means no download happens during a run (e.g. at a walk-in test).
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from groundplan.models.depth import BACKENDS, DEFAULT_BACKEND
from groundplan.models.registry import MODELS

WANTED = [BACKENDS[DEFAULT_BACKEND], MODELS["clip"]]
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


def main() -> int:
    for repo, rev in WANTED:
        path = snapshot_download(repo_id=repo, revision=rev, allow_patterns=PATTERNS)
        print(f"ok  {repo}@{rev} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
