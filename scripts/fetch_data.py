"""Fetch the raw benchmark data (captures, app exports) and verify every archive's SHA-256.

``data/MANIFEST.json`` lists the archives::

    {"archives": [{"name": "flat_a", "url": "https://.../flat_a.zip", "sha256": "...", "dest": "data/flat_a"}]}

Archives are downloaded once into ``data/_downloads/``, checked, and extracted to ``dest``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = ROOT / "data" / "MANIFEST.json"
    if not manifest.exists():
        print("data/MANIFEST.json not found: nothing to fetch (see docs/BENCHMARK_PROTOCOL.md section 5)")
        return 1
    archives = json.loads(manifest.read_text(encoding="utf-8"))["archives"]
    dl = ROOT / "data" / "_downloads"
    dl.mkdir(parents=True, exist_ok=True)
    bad = 0
    for a in archives:
        target = dl / f"{a['name']}.zip"
        if not target.exists() or sha256(target) != a["sha256"]:
            print(f"downloading {a['name']} ...")
            urllib.request.urlretrieve(a["url"], target)
        digest = sha256(target)
        if digest != a["sha256"]:
            print(f"CHECKSUM MISMATCH {a['name']}: {digest} != {a['sha256']}")
            bad += 1
            continue
        dest = ROOT / a["dest"]
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as zf:
            zf.extractall(dest)
        print(f"ok  {a['name']} -> {dest}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
