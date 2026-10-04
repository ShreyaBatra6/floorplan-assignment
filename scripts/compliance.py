"""Render docs/COMPLIANCE_MATRIX.md from docs/compliance.yaml and check every cited path exists."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STATUS = {"done": "done", "partial": "partial", "pending-data": "pending: benchmark data",
          "pending-fix": "pending: fix loop (after benchmark)"}


def main() -> int:
    items = yaml.safe_load((ROOT / "docs" / "compliance.yaml").read_text(encoding="utf-8"))
    missing = [(it["id"], p) for it in items for p in it["paths"] if not (ROOT / p).exists()]
    counts: dict[str, int] = {}
    for it in items:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    lines = ["# Compliance matrix", "",
             "Requirement → file path → artifact → status, for every requirement of the brief. Generated from "
             "`docs/compliance.yaml` by `scripts/compliance.py`, which also fails if any cited path is missing.", "",
             "Summary: " + ", ".join(f"{STATUS[k]} {v}" for k, v in counts.items()), "",
             "| ID | Part | Requirement | File path(s) | Artifact | Status |", "|---|---|---|---|---|---|"]
    for it in items:
        paths = "<br>".join(f"[`{p}`](../{p})" for p in it["paths"])
        lines.append(f"| {it['id']} | {it['part']} | {it['req']} | {paths} | {it['artifact']} | {STATUS[it['status']]} |")
    (ROOT / "docs" / "COMPLIANCE_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    if missing:
        for rid, p in missing:
            print(f"missing: {rid} -> {p}", file=sys.stderr)
        return 1
    print(f"ok: {len(items)} requirements, all paths present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
