"""Regenerate the fix loop's before and after runs from git, and write the comparison and the diff.

    python fixloop/run_fixloop.py [--before fixloop-before] [--after fixloop-after]

Both refs are checked out into git worktrees under fixloop/worktrees/, and the benchmark is run
from each worktree's own code (and its own calibration file) on the same raw data
(``GROUNDPLAN_DATA``). Results go to fixloop/runs/<ref>/; fixloop/RESULT.md compares the gates and
fixloop/fix.diff is the readable source diff between the two refs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixloop"


def sh(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> str:
    out = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return out.stdout


def run_ref(ref: str, manifest: Path) -> dict:
    wt = FIX / "worktrees" / ref
    if wt.exists():
        sh(["git", "worktree", "remove", "--force", str(wt)])
    sh(["git", "worktree", "add", "--detach", str(wt), ref])
    out = FIX / "runs" / ref
    env = dict(os.environ)
    env.setdefault("GROUNDPLAN_DATA", str(ROOT / "data"))
    sh(["uv", "run", "--project", str(wt), "groundplan", "bench", "run", "--manifest", str(manifest), "--out", str(out)],
       cwd=wt, env=env)
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="fixloop-before")
    ap.add_argument("--after", default="fixloop-after")
    ap.add_argument("--manifest", default=str(ROOT / "benchmark" / "manifest.yaml"))
    a = ap.parse_args()
    manifest = Path(a.manifest).resolve()
    before = run_ref(a.before, manifest)
    after = run_ref(a.after, manifest)
    (FIX / "fix.diff").write_text(sh(["git", "diff", f"{a.before}..{a.after}", "--", "src", "tests"]),
                                  encoding="utf-8", newline="\n")
    rows = ["# Fix loop: before vs after", "", f"`{a.before}` vs `{a.after}`, same raw data, each run from its own code.",
            "", "| gate | tier | before | after | before status | after status |", "|---|---|---|---|---|---|"]
    gb = {(g["name"], g["tier"]): g for g in before["gates"]}
    for g in after["gates"]:
        b = gb.get((g["name"], g["tier"]), {})
        st = lambda x: "n/a" if x is None else ("PASS" if x else "FAIL")  # noqa: E731
        rows.append(f"| {g['name']} | {g['tier']} | {b.get('value', '-')} | {g['value']} | {st(b.get('passed'))} | "
                    f"{st(g['passed'])} |")
    rows += ["", "Source diff: [`fix.diff`](fix.diff). Declaration and prediction: [`DECLARATION.md`](DECLARATION.md)."]
    (FIX / "RESULT.md").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print((FIX / "RESULT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
