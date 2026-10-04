"""Regenerate the fix loop's before and after runs from git, and write the comparison and the diff.

    python fixloop/run_fixloop.py [--before fixloop-before] [--after fixloop-after]
                                  [--manifest benchmark/manifest.yaml] [--out fixloop]

Both refs are checked out into git worktrees and the benchmark is run from each worktree's own code
(and its own calibration file), each in its own Python environment, on the same raw data
(``GROUNDPLAN_DATA``). Results go to <out>/runs/<ref>/; <out>/RESULT.md compares every gate and
<out>/fix.diff is the readable source diff between the two refs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_BASE = Path(os.environ.get("GROUNDPLAN_FIXLOOP_ENVS", Path.home() / ".venvs"))


def sh(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> str:
    out = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        sys.stderr.write(out.stdout[-4000:] + out.stderr[-4000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return out.stdout


def find_uv() -> str:
    """uv from $UV, PATH, or the usual per-user install folders (pip --user does not add them to PATH)."""
    found = os.environ.get("UV") or shutil.which("uv")
    if found:
        return found
    homes = [Path(os.environ.get("APPDATA", "")) / "Python", Path.home() / ".local" / "bin", Path.home() / ".cargo" / "bin"]
    for base in homes:
        for pattern in ("*/Scripts/uv.exe", "uv.exe", "uv"):
            hits = sorted(base.glob(pattern)) if base.exists() else []
            if hits:
                return str(hits[-1])
    raise SystemExit("uv not found: `python -m pip install uv`, or set UV to the uv executable")


def safe(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", ref)


def run_ref(ref: str, manifest: Path, out_dir: Path) -> dict:
    wt = out_dir / "worktrees" / safe(ref)
    if wt.exists():
        sh(["git", "worktree", "remove", "--force", str(wt)])
    sh(["git", "worktree", "prune"])
    sh(["git", "worktree", "add", "--detach", str(wt), ref])
    res = out_dir / "runs" / safe(ref)
    env = dict(os.environ)
    env.setdefault("GROUNDPLAN_DATA", str(ROOT / "data"))
    env["UV_PROJECT_ENVIRONMENT"] = str(ENV_BASE / f"groundplan-fixloop-{safe(ref)}")  # one env per ref
    print(f"[{ref}] running the benchmark from {wt} ...", flush=True)
    sh([find_uv(), "run", "--project", str(wt), "--all-extras", "groundplan", "bench", "run",
        "--manifest", str(manifest), "--out", str(res)], cwd=wt, env=env)
    return json.loads((res / "metrics.json").read_text(encoding="utf-8"))


def _status(x) -> str:
    return "n/a" if x is None else ("PASS" if x else "FAIL")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="fixloop-before")
    ap.add_argument("--after", default="fixloop-after")
    ap.add_argument("--manifest", default=str(ROOT / "benchmark" / "manifest.yaml"))
    ap.add_argument("--out", default=str(ROOT / "fixloop"))
    ap.add_argument("--reuse-before", default=None,
                    help="results folder of a benchmark run made at the --before commit (e.g. by `loop.py before`); "
                         "used instead of running the before benchmark again")
    a = ap.parse_args()
    manifest = Path(a.manifest).resolve()
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.reuse_before:
        before = json.loads((Path(a.reuse_before) / "metrics.json").read_text(encoding="utf-8"))
        how = (f"the before run is `{Path(a.reuse_before).as_posix()}`, made at the `{a.before}` commit; the after run "
               "is regenerated from its own worktree")
    else:
        before = run_ref(a.before, manifest, out_dir)
        how = "each run regenerated from its own worktree"
    after = run_ref(a.after, manifest, out_dir)
    (out_dir / "fix.diff").write_text(sh(["git", "diff", f"{a.before}..{a.after}", "--", "src", "tests"]),
                                      encoding="utf-8", newline="\n")
    rows = ["# Fix loop: before vs after", "",
            f"`{a.before}` vs `{a.after}`, same raw data (`{manifest.name}`); {how}.", "",
            "| gate | tier | before | after | before status | after status |", "|---|---|---|---|---|---|"]
    gb = {(g["name"], g["tier"]): g for g in before["gates"]}
    for g in after["gates"]:
        b = gb.get((g["name"], g["tier"]), {})
        rows.append(f"| {g['name']} | {g['tier']} | {b.get('value', '-')} | {g['value']} | {_status(b.get('passed'))} | "
                    f"{_status(g['passed'])} |")
    for label, m in (("before", before), ("after", after)):
        for rp in m.get("repeatability", []):
            rows.append(f"\nRepeatability {label} ({rp['pair'][0]} vs {rp['pair'][1]}): {_status(rp['passed'])}, "
                        f"{sum(r['ok'] for r in rp['rows'])}/{len(rp['rows'])} within tolerance")
    n_diff = len([ln for ln in (out_dir / "fix.diff").read_text(encoding="utf-8").splitlines()
                  if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))])
    rows += ["", f"Source diff: [`fix.diff`](fix.diff) ({n_diff} changed lines). "
             "Declaration and prediction: `fixloop/DECLARATION.md`."]
    (out_dir / "RESULT.md").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print((out_dir / "RESULT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
