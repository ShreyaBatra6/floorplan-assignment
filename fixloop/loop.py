"""The fix loop, one command per step, with the order enforced by git (see fixloop/README.md).

    python fixloop/loop.py status
    python fixloop/loop.py before  [--manifest benchmark/manifest.yaml]
    python fixloop/loop.py declare
    python fixloop/loop.py after   [--manifest benchmark/manifest.yaml]

before   Needs a clean tree and a filled manifest. Tags HEAD ``fixloop-before``, runs the benchmark
         into benchmark/results/<sha>/, ranks the failing gates by how much of their requirement
         they meet, writes section 1 of DECLARATION.md (worst gate, threshold, failing number, run,
         commit, and every failing gate for context) and commits the results with it.
declare  Refuses while sections 1-3 still hold blanks, or if src/ or tests/ changed since
         ``fixloop-before``. Commits the declaration and tags it ``fixloop-declared``: the root
         cause and the predicted number are on record before any fix exists.
after    Refuses unless src/ changed after ``fixloop-declared``. Tags HEAD ``fixloop-after``,
         regenerates the after run from its own worktree (run_fixloop.py; the before run made by
         ``before`` at the tagged commit is reused unless --fresh) and writes the declared gate's
         after-run value into section 4. Prediction vs outcome is then written by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECL = ROOT / "fixloop" / "DECLARATION.md"
BEFORE, DECLARED, AFTER = "fixloop-before", "fixloop-declared", "fixloop-after"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r


def has_tag(tag: str) -> bool:
    return git("rev-parse", "-q", "--verify", f"refs/tags/{tag}", check=False).returncode == 0


def tree_clean() -> bool:
    return git("status", "--porcelain", "--untracked-files=no").stdout.strip() == ""


def changed_since(ref: str, *paths: str) -> bool:
    """True if the working tree differs from ``ref`` under ``paths``."""
    return git("diff", "--quiet", ref, "--", *paths, check=False).returncode == 1


def failing_gates(metrics: dict) -> list[dict]:
    """Every failing gate (repeatability included), worst first: smallest share of its requirement met."""
    rows = []
    for g in metrics.get("gates", []):
        if g.get("passed") is False:
            rows.append({k: g.get(k) for k in ("name", "tier", "threshold", "value", "detail", "score")})
    for rp in metrics.get("repeatability", []):
        if rp.get("passed") is False and rp.get("rows"):
            ok, n = sum(r["ok"] for r in rp["rows"]), len(rp["rows"])
            rows.append({"name": f"repeatability ({rp['pair'][0]} vs {rp['pair'][1]})", "tier": rp.get("tier", "?"),
                         "threshold": "walls within max(1 cm, 0.5 %), ceilings within the spread limit",
                         "value": f"{ok}/{n} within tolerance", "detail": "", "score": ok / n})
    return sorted(rows, key=lambda r: 1.0 if r["score"] is None else r["score"])


_BLANK = re.compile(r"_{4,}|(:\s*\.\.\.\s*$)|(^\s*(\d+\.)?\s*\.\.\.\s*$)", re.M)


def blanks(text: str) -> list[str]:
    """Lines of sections 1-3 that still hold a template placeholder."""
    body = text.split("## 4.")[0]
    return [ln.strip() for ln in body.splitlines() if _BLANK.search(ln)]


def section1(worst: dict, others: list[dict], run_dir: str, sha: str) -> str:
    lines = ["## 1. Worst-performing gate", "",
             f"- Gate: **{worst['name']}** (tier: {worst['tier']})",
             f"- Threshold (from the brief): {worst['threshold']}",
             f"- Failing number: **{worst['value']}** on benchmark run `{run_dir}` (commit `{sha}`, tag `{BEFORE}`)"]
    if worst.get("detail"):
        lines.append(f"- Detail: {worst['detail']}")
    lines += ["", "All failing gates of that run, worst first (share of the requirement met):", "",
              "| gate | tier | value | met |", "|---|---|---|---|"]
    for r in [worst] + others:
        met = "-" if r["score"] is None else f"{r['score']:.0%}"
        lines.append(f"| {r['name']} | {r['tier']} | {r['value']} | {met} |")
    return "\n".join(lines) + "\n\n"


def replace_section(text: str, number: int, new: str) -> str:
    start = text.index(f"## {number}.")
    nxt = text.find(f"## {number + 1}.", start)
    return text[:start] + new + (text[nxt:] if nxt >= 0 else "")


def declared_gate(text: str) -> tuple[str, str] | None:
    m = re.search(r"- Gate: \*\*(.+?)\*\* \(tier: (\w+)\)", text)
    return (m.group(1), m.group(2)) if m else None


def manifest_captures(manifest: Path) -> int:
    import yaml

    doc = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return len(doc.get("captures") or [])


def cmd_status(a) -> int:
    tags = {t: has_tag(t) for t in (BEFORE, DECLARED, AFTER)}
    n = manifest_captures(Path(a.manifest))
    print(f"manifest: {n} capture(s) in {a.manifest}")
    for t, ok in tags.items():
        print(f"  {t:17} {'set at ' + git('rev-parse', '--short', t).stdout.strip() if ok else '-'}")
    if n == 0:
        nxt = "capture the benchmark and fill benchmark/manifest.yaml (docs/BENCHMARK_PROTOCOL.md)"
    elif not tags[BEFORE]:
        nxt = "python fixloop/loop.py before"
    elif not tags[DECLARED]:
        nxt = "fill sections 2-3 of fixloop/DECLARATION.md, then: python fixloop/loop.py declare"
    elif not tags[AFTER]:
        nxt = "commit the fix, then: python fixloop/loop.py after"
    else:
        nxt = "fill section 4 of fixloop/DECLARATION.md (prediction vs outcome) and commit"
    print(f"next: {nxt}")
    return 0


def cmd_before(a) -> int:
    manifest = Path(a.manifest).resolve()
    if manifest_captures(manifest) == 0:
        raise SystemExit("the manifest lists no captures: capture the benchmark first (docs/BENCHMARK_PROTOCOL.md)")
    if has_tag(BEFORE):
        raise SystemExit(f"tag {BEFORE} already exists ({git('rev-parse', '--short', BEFORE).stdout.strip()}); "
                         f"delete it with `git tag -d {BEFORE}` only if you mean to restart the loop")
    if not tree_clean():
        raise SystemExit("commit or stash your changes first: the before-run must come from a committed state")
    sha = git("rev-parse", "--short", "HEAD").stdout.strip()
    git("tag", "-a", BEFORE, "-m", "Fix loop: code state of the before benchmark run")
    out = ROOT / "benchmark" / "results" / sha
    print(f"[before] benchmark at {sha} -> {out.relative_to(ROOT)} (this takes a while) ...", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}  # this checkout's code, whatever is installed
    r = subprocess.run([sys.executable, "-m", "groundplan.cli", "bench", "run", "--manifest", str(manifest),
                        "--out", str(out)], cwd=ROOT, env=env)
    if r.returncode != 0:
        git("tag", "-d", BEFORE)
        raise SystemExit("the benchmark run failed; tag removed, nothing committed")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    fails = failing_gates(metrics)
    run_dir = out.relative_to(ROOT).as_posix()
    if not fails:
        print("every gate passes: there is no failing gate to fix. Commit the results and say so in the report.")
        return 0
    text = DECL.read_text(encoding="utf-8")
    DECL.write_text(replace_section(text, 1, section1(fails[0], fails[1:], run_dir, sha)), encoding="utf-8",
                    newline="\n")
    git("add", run_dir, DECL.relative_to(ROOT).as_posix())
    git("commit", "-q", "-m", f"Fix loop: benchmark before the fix (worst gate: {fails[0]['name']}, "
                              f"{fails[0]['tier']}, {fails[0]['value']})")
    print(f"[before] worst gate: {fails[0]['name']} ({fails[0]['tier']}): {fails[0]['value']} "
          f"vs {fails[0]['threshold']}")
    print("next: write the root cause, evidence, fix and predicted number in sections 2-3 of "
          "fixloop/DECLARATION.md, then run: python fixloop/loop.py declare")
    return 0


def cmd_declare(a) -> int:
    if not has_tag(BEFORE):
        raise SystemExit(f"run `python fixloop/loop.py before` first (no {BEFORE} tag)")
    if has_tag(DECLARED):
        raise SystemExit(f"already declared at {git('rev-parse', '--short', DECLARED).stdout.strip()}")
    if changed_since(BEFORE, "src", "tests"):
        raise SystemExit(f"src/ or tests/ changed since {BEFORE}: the declaration must come before any fix "
                         "(move those changes aside, declare, then re-apply them)")
    left = blanks(DECL.read_text(encoding="utf-8"))
    if left:
        raise SystemExit("sections 1-3 of fixloop/DECLARATION.md still have blanks:\n  " + "\n  ".join(left))
    git("add", DECL.relative_to(ROOT).as_posix())
    if git("diff", "--cached", "--quiet", check=False).returncode == 1:
        git("commit", "-q", "-m", "Fix loop: declaration (root cause, fix, predicted number) before the fix")
    git("tag", "-a", DECLARED, "-m", "Fix loop: prediction on record before the fix")
    print(f"[declare] tagged {DECLARED} at {git('rev-parse', '--short', 'HEAD').stdout.strip()}. "
          "Now write the fix in ordinary commits, then run: python fixloop/loop.py after")
    return 0


def cmd_after(a) -> int:
    if not has_tag(DECLARED):
        raise SystemExit(f"declare first (no {DECLARED} tag)")
    if has_tag(AFTER):
        raise SystemExit(f"tag {AFTER} already exists; delete it with `git tag -d {AFTER}` to move it")
    if not tree_clean():
        raise SystemExit("commit the fix first")
    if git("diff", "--quiet", DECLARED, "HEAD", "--", "src", check=False).returncode == 0:
        raise SystemExit(f"no change under src/ since {DECLARED}: there is no fix to measure")
    git("tag", "-a", AFTER, "-m", "Fix loop: code state of the after benchmark run")
    cmd = [sys.executable, str(ROOT / "fixloop" / "run_fixloop.py"), "--before", BEFORE, "--after", AFTER,
           "--manifest", str(Path(a.manifest).resolve())]
    before_run = ROOT / "benchmark" / "results" / git("rev-parse", "--short", BEFORE).stdout.strip()
    if not a.fresh and (before_run / "metrics.json").exists():
        cmd += ["--reuse-before", str(before_run)]  # made by `before` at the tagged commit; --fresh redoes it
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("regeneration failed (tag kept); fix the problem and re-run run_fixloop.py")
    text = DECL.read_text(encoding="utf-8")
    gate = declared_gate(text)
    after_m = json.loads((ROOT / "fixloop" / "runs" / AFTER / "metrics.json").read_text(encoding="utf-8"))
    hit = [g for g in after_m.get("gates", []) if gate and (g["name"], g["tier"]) == gate]
    if hit:
        g = hit[0]
        status = "PASS" if g["passed"] else "FAIL"
        text = re.sub(r"- After-run value: _+ \(commit `fixloop-after`\)",
                      f"- After-run value: **{g['value']}** ({status}; commit "
                      f"`{git('rev-parse', '--short', AFTER).stdout.strip()}`, tag `{AFTER}`)", text)
        DECL.write_text(text, encoding="utf-8", newline="\n")
        print(f"[after] {gate[0]} ({gate[1]}): {g['value']} -> {status}")
    print("next: fill 'Prediction vs outcome' in section 4 of fixloop/DECLARATION.md, then commit "
          "fixloop/RESULT.md, fixloop/fix.diff and the declaration")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["status", "before", "declare", "after"])
    ap.add_argument("--manifest", default=str(ROOT / "benchmark" / "manifest.yaml"))
    ap.add_argument("--fresh", action="store_true", help="after: regenerate the before run too")
    a = ap.parse_args()
    return {"status": cmd_status, "before": cmd_before, "declare": cmd_declare, "after": cmd_after}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
