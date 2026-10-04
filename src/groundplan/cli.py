"""Command line interface: one command per capture.

    groundplan run <capture>          # tier auto-detected; writes runs/<name>/
    groundplan inspect <capture>      # what the pipeline sees, before running it
    groundplan validate <plan.json>   # check a plan against the published schema
    groundplan schema                 # regenerate schema/groundplan.schema.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from groundplan import SCHEMA_VERSION, __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schema" / "groundplan.schema.json"


@app.command()
def version() -> None:
    """Print the package and schema versions."""
    console.print(f"groundplan {__version__} (schema {SCHEMA_VERSION})")


@app.command()
def schema(out: Path = typer.Option(SCHEMA_PATH, help="where to write the JSON Schema")) -> None:
    """Regenerate the published JSON Schema from the contract models."""
    from groundplan.contract import json_schema

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8", newline="\n")
    console.print(f"wrote {out}")


@app.command()
def run(
    capture: Path = typer.Argument(..., help="Stray Scanner folder/zip, a video, or a folder of room photo folders"),
    out: Path = typer.Option(None, "--out", "-o", help="output folder (default: runs/<capture-name>)"),
    tier: str = typer.Option(None, help="force a tier: lidar | video | photo"),
    drift: bool = typer.Option(True, "--drift/--no-drift", help="drift correction (ablation: --no-drift)"),
    damage: bool = typer.Option(True, "--damage/--no-damage", help="damage detection, flags and scope"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="replay cached model outputs when available"),
) -> None:
    """Process one capture into plan.json, rendered plans and a summary."""
    from groundplan.io.stray import CaptureError
    from groundplan.pipeline import RunOptions, run_capture
    from groundplan.summary import summary_markdown

    out = out or Path("runs") / (capture.stem if capture.is_file() else capture.name)
    try:
        res = run_capture(capture, out, RunOptions(tier=tier, drift=drift, damage=damage, use_cache=cache))
    except CaptureError as exc:
        console.print(f"[red]cannot process capture:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    console.print(summary_markdown(res.plan))
    console.print(f"[green]wrote[/green] {out} ({res.plan.runtime.total_s:.1f} s)")


@app.command()
def inspect(capture: Path = typer.Argument(..., help="capture to inspect")) -> None:
    """Show which tier the capture will run as and what it contains, without processing it."""
    from groundplan.io.detect import detect, unpack_if_zip
    from groundplan.pipeline import CACHE_DIR

    det = detect(unpack_if_zip(capture, CACHE_DIR))
    console.print(f"tier: [bold]{det.tier}[/bold]\nroot: {det.root}\n{det.detail}")
    if det.tier == "lidar":
        from groundplan.io.stray import load_stray

        cap = load_stray(det.root)
        console.print(f"frames: {cap.n} ({int(cap.has_depth.sum())} with depth), {cap.duration_s:.1f} s, "
                      f"rgb {cap.rgb_size}, depth {cap.depth_size}")
        for note in cap.notes:
            console.print(f"[yellow]note:[/yellow] {note}")


@app.command()
def validate(plans: list[Path] = typer.Argument(..., help="plan.json files to check")) -> None:
    """Validate plan documents against the published JSON Schema."""
    from groundplan.validate import validate_file

    failed = 0
    for path in plans:
        errors = validate_file(path)
        if errors:
            failed += 1
            console.print(f"[red]FAIL[/red] {path}")
            for err in errors[:20]:
                console.print(f"   {err}")
        else:
            console.print(f"[green]ok[/green]   {path}")
    raise typer.Exit(code=1 if failed else 0)


bench_app = typer.Typer(help="Benchmark: run every capture in the manifest and score it against laser/tape truth.")
app.add_typer(bench_app, name="bench")


@bench_app.command("run")
def bench_run(
    manifest: Path = typer.Option(REPO_ROOT / "benchmark" / "manifest.yaml", help="benchmark manifest"),
    out: Path = typer.Option(None, help="results folder (default benchmark/results/<git sha>)"),
    only: list[str] = typer.Option(None, help="restrict to these capture ids"),
) -> None:
    """Run all captures live, then score them (gates, repeatability, drift ablation, calibration, head-to-head)."""
    from groundplan.bench.runner import run_benchmark
    from groundplan.pipeline import _git_commit

    out = out or REPO_ROOT / "benchmark" / "results" / (_git_commit() or "local")
    res = run_benchmark(manifest, out, only)
    for g in res["gates"]:
        status = "n/a" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
        console.print(f"{status:5} {g['tier']:6} {g['name']}: {g['value']}")
    console.print(f"report: {out / 'benchmark_report.md'}")


@bench_app.command("score")
def bench_score(results: Path = typer.Argument(..., help="results folder with <capture>/plan.json"),
                manifest: Path = typer.Option(REPO_ROOT / "benchmark" / "manifest.yaml")) -> None:
    """Rescore saved plans without re-running the pipeline."""
    from groundplan.bench.runner import load_manifest, score_benchmark

    res = score_benchmark(load_manifest(manifest), results)
    for g in res["gates"]:
        status = "n/a" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
        console.print(f"{status:5} {g['tier']:6} {g['name']}: {g['value']}")


@bench_app.command("calibrate")
def bench_calibrate(results: Path = typer.Argument(..., help="results folder containing metrics.json"),
                    write: bool = typer.Option(False, help="write the fitted multipliers to calibration.json"),
                    to: Path = typer.Option(None, help="write to this file instead (a calibration for one device "
                                                       "or dataset), starting from the current one")) -> None:
    """Fit split-conformal interval multipliers per tier and quantity (leave-one-capture-out coverage)."""
    from groundplan.bench.calibrate import apply, fit

    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    rep = fit(metrics)
    d = rep["depth_scale"]
    if d["adopted"]:
        console.print(f"LiDAR depth scale {d['previous']} -> {d['depth_scale']} (held-out mean error "
                      f"{d['loo_mae_before_m'] * 100:.1f} -> {d['loo_mae_after_m'] * 100:.1f} cm, n={d['n']})")
    else:
        console.print(f"LiDAR depth scale kept at {d['previous']}: {d.get('reason', '')}")
    for key, row in rep["loo"].items():
        console.print(f"{key:28} n={row['n']:3d} factor={row['factor'] if row['factor'] is None else round(row['factor'], 3)} "
                      f"LOO coverage={row['loo_coverage']}")
    if write or to:
        import os

        from groundplan.calib.intervals import DEFAULT_PATH

        base = Path(os.environ.get("GROUNDPLAN_CALIBRATION") or DEFAULT_PATH)
        console.print(f"wrote {apply(rep, str(results), base, to)}")


@app.command("eval")
def eval_plan(plan: Path = typer.Argument(..., help="plan.json"),
              truth: Path = typer.Argument(..., help="ground-truth YAML of the site")) -> None:
    """Score one plan against its ground truth (walls, ceiling, openings, area, adjacency)."""
    from groundplan.bench.gates import ceiling_gate, opening_gate, score_capture, wall_gate
    from groundplan.bench.gt import load_ground_truth
    from groundplan.contract import Plan

    p = Plan.model_validate_json(plan.read_text(encoding="utf-8"))
    sc = score_capture(p, load_ground_truth(truth), plan.parent.name)
    for g in (wall_gate([sc], p.capture.tier), ceiling_gate([sc], p.capture.tier), opening_gate([sc], p.capture.tier)):
        status = "n/a" if g.passed is None else ("PASS" if g.passed else "FAIL")
        console.print(f"{status:5} {g.name}: {g.value} {g.detail}")
    for it in sc.items:
        console.print(f"  {it.kind:15} {it.room:12} {it.ref:8} truth {it.truth:.3f}  ours {it.value:.3f} "
                      f"[{it.lo:.3f},{it.hi:.3f}]  err {it.err * 100:+.1f} cm  {'in' if it.covered else 'OUT'}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; plans print m2 symbols
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":
    main()
