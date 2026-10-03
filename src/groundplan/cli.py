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


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":
    main()
