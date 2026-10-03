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
    out.write_text(json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8")
    console.print(f"wrote {out}")


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
