"""Human-readable summary (Markdown) of a Plan: the table you read out at the walk-in test."""

from __future__ import annotations

from pathlib import Path

from groundplan.contract import Measurement, Plan


def ci(m: Measurement, unit: str = "m", digits: int = 3) -> str:
    return f"{m.value:.{digits}f} [{m.lo:.{digits}f}, {m.hi:.{digits}f}] {unit}"


def summary_markdown(plan: Plan) -> str:
    c = plan.capture
    lines = [
        f"# {c.id}: {c.tier} tier",
        "",
        f"* intervals: {int(plan.confidence_level * 100)}% two-sided; calibration `{plan.calibration.version}`"
        f" ({'fitted' if plan.calibration.fitted else 'prior, not yet fitted'})",
        f"* scale source: {plan.scale.source} (log-sigma {plan.scale.sigma_log:.3f})",
        f"* footprint: {ci(plan.stitched.footprint_area, 'm²', 2)}; bounding box "
        f"{plan.stitched.bounding_box_width.value:.2f} x {plan.stitched.bounding_box_depth.value:.2f} m",
        f"* drift: {plan.stitched.drift.method}",
        f"* runtime: {plan.runtime.total_s:.1f} s on {plan.runtime.machine}",
        "",
        "| room | floor area | ceiling height | walls | openings |",
        "|---|---|---|---|---|",
    ]
    for r in plan.rooms:
        lines.append(f"| {r.id} {r.name} | {ci(r.floor_area, 'm²', 2)} | {ci(r.ceiling_height)} | {len(r.walls)} | "
                     f"{len(r.openings)} |")
    for r in plan.rooms:
        lines += ["", f"## {r.id} {r.name}", "", "| wall | length | observed |", "|---|---|---|"]
        for w in r.walls:
            lines.append(f"| {w.id} | {ci(w.length)} | {w.observed_fraction:.0%} |")
        if r.openings:
            lines += ["", "| opening | kind | wall | width | height | to |", "|---|---|---|---|---|---|"]
            for o in r.openings:
                lines.append(f"| {o.id} | {o.kind} | {o.wall_id} | {ci(o.width)} | {ci(o.height)} | "
                             f"{o.connects_to_room_id or '-'} |")
    if plan.stitched.adjacency:
        lines += ["", "## Adjacency", ""]
        for a in plan.stitched.adjacency:
            lines.append(f"* {a.room_a} - {a.room_b}: {a.kind} {', '.join(a.via_opening_ids)} ({a.evidence})")
    if plan.damage_regions:
        lines += ["", "## Damage", "", "| id | surface | class | area | w x h |", "|---|---|---|---|---|"]
        for d in plan.damage_regions:
            lines.append(f"| {d.id} | {d.surface_id} | {d.damage_class} | {ci(d.area, 'm²')} | "
                         f"{d.width.value:.2f} x {d.height.value:.2f} m |")
    if plan.concealed_damage_flags:
        lines += ["", "## Concealed-damage flags", ""]
        for f in plan.concealed_damage_flags:
            lines.append(f"* **{f.rule_id}** on {', '.join(f.surface_ids)}: {f.suspected} (rule: {f.rule})")
    if plan.scope:
        lines += ["", "## Scope", "", "| id | code | surface | quantity | description |", "|---|---|---|---|---|"]
        for s in plan.scope:
            lines.append(f"| {s.id} | {s.code} | {s.surface_id} | {s.quantity_imperial:.1f} {s.unit_label} | "
                         f"{s.description} |")
    if plan.warnings:
        lines += ["", "## Warnings", ""] + [f"* {w}" for w in plan.warnings]
    return "\n".join(lines) + "\n"


def write_summary(plan: Plan, path: Path) -> Path:
    path.write_text(summary_markdown(plan), encoding="utf-8", newline="\n")
    return path
