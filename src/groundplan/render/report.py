"""Self-contained HTML report for one run (plan, measurements with intervals, damage, flags, scope)."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from groundplan.contract import Measurement, Plan

CSS = """
:root { --bg:#ffffff; --fg:#1b263b; --muted:#5c677d; --line:#d9dee7; --accent:#3d5a80; --warn:#9a3412; --chip:#eef2f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f141b; --fg:#e6ebf2; --muted:#9aa6b8;
  --line:#2a3340; --accent:#8fb3df; --warn:#fdba74; --chip:#1b2430; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
.meta { color:var(--muted); font-size:13px; } .chips span { background:var(--chip); border-radius:12px; padding:2px 10px; margin-right:6px; font-size:12px; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; } th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; } td.num { font-variant-numeric: tabular-nums; white-space:nowrap; }
.ci { color:var(--muted); } .warn { color:var(--warn); } img.plan { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }
.dmg { display:flex; gap:12px; flex-wrap:wrap; } .dmg figure { margin:0; width:180px; } .dmg img { width:180px; height:180px; object-fit:cover; border-radius:6px; border:1px solid var(--line); }
.dmg figcaption { font-size:12px; color:var(--muted); } .wrap { overflow-x:auto; }
"""


def _ci(m: Measurement | None, digits: int = 3, unit: str | None = None) -> str:
    if m is None:
        return "-"
    u = unit if unit is not None else {"m": " m", "m2": " m²", "ea": ""}.get(m.unit, "")
    return (f"<td class='num'>{m.value:.{digits}f}{u} <span class='ci'>[{m.lo:.{digits}f}, {m.hi:.{digits}f}]"
            f"{'' if m.calibrated else '*'}</span></td>")


def _img64(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def write_report(plan: Plan, out_dir: Path) -> Path:
    e = html.escape
    c = plan.capture
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>{e(c.id)} plan</title><style>{CSS}</style></head><body><main>",
             f"<h1>{e(c.id)}: {c.tier} tier</h1>",
             f"<div class='meta'>{len(plan.rooms)} rooms · footprint {plan.stitched.footprint_area.value:.1f} m² "
             f"[{plan.stitched.footprint_area.lo:.1f}, {plan.stitched.footprint_area.hi:.1f}] · "
             f"{int(plan.confidence_level * 100)}% intervals · calibration {e(plan.calibration.version)}"
             f"{'' if plan.calibration.fitted else ' (prior; * = not yet fitted on the benchmark)'} · "
             f"runtime {plan.runtime.total_s:.0f} s · groundplan {e(plan.runtime.groundplan_version)} "
             f"{e(plan.runtime.git_commit or '')}</div>",
             f"<div class='chips' style='margin-top:8px'><span>scale: {e(plan.scale.source)}</span>"
             f"<span>drift: {e(plan.stitched.drift.method)}</span></div>"]
    png = out_dir / "plan.png"
    if png.exists():
        parts.append(f"<h2>Stitched plan</h2><img class='plan' alt='stitched floor plan' src='{_img64(png)}'>")
    parts.append("<h2>Rooms</h2><div class='wrap'><table><tr><th>Room</th><th>Type</th><th>Floor area</th><th>Ceiling height</th>"
                 "<th>Perimeter</th><th>Walls</th><th>Openings</th><th>Seen</th></tr>")
    for r in plan.rooms:
        parts.append(f"<tr><td>{e(r.id)} {e(r.name)}</td><td>{e(r.type.label)}</td>{_ci(r.floor_area, 2)}"
                     f"{_ci(r.ceiling_height)}{_ci(r.perimeter, 2)}<td>{len(r.walls)}</td><td>{len(r.openings)}</td>"
                     f"<td>floor {r.coverage.floor_observed_fraction:.0%}, walls {r.coverage.walls_observed_fraction:.0%}"
                     f"{', ceiling' if r.coverage.ceiling_observed else ''}</td></tr>")
    parts.append("</table></div>")
    for r in plan.rooms:
        parts.append(f"<h2>{e(r.id)} {e(r.name)}</h2><div class='wrap'><table><tr><th>Wall</th><th>Length</th><th>Observed</th></tr>")
        for w in r.walls:
            parts.append(f"<tr><td>{e(w.id)}</td>{_ci(w.length)}<td>{w.observed_fraction:.0%}</td></tr>")
        parts.append("</table>")
        if r.openings:
            parts.append("<table><tr><th>Opening</th><th>Kind</th><th>Wall</th><th>Width</th><th>Height</th><th>To</th></tr>")
            for o in r.openings:
                parts.append(f"<tr><td>{e(o.id)}</td><td>{e(o.kind)}</td><td>{e(o.wall_id)}</td>{_ci(o.width)}{_ci(o.height)}"
                             f"<td>{e(o.connects_to_room_id or '-')}</td></tr>")
            parts.append("</table>")
        parts.append("</div>")
    if plan.stitched.adjacency:
        parts.append("<h2>Adjacency</h2><ul>")
        for a in plan.stitched.adjacency:
            parts.append(f"<li>{e(a.room_a)} – {e(a.room_b)}: {e(a.kind)} {e(', '.join(a.via_opening_ids))} "
                         f"<span class='ci'>({e(a.evidence)})</span></li>")
        parts.append("</ul>")
    parts.append("<h2>Damage</h2>")
    if plan.damage_regions:
        parts.append("<div class='dmg'>")
        for d in plan.damage_regions:
            src = _img64(out_dir / d.evidence_image) if d.evidence_image else ""
            parts.append(f"<figure>{f'<img alt={chr(39)}{e(d.damage_class)}{chr(39)} src={chr(39)}{src}{chr(39)}>' if src else ''}"
                         f"<figcaption><b>{e(d.id)}</b> {e(d.damage_class.replace('_', ' '))} on {e(d.surface_id)}<br>"
                         f"{d.width.value:.2f} × {d.height.value:.2f} m, {d.area.value:.3f} m² "
                         f"[{d.area.lo:.3f}, {d.area.hi:.3f}] · conf {d.confidence:.2f}</figcaption></figure>")
        parts.append("</div>")
    else:
        parts.append("<p class='meta'>No damage regions reported.</p>")
    if plan.concealed_damage_flags:
        parts.append("<h2>Concealed-damage flags</h2><div class='wrap'><table><tr><th>Flag</th><th>Rule</th><th>Surfaces</th>"
                     "<th>Suspected</th><th>Verify</th></tr>")
        for f in plan.concealed_damage_flags:
            parts.append(f"<tr><td>{e(f.id)} <span class='warn'>{e(f.severity)}</span></td><td><b>{e(f.rule_id)}</b> {e(f.rule)}"
                         f"<br><span class='ci'>{e(f.condition)}</span></td><td>{e(', '.join(f.surface_ids))}</td>"
                         f"<td>{e(f.suspected)}</td><td>{e(f.recommended_verification)}</td></tr>")
        parts.append("</table></div>")
    if plan.scope:
        parts.append("<h2>Scope</h2><div class='wrap'><table><tr><th>Item</th><th>Code</th><th>Surface</th><th>Quantity</th>"
                     "<th>Description</th><th>Derivation</th></tr>")
        for s in plan.scope:
            parts.append(f"<tr><td>{e(s.id)}</td><td>{e(s.code)}</td><td>{e(s.surface_id)}</td><td class='num'>"
                         f"{s.quantity_imperial:.1f} {s.unit_label} <span class='ci'>({s.quantity.value:.2f} "
                         f"[{s.quantity.lo:.2f}, {s.quantity.hi:.2f}] {e(s.quantity.unit)})</span></td>"
                         f"<td>{e(s.description)}</td><td class='ci'>{e(s.derivation)}</td></tr>")
        parts.append("</table></div>")
    if plan.warnings:
        parts.append("<h2>Warnings</h2><ul>" + "".join(f"<li class='warn'>{e(w)}</li>" for w in plan.warnings) + "</ul>")
    parts.append("<p class='meta' style='margin-top:32px'>Machine-readable output: plan.json (schema "
                 f"{e(plan.schema_version)}). Intervals are two-sided at {int(plan.confidence_level * 100)}%.</p>")
    parts.append("</main></body></html>")
    path = out_dir / "report.html"
    path.write_text("".join(parts), encoding="utf-8", newline="\n")
    return path
