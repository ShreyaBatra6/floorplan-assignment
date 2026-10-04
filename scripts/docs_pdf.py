"""Render a docs/*.md page to PDF (A4) with a headless Chromium browser, and report its page count.

    python scripts/docs_pdf.py docs/CAPTURE_PROTOCOL.md [--compact]

Supports the Markdown subset the docs use (headings, paragraphs, bullet/numbered lists, tables,
bold, italics, inline code). Used to check the one-page protocol and the six-page report limits.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
]


def inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    return t


def md_to_html(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1 : j])) + "</pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            head, body = rows[0], rows[1:]
            out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</table>")
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.", ln))
            items = []
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                item = re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    item += " " + lines[i].strip()
                    i += 1
                items.append(f"<li>{inline(item)}</li>")
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue
        para = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|\s*([-*]|\d+\.)\s)", lines[i]):
            para.append(lines[i])
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def css(compact: bool) -> str:
    base = 9.2 if compact else 10.5
    return f"""@page {{ size: A4; margin: {'10mm 11mm' if compact else '16mm 16mm'}; }}
body {{ font-family: Segoe UI, Helvetica, Arial, sans-serif; font-size: {base}pt; line-height: 1.32; color: #111; }}
h1 {{ font-size: {base + 5}pt; margin: 0 0 4pt; }} h2 {{ font-size: {base + 1.5}pt; margin: 7pt 0 3pt; border-bottom: 1px solid #bbb; }}
h3 {{ font-size: {base + 0.5}pt; margin: 6pt 0 2pt; }} p {{ margin: 3pt 0; }} ul, ol {{ margin: 2pt 0 2pt 15pt; padding: 0; }}
li {{ margin: 1pt 0; }} table {{ border-collapse: collapse; margin: 4pt 0; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 2pt 4pt; text-align: left; vertical-align: top; }} th {{ background: #f0f2f5; }}
code {{ font-family: Consolas, monospace; font-size: {base - 0.6}pt; }} pre {{ font-size: {base - 1.4}pt; background: #f6f7f9; padding: 4pt; white-space: pre-wrap; }}"""


def find_browser() -> str | None:
    for b in BROWSERS:
        if os.path.isabs(b) and Path(b).exists():
            return b
        if not os.path.isabs(b) and shutil.which(b):
            return shutil.which(b)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown")
    ap.add_argument("--compact", action="store_true")
    a = ap.parse_args()
    src = Path(a.markdown).resolve()
    html_path = src.with_suffix(".print.html")
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{css(a.compact)}</style></head>"
                         f"<body>{md_to_html(src.read_text(encoding='utf-8'))}</body></html>", encoding="utf-8")
    pdf = src.with_suffix(".pdf")
    browser = find_browser()
    if browser is None:
        print("no Chromium-based browser found; open the .print.html and print to PDF")
        return 1
    import tempfile

    profile = tempfile.mkdtemp(prefix="docs_pdf_")
    pdf.unlink(missing_ok=True)  # otherwise the wait below finds the previous PDF and reports on it
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html_path.as_uri()],
                   capture_output=True, timeout=180)
    import time

    for _ in range(120):  # the launcher returns before its renderer process has written the file
        if pdf.exists() and pdf.stat().st_size > 0:
            time.sleep(1.0)
            break
        time.sleep(0.5)
    shutil.rmtree(profile, ignore_errors=True)
    if not pdf.exists():
        print(f"browser did not write {pdf}; open {html_path} and print to PDF")
        return 1
    html_path.unlink(missing_ok=True)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    print(f"{pdf} : {pages} page(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
