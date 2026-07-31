"""Escaped, dependency-free HTML template for experiment reports."""

from __future__ import annotations

import html


def render_html(title: str, summary: dict, figures: list[str], warnings: list[str]) -> str:
    """Render report content while escaping all metric and warning text."""
    rows = "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in summary.items())
    images = "".join(f'<figure><img src="{html.escape(path)}" alt="{html.escape(path)}"><figcaption>{html.escape(path)}</figcaption></figure>' for path in figures if path.endswith(".png"))
    warning_list = "".join(f"<li>{html.escape(item)}</li>" for item in warnings) or "<li>None</li>"
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font-family:Arial;margin:2rem}}img{{max-width:900px}}th{{text-align:left;padding-right:1rem}}</style></head><body><h1>{html.escape(title)}</h1><h2>Summary</h2><table>{rows}</table><h2>Warnings</h2><ul>{warning_list}</ul><h2>Figures</h2>{images}</body></html>"
