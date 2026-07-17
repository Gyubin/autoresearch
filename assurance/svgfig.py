"""Minimal dependency-free SVG builder for Phase 5 report figures.

stdlib only (no matplotlib). Everything is byte-deterministic: coordinates are
formatted with a single fixed format, colors come from constants, and no
timestamps/ids are emitted into the SVG body. Given the same data a figure
renders to identical bytes, so figures are drillable by hash.
"""

from __future__ import annotations

from xml.sax.saxutils import escape


def _f(v: float) -> str:
    """Fixed 2-decimal coordinate format (stable across machines)."""
    return f"{v:.2f}"


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._parts: list[str] = []

    def rect(self, x: float, y: float, w: float, h: float, *,
             fill: str = "none", stroke: str = "none",
             stroke_width: float = 1.0) -> None:
        self._parts.append(
            f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{_f(stroke_width)}"/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, *,
             stroke: str = "#888", stroke_width: float = 1.0,
             dash: str | None = None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self._parts.append(
            f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" '
            f'stroke="{stroke}" stroke-width="{_f(stroke_width)}"{d}/>')

    def circle(self, cx: float, cy: float, r: float, *, fill: str = "#000",
               stroke: str = "none", stroke_width: float = 1.0) -> None:
        self._parts.append(
            f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{_f(stroke_width)}"/>')

    def polyline(self, points: list[tuple[float, float]], *,
                 stroke: str = "#000", stroke_width: float = 1.5,
                 fill: str = "none") -> None:
        pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
        self._parts.append(
            f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{_f(stroke_width)}"/>')

    def text(self, x: float, y: float, s: str, *, size: float = 12.0,
             fill: str = "#222", anchor: str = "start",
             weight: str = "normal") -> None:
        self._parts.append(
            f'<text x="{_f(x)}" y="{_f(y)}" font-family="sans-serif" '
            f'font-size="{_f(size)}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(s)}</text>')

    def render(self, title: str) -> str:
        body = "\n  ".join(self._parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" '
            f'role="img" aria-label="{escape(title)}">\n'
            f'  <rect x="0" y="0" width="{self.width}" height="{self.height}" '
            f'fill="#ffffff"/>\n  {body}\n</svg>\n')


def linear_scale(dmin: float, dmax: float, pmin: float, pmax: float):
    """Map data range [dmin, dmax] to pixel range [pmin, pmax]."""
    span = dmax - dmin
    if span == 0:
        span = 1.0

    def scale(v: float) -> float:
        return pmin + (v - dmin) / span * (pmax - pmin)

    return scale
