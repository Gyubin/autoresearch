"""Deterministic report figures as stdlib SVG (Phase 5).

Built directly from immutable log artifacts (the ledger + the bootstrap
result), never recomputed. Byte-deterministic: sorted iteration, a fixed
palette, a fixed viewport, and no timestamps in the SVG body — so a figure is
reproducible from the logs and drillable by hash.

build_figures(records, boot) -> {filename: svg_text}
  dev_trajectory.svg  — incumbent dev score per generation (carry-forward)
  test_paired_rmse.svg — per-seed baseline->incumbent dumbbell + pooled effect
  verdict_mix.svg     — verdict composition per generation (stacked bars)
"""

from __future__ import annotations

from .svgfig import Svg, linear_scale

_W, _H = 640, 360
_ML, _MR, _MT, _MB = 70, 30, 40, 50  # margins

_VERDICT_ORDER = ("valid_positive", "valid_inconclusive", "valid_negative",
                  "invalid_implementation", "contract_violation", "pruned",
                  "aborted")
_VERDICT_COLOR = {
    "valid_positive": "#2e7d32", "valid_inconclusive": "#9e9e9e",
    "valid_negative": "#c62828", "invalid_implementation": "#6a1b9a",
    "contract_violation": "#000000", "pruned": "#ef6c00", "aborted": "#455a64",
}


def _axes(svg: Svg, title: str, xlabel: str, ylabel: str) -> tuple:
    x0, x1 = _ML, _W - _MR
    y0, y1 = _H - _MB, _MT
    svg.text(_ML, _MT - 18, title, size=15, weight="bold")
    svg.line(x0, y0, x1, y0, stroke="#444", stroke_width=1.2)   # x axis
    svg.line(x0, y0, x0, y1, stroke="#444", stroke_width=1.2)   # y axis
    svg.text((x0 + x1) / 2, _H - 14, xlabel, size=12, anchor="middle")
    svg.text(16, (y0 + y1) / 2, ylabel, size=12, anchor="middle")
    return x0, x1, y0, y1


def _baseline_primary(records: list[dict]) -> float | None:
    for r in records:
        if r.get("record_type") == "baseline":
            v = r.get("primary")
            return v if isinstance(v, (int, float)) else None
    return None


def _experiments(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("record_type") == "experiment"]


def _dev_trajectory(records: list[dict]) -> str:
    svg = Svg(_W, _H)
    x0, x1, y0, y1 = _axes(svg, "Dev incumbent trajectory",
                           "generation", "mean_tour_length (dev)")
    base = _baseline_primary(records)
    exps = _experiments(records)
    gens = sorted({g for r in exps
                   if isinstance((g := r.get("generation")), int)})
    # Carry-forward best (minimize) dev score of the incumbent per generation.
    series: list[tuple[int, float]] = []
    best = base
    if base is not None:
        series.append((0, base))
    for g in gens:
        for r in exps:
            if (r.get("generation") == g and r.get("decision") == "accept"
                    and isinstance(r.get("primary"), (int, float))):
                if best is None or r["primary"] < best:
                    best = r["primary"]
        if best is not None:
            series.append((g, best))
    if len(series) < 1:
        svg.text((x0 + x1) / 2, (y0 + y1) / 2, "no dev trajectory",
                 anchor="middle", fill="#999")
        return svg.render("dev trajectory")
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    sx = linear_scale(min(xs), max(xs) if max(xs) != min(xs) else min(xs) + 1,
                      x0, x1)
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.1 or 0.01
    sy = linear_scale(lo - pad, hi + pad, y0, y1)
    pts = [(sx(x), sy(y)) for x, y in series]
    svg.polyline(pts, stroke="#1565c0", stroke_width=2.0)
    for (x, y), r in zip(pts, series):
        svg.circle(x, y, 3.5, fill="#1565c0")
    return svg.render("dev trajectory")


def _test_paired(boot) -> str:
    svg = Svg(_W, _H)
    x0, x1, y0, y1 = _axes(svg, "Per-seed test tour length (baseline -> incumbent)",
                           "test seed", "mean_tour_length (test)")
    seeds = list(boot.per_seed)
    vals = [v for s in seeds for v in (s.rmse_baseline, s.rmse_incumbent)
            if isinstance(v, (int, float))]
    if not vals:
        svg.text((x0 + x1) / 2, (y0 + y1) / 2, "no clean test runs",
                 anchor="middle", fill="#999")
        return svg.render("test paired rmse")
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.15 or 0.01
    sy = linear_scale(lo - pad, hi + pad, y0, y1)
    n = len(seeds)
    step = (x1 - x0) / (n + 1)
    for i, s in enumerate(seeds):
        cx = x0 + step * (i + 1)
        if isinstance(s.rmse_baseline, (int, float)) and isinstance(
                s.rmse_incumbent, (int, float)):
            yb, yi = sy(s.rmse_baseline), sy(s.rmse_incumbent)
            svg.line(cx, yb, cx, yi, stroke="#bbb", stroke_width=2.0)
            svg.circle(cx, yb, 4.0, fill="#c62828")   # baseline
            svg.circle(cx, yi, 4.0, fill="#2e7d32")   # incumbent
        svg.text(cx, y0 + 16, f"s{s.seed_index}", size=11, anchor="middle")
    # legend
    svg.circle(x1 - 96, _MT - 6, 4.0, fill="#c62828")
    svg.text(x1 - 88, _MT - 2, "baseline", size=11)
    svg.circle(x1 - 96, _MT + 10, 4.0, fill="#2e7d32")
    svg.text(x1 - 88, _MT + 14, "incumbent", size=11)
    return svg.render("test paired rmse")


def _verdict_mix(records: list[dict]) -> str:
    svg = Svg(_W, _H)
    x0, x1, y0, y1 = _axes(svg, "Verdict composition per generation",
                           "generation", "count")
    exps = _experiments(records)
    gens = sorted({g for r in exps
                   if isinstance((g := r.get("generation")), int)})
    if not gens:
        svg.text((x0 + x1) / 2, (y0 + y1) / 2, "no experiments",
                 anchor="middle", fill="#999")
        return svg.render("verdict mix")
    counts: dict[int, dict[str, int]] = {g: {} for g in gens}
    for r in exps:
        g = r.get("generation")
        if g in counts:
            v = str(r.get("verdict"))
            counts[g][v] = counts[g].get(v, 0) + 1
    max_total = max((sum(c.values()) for c in counts.values()), default=1) or 1
    sy = linear_scale(0, max_total, y0, y1)
    step = (x1 - x0) / (len(gens) + 1)
    bar_w = min(40.0, step * 0.6)
    for i, g in enumerate(gens):
        cx = x0 + step * (i + 1)
        stack = 0
        for verdict in _VERDICT_ORDER:
            c = counts[g].get(verdict, 0)
            if c == 0:
                continue
            y_top = sy(stack + c)
            y_bot = sy(stack)
            svg.rect(cx - bar_w / 2, y_top, bar_w, y_bot - y_top,
                     fill=_VERDICT_COLOR[verdict])
            stack += c
        svg.text(cx, y0 + 16, f"g{g}", size=11, anchor="middle")
    return svg.render("verdict mix")


def build_figures(records: list[dict], boot) -> dict[str, str]:
    return {
        "dev_trajectory.svg": _dev_trajectory(records),
        "test_paired_rmse.svg": _test_paired(boot),
        "verdict_mix.svg": _verdict_mix(records),
    }
