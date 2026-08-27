"""Draw the chart of all 32 DX7 algorithms as a standalone SVG.

The topology is not hand-drawn: it is decoded from the operator routing table
in ``src/msfa/fm_core.cc`` (the same table Dexed feeds to the FM core), so the
picture cannot drift away from what the synth actually renders.

Layout convention: signal flows downward, carriers sit on the bottom row, and
operators are ordered left to right by ascending operator number. Each panel is
only as wide as its algorithm needs, and the panels flow into balanced rows.

Licensing: the chart this produces -- docs/dx7_algorithms.svg -- is released
under the MIT License, so it can be reused outside a GPL context. It contains no
code: the topology it draws is the DX7's own, and the layout and styling are
original to this project. The generator below, like the rest of dexed-py, is
GPL-3.0-or-later.

Usage:
    python docs/make_algorithm_chart.py [-o docs/dx7_algorithms.svg]
    python docs/make_algorithm_chart.py --png docs/dx7_algorithms.png

Or ``make chart``. The PNG needs ``pip install cairosvg``; the SVG is what the
docs reference, and it is the file the test suite checks for staleness.
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. Decode the routing table
# ---------------------------------------------------------------------------

OUT_BUS_ADD, FB_IN, FB_OUT = 0x04, 0x40, 0x80


def parse_algorithms(source: Path) -> list[dict]:
    """Decode FmCore::algorithms[32] into modulation graphs.

    Each row is six bytes, one per rendering slot. Slot i is DX7 operator 6-i
    (the core renders operator 6 first). A byte holds: input bus in bits 4-5,
    output bus in bits 0-1, an "add to bus" flag, and the two feedback-tap bits.
    Writing to bus 0 means writing to the audio output, i.e. being a carrier.
    """
    text = source.read_text(encoding="utf-8")
    body = text.split("algorithms[32] = {")[1].split("};")[0]
    rows = re.findall(r"\{\s*\{([^}]*)\}\s*\}", body)
    if len(rows) != 32:
        raise RuntimeError(f"expected 32 algorithms, found {len(rows)}")

    out = []
    for number, row in enumerate(rows, start=1):
        flags = [int(x, 16) for x in row.split(",")]
        bus: dict[int, list[int]] = {1: [], 2: []}
        edges: list[tuple[int, int]] = []
        carriers: list[int] = []
        fb_in = fb_out = None

        for slot, f in enumerate(flags):
            op = 6 - slot
            in_bus, out_bus, add = (f >> 4) & 3, f & 3, bool(f & OUT_BUS_ADD)
            if f & FB_IN:
                fb_in = op
            if f & FB_OUT:
                fb_out = op
            # Everything sitting in the input bus modulates this operator.
            edges.extend((m, op) for m in bus[in_bus]) if in_bus else None
            if out_bus == 0:
                carriers.append(op)
            else:
                bus[out_bus] = (bus[out_bus] if add else []) + [op]

        out.append(
            {
                "number": number,
                "edges": sorted(edges),
                "carriers": sorted(carriers),
                # fb_out taps the loop, fb_in receives it; equal for the usual
                # single-operator feedback, spread apart in algorithms 4 and 6.
                "fb_in": fb_in,
                "fb_out": fb_out,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 2. Lay each graph out on a unit grid
# ---------------------------------------------------------------------------


def _isotonic(values: list[float]) -> list[float]:
    """Pool-adjacent-violators: nearest non-decreasing sequence, least squares."""
    stack: list[tuple[float, int]] = []  # (mean, weight)
    for v in values:
        total, weight = v, 1
        while stack and stack[-1][0] >= total / weight:
            mean, w = stack.pop()
            total += mean * w
            weight += w
        stack.append((total / weight, weight))
    out: list[float] = []
    for mean, weight in stack:
        out.extend([mean] * weight)
    return out


def _pack(order: list[int], desired: dict[int, float]) -> dict[int, float]:
    """Place `order` left to right, one column apart minimum, as near `desired` as possible."""
    shifted = [desired[op] - i for i, op in enumerate(order)]
    return {op: y + i for i, (op, y) in enumerate(zip(order, _isotonic(shifted)))}


def _levels(alg: dict) -> tuple[dict[int, int], dict[int, list[int]], dict[int, list[int]]]:
    targets: dict[int, list[int]] = {op: [] for op in range(1, 7)}
    parents: dict[int, list[int]] = {op: [] for op in range(1, 7)}
    for mod, car in alg["edges"]:
        targets[mod].append(car)
        parents[car].append(mod)

    # Longest path to a carrier, so a modulator always sits above everything it
    # feeds, even when it feeds two chains of different depth.
    level: dict[int, int] = {}

    def depth(op: int) -> int:
        if op not in level:
            level[op] = 1 + max(depth(t) for t in targets[op]) if targets[op] else 0
        return level[op]

    for op in range(1, 7):
        depth(op)
    return level, targets, parents


def _cost(x: dict[int, float], targets: dict[int, list[int]],
          parents: dict[int, list[int]]) -> float:
    """Lower is better: narrow, with every operator centred on its neighbours."""
    width = max(x.values()) - min(x.values())
    centring = 0.0
    for op in range(1, 7):
        for near in (targets[op], parents[op]):
            if near:
                centring += abs(x[op] - sum(x[n] for n in near) / len(near))
    stretch = sum(abs(x[m] - x[c]) for m in range(1, 7) for c in targets[m])
    return 6 * width + 2 * centring + stretch


def layout(alg: dict) -> dict[int, tuple[float, int]]:
    """Return {operator: (x, level)}; level 0 is the carrier row at the bottom.

    Six operators is small enough to place exactly: try every way of dealing
    columns out to the operators on each row and keep the tidiest, rather than
    relaxing toward a local minimum.
    """
    level, targets, parents = _levels(alg)
    rows: dict[int, list[int]] = {}
    for op in range(1, 7):
        rows.setdefault(level[op], []).append(op)
    for row in rows.values():
        row.sort()

    # Operators read left to right in ascending order within a row, so each row
    # is a choice of columns rather than a permutation of them.
    order = sorted(rows)
    choices = [list(combinations(range(6), len(rows[lvl]))) for lvl in order]

    best: dict[int, float] | None = None
    best_cost = float("inf")
    for combo in product(*choices):
        x = {op: float(col) for lvl, cols in zip(order, combo)
             for op, col in zip(rows[lvl], cols)}
        c = _cost(x, targets, parents)
        if c < best_cost:
            best_cost, best = c, x
    assert best is not None

    # Integer columns can only get a fan-out half right: an operator feeding two
    # carriers wants to sit between them, half a column off the grid. Sweep once
    # up (centre each operator over what it feeds) and once down (centre each
    # over what feeds it). Two fixed sweeps, so unlike a relaxation this cannot
    # wander off.
    x = best
    for sweep, near_of in ((order[1:], targets), (order[::-1], parents)):
        for lvl in sweep:
            want = {}
            for op in rows[lvl]:
                near = near_of[op]
                want[op] = sum(x[n] for n in near) / len(near) if near else x[op]
            x.update(_pack(rows[lvl], want))

    left = min(x.values())
    return {op: (round(x[op] - left, 4), level[op]) for op in x}


# ---------------------------------------------------------------------------
# 3. Draw
# ---------------------------------------------------------------------------

BOX_W, BOX_H = 32.0, 25.0
COL, ROW = 46.0, 44.0        # unit grid pitch inside a panel
MARGIN_X, MARGIN_TOP = 30.0, 78.0
PAD_TOP, PAD_BOTTOM = 16.0, 4.0
PANEL_PAD = 18.0             # breathing room either side of a panel's drawing
MIN_CELL_W = 132.0           # so the narrow chains still read as panels
LABEL_H = 26.0               # strip under each panel holding its number
FB_OUT_X, FB_UP = 9.0, 13.0  # reach of the feedback loop, sideways and up
CLEARANCE = 3.0              # keep the feedback loop this far off other wiring


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _overlap(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> bool:
    """Do two axis-aligned segments come within CLEARANCE of each other?"""
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    return (ax0 - CLEARANCE < bx1 and bx0 - CLEARANCE < ax1
            and ay0 - CLEARANCE < by1 and by0 - CLEARANCE < ay1)


def _path_segments(points: list[tuple[float, float]]) -> list[tuple[float, float, float, float]]:
    return [(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
            for i in range(len(points) - 1)]


def _svg_path(points: list[tuple[float, float]]) -> str:
    head = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
    rest = []
    for (px, py), (x, y) in zip(points, points[1:]):
        rest.append(f"V {y:.1f}" if abs(x - px) < 1e-6 else f"H {x:.1f}")
    return head + " " + " ".join(rest)


def panel_geometry(alg: dict) -> dict:
    """Everything drawn inside one panel, in local coordinates.

    x grows rightwards from the leftmost box column; y grows downwards with the
    carrier row's top edge at 0, so upper levels are negative. Measuring the
    drawing rather than assuming its extent lets the feedback loop take
    whichever route is clear without the panel clipping it.
    """
    pos = layout(alg)

    def box_xy(op: int) -> tuple[float, float]:
        x, lvl = pos[op]
        return x * COL, -lvl * ROW

    wires: list[tuple[float, float, float, float]] = []
    arrows: list[tuple[float, float, float, float]] = []

    by_target: dict[int, list[int]] = {}
    for mod, car in alg["edges"]:
        by_target.setdefault(car, []).append(mod)

    for car, mods in sorted(by_target.items()):
        cx, cy = box_xy(car)
        cx += BOX_W / 2
        rail = cy - (ROW - BOX_H) / 2
        xs = []
        for mod in mods:
            mx, my = box_xy(mod)
            mx += BOX_W / 2
            xs.append(mx)
            wires.append((mx, my + BOX_H, mx, rail))
        lo, hi = min(xs + [cx]), max(xs + [cx])
        if hi - lo > 0.5:
            wires.append((lo, rail, hi, rail))
        arrows.append((cx, rail, cx, cy))

    feedback = _route_feedback(alg, box_xy, wires + arrows, pos)

    xs = [box_xy(op)[0] for op in pos] + [box_xy(op)[0] + BOX_W for op in pos]
    for seg in wires + arrows + (feedback["segments"] if feedback else []):
        xs += [seg[0], seg[2]]

    return {
        "pos": pos,
        "box_xy": box_xy,
        "wires": wires,
        "arrows": arrows,
        "feedback": feedback,
        "min_x": min(xs),
        "max_x": max(xs),
        "depth": max(l for _, l in pos.values()),
    }


def _route_feedback(alg, box_xy, obstacles, pos):
    """Pick a feedback route that clears the modulation wiring.

    Both shapes end by pointing down into the middle of the operator that
    receives the feedback, so they read the same way as a modulation arrow --
    and so algorithms 4 and 6, whose loop wraps a whole chain, do not stop at
    the side of the box.
    """
    fi, fo = alg["fb_in"], alg["fb_out"]
    if fi is None or fo is None:
        return None

    ix, iy = box_xy(fi)
    fx, fy = box_xy(fo)
    centre = ix + BOX_W / 2

    def variants():
        if fi == fo:
            # Out of the side, up over the box, back down into its top.
            for side in (1, -1):
                edge = fx + BOX_W if side > 0 else fx
                lane = edge + side * FB_OUT_X
                yield [(edge, fy + BOX_H * 0.62), (lane, fy + BOX_H * 0.62),
                       (lane, fy - FB_UP), (centre, fy - FB_UP), (centre, fy)]
        else:
            # The loop wraps a chain: run up the outside, then down into the top.
            for side in (1, -1):
                span = [box_xy(op)[0] for op in pos]
                edge = fx + BOX_W if side > 0 else fx
                lane = (max(span) + BOX_W + FB_OUT_X if side > 0
                        else min(span) - FB_OUT_X)
                yield [(edge, fy + BOX_H / 2), (lane, fy + BOX_H / 2),
                       (lane, iy - FB_UP), (centre, iy - FB_UP), (centre, iy)]

    fallback = None
    for points in variants():
        segments = _path_segments(points)
        if fallback is None:
            fallback = {"points": points, "segments": segments}
        if not any(_overlap(s, o) for s in segments for o in obstacles):
            return {"points": points, "segments": segments}
    return fallback


def cell_size(alg: dict) -> tuple[float, float, int]:
    """Natural (width, drawing height, depth) of one algorithm's panel.

    Algorithms differ a lot: a single chain is 87 units wide, six parallel
    carriers 271. Measuring each one lets the sheet give every algorithm the
    room it actually needs instead of the widest one's.
    """
    g = panel_geometry(alg)
    return (max(MIN_CELL_W, g["max_x"] - g["min_x"] + 2 * PANEL_PAD),
            g["depth"] * ROW + BOX_H, g["depth"])


def _partition(widths: list[float], rows: int) -> list[list[int]]:
    """Split the sequence into `rows` contiguous groups, minimising the widest.

    Keeps the algorithms in order -- the numbering is the point of the chart --
    while letting each row hold as many as its algorithms happen to need.
    """
    n = len(widths)
    if rows >= n:
        return [[i] for i in range(n)]
    prefix = [0.0]
    for w in widths:
        prefix.append(prefix[-1] + w)

    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(rows + 1)]
    cut = [[0] * (n + 1) for _ in range(rows + 1)]
    best[0][0] = 0.0
    for r in range(1, rows + 1):
        for i in range(r, n + 1):
            for j in range(r - 1, i):
                widest = max(best[r - 1][j], prefix[i] - prefix[j])
                if widest < best[r][i]:
                    best[r][i], cut[r][i] = widest, j

    groups, i = [], n
    for r in range(rows, 0, -1):
        j = cut[r][i]
        groups.append(list(range(j, i)))
        i = j
    return groups[::-1]


def draw_cell(alg: dict, ox: float, oy: float, cell_w: float, cell_h: float) -> list[str]:
    g = panel_geometry(alg)
    carriers = set(alg["carriers"])

    span_w = g["max_x"] - g["min_x"]
    # Centre horizontally; sit the carrier row on a common baseline so every
    # panel in the row lines up along the bottom.
    x0 = ox + (cell_w - span_w) / 2 - g["min_x"]
    y0 = oy + cell_h - LABEL_H - PAD_BOTTOM - BOX_H

    def place(points):
        return [(x0 + px, y0 + py) for px, py in points]

    out: list[str] = [
        f'<rect class="panel" x="{ox + 3:.1f}" y="{oy:.1f}" '
        f'width="{cell_w - 6:.1f}" height="{cell_h - 4:.1f}" rx="8"/>'
    ]

    for x1, y1, x2, y2 in g["wires"]:
        out.append(f'<path class="wire" d="{_svg_path(place([(x1, y1), (x2, y2)]))}"/>')
    for x1, y1, x2, y2 in g["arrows"]:
        out.append(f'<path class="wire arrow" d="{_svg_path(place([(x1, y1), (x2, y2)]))}"/>')

    if g["feedback"]:
        out.append(
            f'<path class="wire fb arrow" d="{_svg_path(place(g["feedback"]["points"]))}"/>'
        )

    # Boxes last, so the wires tuck underneath them.
    for op in sorted(g["pos"]):
        px, py = g["box_xy"](op)
        bx, by = x0 + px, y0 + py
        kind = "carrier" if op in carriers else "modulator"
        out.append(
            f'<rect class="op {kind}" x="{bx:.1f}" y="{by:.1f}" '
            f'width="{BOX_W}" height="{BOX_H}" rx="5"/>'
        )
        out.append(
            f'<text class="op-num {kind}" x="{bx + BOX_W / 2:.1f}" '
            f'y="{by + BOX_H / 2:.1f}">{op}</text>'
        )

    out.append(
        f'<text class="alg-num" x="{ox + cell_w / 2:.1f}" '
        f'y="{oy + cell_h - 12:.1f}">{alg["number"]}</text>'
    )
    return out


def build_svg(algs: list[dict], title: str, credit: str, rows: int = 5) -> str:
    sizes = [cell_size(a) for a in algs]
    natural = [w for w, _, _ in sizes]
    groups = _partition(natural, rows)

    # Every row justifies to the widest row's natural total, so the sheet keeps
    # a flush right edge while the panels inside it stay individually sized.
    content_w = max(sum(natural[i] for i in g) for g in groups)
    widths: list[float] = list(natural)
    for g in groups:
        slack = (content_w - sum(natural[i] for i in g)) / len(g)
        for i in g:
            widths[i] += slack

    heights = [
        max(sizes[i][1] for i in g) + PAD_TOP + PAD_BOTTOM + LABEL_H for g in groups
    ]
    tops = [MARGIN_TOP + sum(heights[:r]) for r in range(len(groups))]

    width = MARGIN_X * 2 + content_w
    legend_h = 62.0
    height = tops[-1] + heights[-1] + legend_h

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- Generated by docs/make_algorithm_chart.py from src/msfa/fm_core.cc."
        " Do not edit by hand; run `make chart`. -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
        "<style>",
        "  .bg { fill: #ffffff; }",
        "  .title { font-size: 30px; font-weight: 700; fill: #14161a; }",
        "  .credit { font-size: 13px; fill: #868d98; }",
        "  .alg-num { font-size: 15px; font-weight: 700; fill: #6b7280;"
        "    text-anchor: middle; }",
        "  .wire { fill: none; stroke: #9aa2ad; stroke-width: 1.6;"
        "    stroke-linecap: round; stroke-linejoin: round; }",
        "  .arrow { marker-end: url(#tip); }",
        "  .fb { stroke: #c2703a; marker-end: url(#tip-fb); }",
        "  .op { stroke-width: 1.8; }",
        "  .op.modulator { fill: #ffffff; stroke: #4b5563; }",
        "  .op.carrier { fill: #1f3a5f; stroke: #1f3a5f; }",
        "  .op-num { font-size: 15px; font-weight: 600; text-anchor: middle;"
        "    dominant-baseline: central; }",
        "  .op-num.modulator { fill: #1f2937; }",
        "  .op-num.carrier { fill: #ffffff; }",
        "  .legend { font-size: 13px; fill: #3f4652; }",
        "  .panel { fill: #f6f7f9; }",
        "</style>",
        "<defs>",
        '  <marker id="tip" viewBox="0 0 8 8" refX="6.4" refY="4" markerWidth="5"'
        ' markerHeight="5" orient="auto-start-reverse">'
        '<path d="M 0 0.8 L 7 4 L 0 7.2 z" fill="#9aa2ad"/></marker>',
        '  <marker id="tip-fb" viewBox="0 0 8 8" refX="6.4" refY="4" markerWidth="5"'
        ' markerHeight="5" orient="auto-start-reverse">'
        '<path d="M 0 0.8 L 7 4 L 0 7.2 z" fill="#c2703a"/></marker>',
        "</defs>",
        f'<rect class="bg" width="{width:.0f}" height="{height:.0f}"/>',
        f'<text class="title" x="{MARGIN_X:.0f}" y="46">{esc(title)}</text>',
    ]

    for g, top, cell_h in zip(groups, tops, heights):
        ox = MARGIN_X
        for i in g:
            parts.extend(draw_cell(algs[i], ox, top, widths[i], cell_h))
            ox += widths[i]

    # Legend
    ly = tops[-1] + heights[-1] + 26
    lx = MARGIN_X
    swatch = 22.0
    for kind in ("carrier", "modulator"):
        parts += [
            f'<rect class="op {kind}" x="{lx:.1f}" y="{ly - swatch / 2:.1f}" '
            f'width="{swatch * 1.3:.1f}" height="{swatch:.1f}" rx="4"/>',
            f'<text class="legend" x="{lx + swatch * 1.3 + 9:.1f}" y="{ly + 4.5:.1f}">'
            f"{kind}</text>",
        ]
        lx += swatch * 1.3 + 9 + 8.5 * len(kind) + 26
    parts += [
        f'<path class="wire fb arrow" d="M {lx:.1f} {ly + 6:.1f} H {lx + 16:.1f} '
        f'V {ly - 6:.1f} H {lx:.1f}"/>',
        f'<text class="legend" x="{lx + 34:.1f}" y="{ly + 4.5:.1f}">feedback</text>',
    ]
    parts.append(
        f'<text class="credit" x="{width - MARGIN_X:.1f}" y="{ly + 4.5:.1f}" '
        f'text-anchor="end">{esc(credit)}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_chart(source: Path | None = None, rows: int = 5) -> str:
    """The finished SVG as text. Shared by main() and by docs/conf.py."""
    algs = parse_algorithms(source or ROOT / "src" / "msfa" / "fm_core.cc")
    return build_svg(
        algs,
        title="The 32 DX7 algorithms",
        credit="Diagram MIT \u00b7 Code GPL-3.0 \u00b7 github.com/DBraun/dexed-py",
        rows=rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=str(ROOT / "docs" / "dx7_algorithms.svg"))
    ap.add_argument("--source", default=str(ROOT / "src" / "msfa" / "fm_core.cc"))
    ap.add_argument(
        "--png",
        metavar="PATH",
        help="also rasterize to PATH at 2x (requires cairosvg)",
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=5,
        help="how many rows to flow the 32 panels into (default: 5)",
    )
    args = ap.parse_args()

    svg = render_chart(Path(args.source), args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out} ({len(svg) / 1024:.1f} KB)")

    if args.png:
        try:
            import cairosvg
        except ImportError:
            raise SystemExit(
                "--png needs cairosvg: pip install cairosvg\n"
                "(the SVG above was still written)"
            )
        png = Path(args.png)
        png.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), scale=2)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
