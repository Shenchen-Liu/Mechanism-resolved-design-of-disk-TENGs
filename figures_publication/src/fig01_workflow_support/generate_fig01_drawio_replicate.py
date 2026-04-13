#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent


def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
      return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def attrs(**kwargs: object) -> str:
    parts = []
    for key, value in kwargs.items():
        if value is None:
            continue
        name = key.rstrip("_").replace("_", "-")
        parts.append(f'{name}="{escape(str(value))}"')
    return " ".join(parts)


def tag(name: str, content: str = "", **kwargs: object) -> str:
    if content == "":
        return f"<{name} {attrs(**kwargs)}/>"
    return f"<{name} {attrs(**kwargs)}>{content}</{name}>"


def svg_wrap(width: int, height: int, defs: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f"{defs}{body}</svg>\n"
    )


def text_line(x: float, y: float, text: str, size: int = 16, weight: int = 400,
              anchor: str = "start", fill: str = "#111827", style: str = "",
              family: str = "Arial, Helvetica, sans-serif") -> str:
    extra = style if style else ""
    return (
        f'<text x="{fmt(x)}" y="{fmt(y)}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" style="{escape(extra)}">{escape(text)}</text>'
    )


def multiline_text(x: float, y: float, lines: Sequence[str], size: int = 16,
                   weight: int = 400, line_gap: float = 1.25,
                   anchor: str = "start", fill: str = "#111827",
                   italic: bool = False,
                   family: str = "Arial, Helvetica, sans-serif") -> str:
    parts = [
        f'<text x="{fmt(x)}" y="{fmt(y)}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" font-style="{"italic" if italic else "normal"}">'
    ]
    first = True
    for line in lines:
        if first:
            parts.append(f"<tspan x=\"{fmt(x)}\">{escape(line)}</tspan>")
            first = False
        else:
            dy = size * line_gap
            parts.append(f"<tspan x=\"{fmt(x)}\" dy=\"{fmt(dy)}\">{escape(line)}</tspan>")
    parts.append("</text>")
    return "".join(parts)


def ellipse_point(cx: float, cy: float, rx: float, ry: float, angle: float) -> tuple[float, float]:
    return cx + rx * math.cos(angle), cy + ry * math.sin(angle)


def annular_sector_path(cx: float, cy: float, rox: float, roy: float,
                        rix: float, riy: float, a0: float, a1: float) -> str:
    x0, y0 = ellipse_point(cx, cy, rox, roy, a0)
    x1, y1 = ellipse_point(cx, cy, rox, roy, a1)
    xi1, yi1 = ellipse_point(cx, cy, rix, riy, a1)
    xi0, yi0 = ellipse_point(cx, cy, rix, riy, a0)
    large = "1" if (a1 - a0) % (math.tau) > math.pi else "0"
    return (
        f"M {fmt(x0)} {fmt(y0)} "
        f"A {fmt(rox)} {fmt(roy)} 0 {large} 1 {fmt(x1)} {fmt(y1)} "
        f"L {fmt(xi1)} {fmt(yi1)} "
        f"A {fmt(rix)} {fmt(riy)} 0 {large} 0 {fmt(xi0)} {fmt(yi0)} Z"
    )


def front_band_path(cx: float, top_y: float, rx: float, ry: float, height: float) -> str:
    left_top = (cx - rx, top_y)
    right_top = (cx + rx, top_y)
    left_bottom = (cx - rx, top_y + height)
    right_bottom = (cx + rx, top_y + height)
    return (
        f"M {fmt(left_top[0])} {fmt(left_top[1])} "
        f"A {fmt(rx)} {fmt(ry)} 0 0 0 {fmt(right_top[0])} {fmt(right_top[1])} "
        f"L {fmt(right_bottom[0])} {fmt(right_bottom[1])} "
        f"A {fmt(rx)} {fmt(ry)} 0 0 1 {fmt(left_bottom[0])} {fmt(left_bottom[1])} Z"
    )


def curved_arrow_path(points: Sequence[tuple[float, float]]) -> str:
    if len(points) < 4:
        raise ValueError("Need at least four control points for curved path.")
    p0, p1, p2, p3 = points[:4]
    return (
        f"M {fmt(p0[0])} {fmt(p0[1])} "
        f"C {fmt(p1[0])} {fmt(p1[1])}, {fmt(p2[0])} {fmt(p2[1])}, {fmt(p3[0])} {fmt(p3[1])}"
    )


def standard_defs(prefix: str) -> str:
    return (
        "<defs>"
        f'<linearGradient id="{prefix}-card" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#F8FBFE"/>'
        '<stop offset="100%" stop-color="#EEF4FA"/>'
        "</linearGradient>"
        f'<linearGradient id="{prefix}-arrow" x1="0%" y1="0%" x2="100%" y2="0%">'
        '<stop offset="0%" stop-color="#DCE5EE" stop-opacity="0.35"/>'
        '<stop offset="100%" stop-color="#C9D7E3" stop-opacity="0.80"/>'
        "</linearGradient>"
        f'<marker id="{prefix}-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
        '<path d="M 0 0 L 8 4 L 0 8 Z" fill="#374151"/>'
        "</marker>"
        f'<marker id="{prefix}-grayhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
        '<path d="M 0 0 L 8 4 L 0 8 Z" fill="#9CA3AF"/>'
        "</marker>"
        f'<filter id="{prefix}-blur" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="10"/>'
        "</filter>"
        f'<radialGradient id="{prefix}-hotspot" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#D7191C"/>'
        '<stop offset="28%" stop-color="#F46D43"/>'
        '<stop offset="55%" stop-color="#FDAE61"/>'
        '<stop offset="78%" stop-color="#ABD9E9"/>'
        '<stop offset="100%" stop-color="#2C7BB6"/>'
        "</radialGradient>"
        f'<radialGradient id="{prefix}-disc" cx="50%" cy="42%" r="70%">'
        '<stop offset="0%" stop-color="#F8FAFC"/>'
        '<stop offset="65%" stop-color="#D8E3EE"/>'
        '<stop offset="100%" stop-color="#B6C8D8"/>'
        "</radialGradient>"
        "</defs>"
    )


def panel_card(x: float, y: float, w: float, h: float, grad_id: str) -> str:
    return (
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="18" ry="18" '
        f'fill="url(#{grad_id})" stroke="none"/>'
    )


def build_teng_group(cx: float, cy: float) -> str:
    parts = []
    parts.append('<g id="teng-device">')
    layer_specs = [
        (0, "#F5F7FA", "#6B7280", 0.95),
        (24, "#E5EBF1", "#6B7280", 0.98),
        (48, "#D9E1E8", "#6B7280", 1.0),
    ]
    rx = 114
    ry = 40
    height = 22
    base_top = cy + 36
    for offset, fill, stroke, opacity in reversed(layer_specs):
        y_top = base_top + offset
        parts.append(
            f'<path d="{front_band_path(cx, y_top, rx, ry, height)}" fill="{fill}" '
            f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.3"/>'
        )
        parts.append(
            f'<ellipse cx="{fmt(cx)}" cy="{fmt(y_top)}" rx="{fmt(rx)}" ry="{fmt(ry)}" '
            f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.3"/>'
        )
        parts.append(
            f'<path d="M {fmt(cx-rx)} {fmt(y_top+height)} A {fmt(rx)} {fmt(ry)} 0 0 0 {fmt(cx+rx)} {fmt(y_top+height)}" '
            f'stroke="{stroke}" stroke-width="1.3" fill="none"/>'
        )

    top_cy = cy
    parts.append(
        f'<ellipse cx="{fmt(cx)}" cy="{fmt(top_cy)}" rx="{fmt(rx+10)}" ry="{fmt(ry+8)}" '
        f'fill="#F6F9FC" stroke="#4B5563" stroke-width="1.6"/>'
    )
    parts.append(
        f'<ellipse cx="{fmt(cx)}" cy="{fmt(top_cy)}" rx="{fmt(rx)}" ry="{fmt(ry)}" '
        f'fill="url(#left-disc)" stroke="#4B5563" stroke-width="1.4"/>'
    )

    sectors = 12
    gap = math.radians(6)
    for idx in range(sectors):
        start = -math.pi / 2 + idx * math.tau / sectors + gap / 2
        end = -math.pi / 2 + (idx + 1) * math.tau / sectors - gap / 2
        parts.append(
            f'<path d="{annular_sector_path(cx, top_cy, 90, 31, 28, 11, start, end)}" '
            f'fill="#4D86B3" stroke="#3F6B91" stroke-width="1.0"/>'
        )
    parts.append(
        f'<ellipse cx="{fmt(cx)}" cy="{fmt(top_cy)}" rx="28" ry="11" fill="#F8FAFC" '
        f'stroke="#6B7280" stroke-width="1.4"/>'
    )
    parts.append(
        f'<ellipse cx="{fmt(cx)}" cy="{fmt(top_cy)}" rx="17" ry="6.5" fill="#E5EBF1" '
        f'stroke="#9CA3AF" stroke-width="1.1"/>'
    )

    start_arc = ellipse_point(cx, top_cy - 2, 118, 49, math.radians(200))
    end_arc = ellipse_point(cx, top_cy - 2, 118, 49, math.radians(-20))
    parts.append(
        f'<path d="M {fmt(start_arc[0])} {fmt(start_arc[1])} A 118 49 0 0 1 {fmt(end_arc[0])} {fmt(end_arc[1])}" '
        f'stroke="#1F2937" stroke-width="1.4" fill="none" marker-start="url(#left-grayhead)" marker-end="url(#left-grayhead)"/>'
    )
    parts.append(text_line(cx, top_cy - 69, "n", size=18, weight=400, anchor="middle", fill="#111827", style="font-style: italic"))
    parts.append(text_line(cx + 8, top_cy + 85, "ε", size=20, anchor="middle", fill="#111827", style="font-style: italic"))
    parts.append(multiline_text(cx + 66, base_top + 69, ["dielectric layer"], size=15, anchor="start", fill="#374151", italic=True))
    parts.append(multiline_text(cx + 48, base_top + 112, ["bottom electrode"], size=15, anchor="start", fill="#374151", italic=True))

    parts.append(
        f'<line x1="{fmt(cx+146)}" y1="{fmt(base_top+20)}" x2="{fmt(cx+146)}" y2="{fmt(base_top+58)}" '
        f'stroke="#6B7280" stroke-width="1.2" marker-start="url(#left-grayhead)" marker-end="url(#left-grayhead)"/>'
    )
    parts.append(text_line(cx + 154, base_top + 44, "d/R", size=15, anchor="start", fill="#111827"))
    parts.append(
        f'<line x1="{fmt(cx+146)}" y1="{fmt(base_top+60)}" x2="{fmt(cx+146)}" y2="{fmt(base_top+97)}" '
        f'stroke="#6B7280" stroke-width="1.2" marker-start="url(#left-grayhead)" marker-end="url(#left-grayhead)"/>'
    )
    parts.append(text_line(cx + 154, base_top + 84, "h/R", size=15, anchor="start", fill="#111827"))
    parts.append("</g>")
    return "".join(parts)


def comsol_thumb(x: float, y: float, label: str, radial: bool = True) -> str:
    parts = []
    parts.append(
        f'<g><rect x="{fmt(x)}" y="{fmt(y)}" width="102" height="82" rx="10" ry="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.1"/>'
    )
    if radial:
        parts.append(f'<circle cx="{fmt(x+40)}" cy="{fmt(y+40)}" r="30" fill="url(#left-hotspot)"/>')
    else:
        parts.append(f'<circle cx="{fmt(x+51)}" cy="{fmt(y+40)}" r="31" fill="url(#left-hotspot)"/>')
        for i in range(18):
            angle = math.radians(i * 20)
            x0 = x + 51
            y0 = y + 40
            x1 = x0 + 31 * math.cos(angle)
            y1 = y0 + 31 * math.sin(angle)
            parts.append(
                f'<line x1="{fmt(x0)}" y1="{fmt(y0)}" x2="{fmt(x1)}" y2="{fmt(y1)}" '
                f'stroke="#475569" stroke-width="0.8" stroke-opacity="0.55"/>'
            )
    parts.append(text_line(x + 51, y + 19, label, size=14, anchor="middle", fill="#374151"))
    parts.append("</g>")
    return "".join(parts)


def left_panel_group(x: float = 0, y: float = 0) -> str:
    parts = [f'<g transform="translate({fmt(x)} {fmt(y)})">']
    parts.append(panel_card(0, 0, 360, 520, "left-card"))
    parts.append(multiline_text(180, 40, ["Disk TENG inputs"], size=29, weight=700, anchor="middle"))
    parts.append(build_teng_group(166, 178))
    parts.append(comsol_thumb(18, 390, "", radial=True))
    parts.append(text_line(180, 447, "COMSOL", size=18, anchor="middle", fill="#111827"))
    parts.append(comsol_thumb(236, 390, "COMSOL/FEM", radial=False))
    parts.append("</g>")
    return "".join(parts)


def surrogate_panel_group(x: float = 0, y: float = 0) -> str:
    parts = [f'<g transform="translate({fmt(x)} {fmt(y)})">']
    parts.append(panel_card(0, 0, 580, 520, "mid-card"))
    parts.append(multiline_text(290, 38, ["Physics-consistent", "multi-task surrogate"], size=28, weight=700, anchor="middle"))
    parts.append(multiline_text(145, 265, ["Structural-", "dielectric inputs"], size=18, anchor="middle", fill="#374151"))
    parts.append(
        '<rect x="108" y="178" width="168" height="214" rx="12" ry="12" '
        'fill="rgba(255,255,255,0.35)" stroke="#94A3B8" stroke-width="1.6"/>'
    )
    parts.append(multiline_text(178, 156, ["Multi-task", "surrogate"], size=18, anchor="middle", fill="#111827"))
    parts.append('<rect x="126" y="190" width="16" height="190" rx="6" fill="#74A7CD" stroke="#3F6B91" stroke-width="1.2"/>')
    hidden_y = [212, 247, 282, 317, 352]
    for yy in hidden_y:
        parts.append(f'<circle cx="180" cy="{yy}" r="10" fill="#D1D5DB" stroke="#9CA3AF" stroke-width="1.0"/>')
        parts.append(f'<line x1="142" y1="{yy}" x2="170" y2="{yy}" stroke="#C7D2DA" stroke-width="1.1"/>')
    output_nodes = [(238, 246, "#E56B48"), (238, 282, "#F08A5B"), (238, 335, "#4D86B3")]
    for ox, oy, color in output_nodes:
        parts.append(f'<circle cx="{ox}" cy="{oy}" r="10" fill="{color}" stroke="#334155" stroke-width="1.0"/>')
    for yy in hidden_y:
        for ox, oy, _ in output_nodes:
            parts.append(
                f'<line x1="190" y1="{yy}" x2="{ox-10}" y2="{oy}" stroke="#C7D2DA" stroke-width="1.0" stroke-opacity="0.85"/>'
            )
    parts.append(
        f'<path d="{curved_arrow_path([(248, 246), (312, 232), (366, 225), (456, 262)])}" '
        'fill="none" stroke="#E56B48" stroke-width="5.5" stroke-linecap="round" marker-end="url(#mid-arrowhead)" stroke-opacity="0.95"/>'
    )
    parts.append(
        f'<path d="{curved_arrow_path([(248, 335), (308, 346), (366, 338), (452, 318)])}" '
        'fill="none" stroke="#4D86B3" stroke-width="5.5" stroke-linecap="round" marker-end="url(#mid-arrowhead)" stroke-opacity="0.95"/>'
    )
    parts.append(text_line(284, 224, "Qsc", size=18, weight=400, fill="#D95F3A"))
    parts.append(text_line(286, 384, "invC", size=18, weight=400, fill="#3F7AA6", style="font-style: italic"))
    parts.append(
        '<text x="328" y="310" font-family="Arial, Helvetica, sans-serif" font-size="25" fill="#111827" font-style="italic">'
        'FOMS'
        '<tspan baseline-shift="sub" font-size="16">direct</tspan>'
        ' ≈ FOMS'
        '<tspan baseline-shift="sub" font-size="16">phys</tspan>'
        "</text>"
    )
    parts.append(multiline_text(455, 352, ["Physics", "consistency"], size=16, anchor="start", fill="#374151", italic=True))
    parts.append("</g>")
    return "".join(parts)


def outcome_thumb(x: float, y: float, w: float, h: float, fill: str = "#FFFFFF") -> str:
    return f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="8" ry="8" fill="{fill}" stroke="#6B7280" stroke-width="1.2"/>'


def outcomes_panel_group(x: float = 0, y: float = 0) -> str:
    parts = [f'<g transform="translate({fmt(x)} {fmt(y)})">']
    parts.append(panel_card(0, 0, 236, 520, "right-card"))
    parts.append(multiline_text(118, 34, ["Physics-guided", "outcomes"], size=28, weight=700, anchor="middle"))

    parts.append(text_line(118, 80, "Mechanism landscape", size=16, anchor="middle"))
    parts.append(outcome_thumb(38, 92, 108, 72))
    parts.append('<clipPath id="right-mech-clip"><rect x="38" y="92" width="108" height="72" rx="8" ry="8"/></clipPath>')
    parts.append('<g clip-path="url(#right-mech-clip)">')
    parts.append('<rect x="38" y="92" width="108" height="72" fill="#EEF6FA"/>')
    parts.append('<rect x="18" y="90" width="72" height="84" fill="#2C6A96" filter="url(#right-blur)" opacity="0.95"/>')
    parts.append('<rect x="92" y="84" width="70" height="90" fill="#EA6A4C" filter="url(#right-blur)" opacity="0.95"/>')
    parts.append('<rect x="72" y="96" width="24" height="66" fill="#F6F7F9" filter="url(#right-blur)" opacity="0.85"/>')
    parts.append("</g>")

    parts.append(text_line(118, 210, "Design window", size=16, anchor="middle"))
    parts.append(outcome_thumb(38, 222, 108, 72))
    parts.append('<path d="M 72 273 A 86 58 0 0 1 129 240 L 118 231 A 72 46 0 0 0 68 259 Z" fill="#B9CCD9" stroke="#5B7082" stroke-width="1.0"/>')
    parts.append('<path d="M 112 266 A 86 58 0 0 1 129 240 L 118 231 A 72 46 0 0 0 105 255 Z" fill="#E57A57" stroke="#A95035" stroke-width="1.0"/>')
    parts.append('<path d="M 84 282 A 48 32 0 0 1 104 255 L 98 249 A 38 24 0 0 0 80 273 Z" fill="#E7EBEF" stroke="#5B7082" stroke-width="0.9"/>')

    parts.append(text_line(118, 338, "Safe region", size=16, anchor="middle"))
    parts.append(outcome_thumb(38, 350, 108, 72, fill="#E2F3EA"))
    parts.append('<path d="M 70 401 A 82 54 0 0 1 132 374 L 123 365 A 68 43 0 0 0 66 390 Z" fill="none" stroke="#586A79" stroke-width="4.2"/>')
    parts.append('<path d="M 92 407 A 42 27 0 0 1 112 391" fill="none" stroke="#586A79" stroke-width="4.2"/>')

    parts.append("</g>")
    return "".join(parts)


def bottom_pipeline_group(x: float = 0, y: float = 0) -> str:
    parts = [f'<g transform="translate({fmt(x)} {fmt(y)})">']
    labels = [
        "COMSOL data",
        "Multi-task surrogate",
        "Physics consistency",
        "Mechanism landscape",
        "Design window",
        "Unseen validation",
        "Web tool",
    ]
    xs = [0, 188, 378, 570, 758, 946, 1134]
    for title, xx in zip(labels, xs):
        parts.append(text_line(xx + 68, 18, title, size=15, anchor="middle"))
    thumb_w = 116
    thumb_h = 68
    top_y = 28

    # 1
    parts.append(outcome_thumb(xs[0], top_y, thumb_w, thumb_h, fill="#F5FAFD"))
    for i in range(6):
        parts.append(f'<line x1="{20 + i*14}" y1="{top_y+16}" x2="{20 + i*14}" y2="{top_y+52}" stroke="#B4C3D0" stroke-width="0.8"/>')
    for i in range(4):
        yy = top_y + 16 + i * 12
        parts.append(f'<line x1="12" y1="{yy}" x2="80" y2="{yy}" stroke="#B4C3D0" stroke-width="0.8"/>')
    parts.append('<circle cx="86" cy="50" r="18" fill="url(#bottom-hotspot)" stroke="#5B7082" stroke-width="1.0"/>')
    parts.append('<ellipse cx="86" cy="50" rx="30" ry="11" fill="none" stroke="#5B7082" stroke-width="1.0"/>')
    parts.append(multiline_text(xs[0] + 58, 112, ["small simulation-derived", "disk-TENG sample"], size=14, anchor="middle"))

    # 2
    parts.append(outcome_thumb(xs[1], top_y, thumb_w, thumb_h, fill="#F5FAFD"))
    parts.append('<rect x="206" y="40" width="12" height="44" rx="5" fill="#74A7CD" stroke="#3F6B91" stroke-width="0.9"/>')
    for cy in [44, 60, 76]:
        parts.append(f'<circle cx="248" cy="{cy}" r="7" fill="#D1D5DB" stroke="#9CA3AF" stroke-width="0.9"/>')
    for cy in [54, 70]:
        parts.append(f'<circle cx="280" cy="{cy}" r="6.5" fill="#E56B48" stroke="#9CA3AF" stroke-width="0.9"/>')
    parts.append('<circle cx="280" cy="86" r="6.5" fill="#4D86B3" stroke="#9CA3AF" stroke-width="0.9"/>')
    for sy in [44, 60, 76]:
        for ty in [54, 70, 86]:
            parts.append(f'<line x1="255" y1="{sy}" x2="273.5" y2="{ty}" stroke="#C7D2DA" stroke-width="0.8"/>')

    # 3
    parts.append(outcome_thumb(xs[2], top_y, thumb_w, thumb_h, fill="#F7FBFD"))
    parts.append(
        '<text x="437" y="70" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" '
        'font-style="italic" fill="#111827">FOMS<tspan baseline-shift="sub" font-size="9">direct</tspan> ≈ '
        'FOMS<tspan baseline-shift="sub" font-size="9">phys</tspan></text>'
    )

    # 4
    parts.append(outcome_thumb(xs[3], top_y, thumb_w, thumb_h))
    parts.append('<clipPath id="bottom-mech-clip"><rect x="570" y="28" width="116" height="68" rx="8" ry="8"/></clipPath>')
    parts.append('<g clip-path="url(#bottom-mech-clip)">')
    parts.append('<rect x="570" y="28" width="116" height="68" fill="#EEF6FA"/>')
    parts.append('<rect x="548" y="16" width="72" height="88" fill="#2C6A96" filter="url(#bottom-blur)" opacity="0.95"/>')
    parts.append('<rect x="636" y="16" width="70" height="88" fill="#EA6A4C" filter="url(#bottom-blur)" opacity="0.95"/>')
    parts.append('<rect x="620" y="20" width="18" height="88" fill="#F6F7F9" filter="url(#bottom-blur)" opacity="0.78"/>')
    parts.append("</g>")

    # 5
    parts.append(outcome_thumb(xs[4], top_y, thumb_w, thumb_h, fill="#EAF5EF"))
    parts.append('<path d="M 782 86 A 88 58 0 0 1 840 50 L 828 40 A 72 46 0 0 0 776 72 Z" fill="#B9CCD9" stroke="#5B7082" stroke-width="1.0"/>')
    parts.append('<path d="M 810 88 A 62 40 0 0 1 842 54 L 833 47 A 51 31 0 0 0 807 77 Z" fill="none" stroke="#5B7082" stroke-width="1.2"/>')
    parts.append('<path d="M 824 76 A 72 46 0 0 1 842 54 L 833 47 A 56 34 0 0 0 819 67 Z" fill="#E57A57" stroke="#A95035" stroke-width="0.9"/>')

    # 6
    parts.append(outcome_thumb(xs[5], top_y, thumb_w, thumb_h, fill="#F8FBFD"))
    parts.append('<line x1="977" y1="36" x2="1040" y2="92" stroke="#94A3B8" stroke-width="1.2"/>')
    parts.append('<line x1="981" y1="82" x2="1036" y2="42" stroke="#94A3B8" stroke-width="1.0"/>')
    parts.append('<path d="M 984 84 L 1030 46" stroke="#4D86B3" stroke-width="2.1" fill="none"/>')
    for c in [(988, 80), (1004, 66), (1021, 54), (1032, 46)]:
        parts.append(f'<circle cx="{c[0]}" cy="{c[1]}" r="2.3" fill="#4D86B3"/>')
    parts.append(multiline_text(957, 56, ["parity", "comparison"], size=11, anchor="start", fill="#475569", italic=True))

    # 7
    parts.append(outcome_thumb(xs[6], top_y, thumb_w, thumb_h, fill="#F8FBFD"))
    parts.append('<rect x="1166" y="38" width="84" height="46" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.0"/>')
    parts.append('<line x1="1174" y1="50" x2="1242" y2="50" stroke="#B4C3D0" stroke-width="1.0"/>')
    parts.append('<line x1="1174" y1="60" x2="1235" y2="60" stroke="#B4C3D0" stroke-width="1.0"/>')
    parts.append('<line x1="1174" y1="70" x2="1218" y2="70" stroke="#B4C3D0" stroke-width="1.0"/>')

    for start_x in [118, 308, 500, 690, 878, 1066]:
        parts.append(
            f'<line x1="{start_x}" y1="{top_y+34}" x2="{start_x+44}" y2="{top_y+34}" stroke="#334155" stroke-width="1.6" marker-end="url(#bottom-arrowhead)"/>'
        )
    parts.append("</g>")
    return "".join(parts)


def arrow_svg(width: int, height: int) -> str:
    defs = standard_defs("arrow")
    body = (
        f'<polygon points="0,{height/2} {width-18},0 {width-18},{height*0.28} {width},{height/2} '
        f'{width-18},{height*0.72} {width-18},{height}" fill="url(#arrow-arrow)" fill-opacity="0.82" stroke="none"/>'
    )
    return svg_wrap(width, height, defs, body)


def render_panel_assets() -> dict[str, str]:
    assets: dict[str, str] = {}
    assets["fig01_panel_a_left_teng.svg"] = svg_wrap(360, 520, standard_defs("left"), left_panel_group())
    assets["fig01_panel_a_mid_surrogate.svg"] = svg_wrap(580, 520, standard_defs("mid"), surrogate_panel_group())
    assets["fig01_panel_a_right_outcomes.svg"] = svg_wrap(236, 520, standard_defs("right"), outcomes_panel_group())
    assets["fig01_panel_b_pipeline.svg"] = svg_wrap(1250, 120, standard_defs("bottom"), bottom_pipeline_group())
    assets["fig01_interpanel_arrow.svg"] = arrow_svg(58, 92)
    return assets


def main_svg(assets: dict[str, str]) -> str:
    defs = standard_defs("main")
    body = [
        '<rect x="0" y="0" width="1400" height="768" fill="#FFFFFF"/>',
        text_line(12, 30, "(a)", size=30, weight=700),
        left_panel_group(52, 54),
        f'<g transform="translate(416 238)">{arrow_svg_group("main")}</g>',
        surrogate_panel_group(478, 54),
        f'<g transform="translate(1050 238)">{arrow_svg_group("main")}</g>',
        outcomes_panel_group(1110, 54),
        text_line(12, 612, "(b)", size=30, weight=700),
        bottom_pipeline_group(50, 604),
    ]
    return svg_wrap(1400, 768, defs, "".join(body))


def arrow_svg_group(prefix: str) -> str:
    return (
        '<polygon points="0,46 40,0 40,24 58,46 40,68 40,92" '
        f'fill="url(#{prefix}-arrow)" fill-opacity="0.82" stroke="none"/>'
    )


def data_uri(svg_text: str) -> str:
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def drawio_image_cell(cell_id: str, x: float, y: float, width: float, height: float, image_uri: str) -> str:
    style = (
        "shape=image;html=1;verticalAlign=middle;align=center;imageAspect=0;aspect=fixed;"
        f"image={image_uri};strokeColor=none;fillColor=none;"
    )
    return (
        f'<mxCell id="{cell_id}" value="" style="{escape(style)}" vertex="1" parent="1">'
        f'<mxGeometry x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(height)}" as="geometry"/>'
        "</mxCell>"
    )


def drawio_text_cell(cell_id: str, x: float, y: float, width: float, height: float, value: str, size: int = 28) -> str:
    style = (
        "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
        f"fontSize={size};fontStyle=1;fontFamily=Arial;fontColor=#111827;"
    )
    return (
        f'<mxCell id="{cell_id}" value="{escape(value)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(height)}" as="geometry"/>'
        "</mxCell>"
    )


def build_drawio(assets: dict[str, str]) -> str:
    left_uri = data_uri(assets["fig01_panel_a_left_teng.svg"])
    mid_uri = data_uri(assets["fig01_panel_a_mid_surrogate.svg"])
    right_uri = data_uri(assets["fig01_panel_a_right_outcomes.svg"])
    bottom_uri = data_uri(assets["fig01_panel_b_pipeline.svg"])
    arrow_uri = data_uri(assets["fig01_interpanel_arrow.svg"])
    cells = [
        drawio_text_cell("2", 12, 10, 48, 36, "(a)", 30),
        drawio_image_cell("3", 52, 54, 360, 520, left_uri),
        drawio_image_cell("4", 416, 238, 58, 92, arrow_uri),
        drawio_image_cell("5", 478, 54, 580, 520, mid_uri),
        drawio_image_cell("6", 1050, 238, 58, 92, arrow_uri),
        drawio_image_cell("7", 1110, 54, 236, 520, right_uri),
        drawio_text_cell("8", 12, 612, 48, 36, "(b)", 30),
        drawio_image_cell("9", 50, 604, 1250, 120, bottom_uri),
    ]
    model = (
        '<mxGraphModel dx="1460" dy="860" grid="0" gridSize="8" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1400" pageHeight="768" background="#FFFFFF" math="1">'
        "<root><mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
        + "".join(cells)
        + "</root></mxGraphModel>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="cli" modified="" agent="generate_fig01_drawio_replicate.py" version="21.0.0">\n'
        '  <diagram name="Page-1">\n'
        f'    {model}\n'
        '  </diagram>\n'
        '</mxfile>\n'
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def maybe_render_png(svg_path: Path) -> Path | None:
    try:
        import cairosvg  # type: ignore
    except Exception:
        return None

    png_path = svg_path.with_suffix(".png")
    cairosvg.svg2png(bytestring=svg_path.read_bytes(), write_to=str(png_path))
    return png_path


def main() -> None:
    assets = render_panel_assets()
    for name, content in assets.items():
        write_text(OUT_DIR / name, content)

    full_svg = main_svg(assets)
    full_svg_path = OUT_DIR / "fig01_workflow_replicated.svg"
    write_text(full_svg_path, full_svg)
    maybe_render_png(full_svg_path)

    drawio_text = build_drawio(assets)
    drawio_path = OUT_DIR / "fig01_workflow_replicated.drawio"
    write_text(drawio_path, drawio_text)

    arch_path = OUT_DIR / "fig01_workflow_replicated.arch.json"
    if arch_path.exists():
        arch = json.loads(arch_path.read_text(encoding="utf-8"))
    else:
        arch = {}
    arch.update(
        {
            "version": arch.get("version", 1),
            "title": "Physics-consistent Disk-TENG workflow",
            "type": "academic-diagram",
            "source": "replicated",
            "profile": "academic-paper",
            "theme": "academic-color",
            "layout": "horizontal",
            "generator": "figures_publication/src/fig01_workflow_support/generate_fig01_drawio_replicate.py",
            "artifacts": {
                "drawio": "fig01_workflow_replicated.drawio",
                "svg": "fig01_workflow_replicated.svg",
                "panels": [
                    "fig01_panel_a_left_teng.svg",
                    "fig01_panel_a_mid_surrogate.svg",
                    "fig01_panel_a_right_outcomes.svg",
                    "fig01_panel_b_pipeline.svg",
                    "fig01_interpanel_arrow.svg",
                ],
            },
            "notes": [
                "The final drawio is self-contained with embedded SVG panel assets.",
                "Panel SVGs remain editable for geometry-level refinement, especially the left Disk-TENG structure.",
            ],
        }
    )
    write_text(arch_path, json.dumps(arch, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
