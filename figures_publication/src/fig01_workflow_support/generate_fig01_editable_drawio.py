#!/usr/bin/env python3

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from PIL import Image
from fig01_integrated_baseline import load_baseline_xml
from generate_fig01_panel_b_icons import generate_panel_b_icons


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = OUT_DIR / "crops"
BACKUP_DRAWIO = REPO_ROOT / "figures_publication" / "src" / "fig01_method_workflow.drawio"
# Manual reference only. This script must never overwrite the backup file.
PANEL_B_TITLES = [
    "COMSOL dataset",
    "Channel decomposition",
    "Landscape construction",
    "Robust screening",
    "Unseen validation",
    "Open design tool",
]
PANEL_B_ICON_KEYS = [
    "panel_b_icon_comsol_dataset",
    "panel_b_icon_channel_decomposition",
    "panel_b_icon_landscape_construction",
    "panel_b_icon_robust_screening",
    "panel_b_icon_unseen_validation",
    "panel_b_icon_open_design_tool",
]


def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def shade_color(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02X}{g:02X}{b:02X}"


@dataclass
class TENGGeometry:
    n_sectors: int = 8
    rx: float = 56
    ry: float = 16
    hub_rx: float = 11
    hub_ry: float = 4
    top_thickness: float = 10
    diel_thickness: float = 10
    gap: float = 18
    bottom_thickness: float = 10


class DrawioBuilder:
    def __init__(self) -> None:
        self.cells: list[str] = []
        self.next_id = 2

    def alloc(self) -> str:
        out = str(self.next_id)
        self.next_id += 1
        return out

    def add_vertex(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        style: str,
        value: str = "",
        parent: str = "1",
    ) -> str:
        cell_id = self.alloc()
        self.cells.append(
            f'<mxCell id="{cell_id}" value="{escape(value)}" style="{escape(style)}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" as="geometry"/>'
            "</mxCell>"
        )
        return cell_id

    def add_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        size: int = 16,
        bold: bool = False,
        italic: bool = False,
        align: str = "center",
        vertical: str = "middle",
        color: str = "#111827",
        parent: str = "1",
    ) -> str:
        font_style = 0
        if bold:
            font_style += 1
        if italic:
            font_style += 2
        style = (
            "text;html=1;strokeColor=none;fillColor=none;"
            f"align={align};verticalAlign={vertical};fontSize={size};"
            f"fontStyle={font_style};fontFamily=Arial;fontColor={color};"
            "whiteSpace=wrap;"
        )
        return self.add_vertex(x, y, w, h, style, value, parent=parent)

    def add_image(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        image_path: Path,
        parent: str = "1",
    ) -> str:
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }.get(image_path.suffix.lower(), "image/png")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        style = (
            "shape=image;html=1;verticalAlign=middle;align=center;"
            f"imageAspect=0;aspect=fixed;image=data:{mime};base64,{encoded};"
            "strokeColor=none;fillColor=none;"
        )
        return self.add_vertex(x, y, w, h, style, "", parent=parent)

    def add_edge(
        self,
        style: str,
        value: str = "",
        parent: str = "1",
        cell_id: str | None = None,
        source: str | None = None,
        target: str | None = None,
        source_point: tuple[float, float] | None = None,
        target_point: tuple[float, float] | None = None,
        points: list[tuple[float, float]] | None = None,
    ) -> str:
        cell_id = cell_id or self.alloc()
        attrs = [
            f'id="{cell_id}"',
            f'value="{escape(value)}"',
            f'style="{escape(style)}"',
            'edge="1"',
            f'parent="{parent}"',
        ]
        if source:
            attrs.append(f'source="{source}"')
        if target:
            attrs.append(f'target="{target}"')
        geometry = ['<mxGeometry relative="1" as="geometry">']
        if points:
            geometry.append('<Array as="points">')
            for px, py in points:
                geometry.append(f'<mxPoint x="{fmt(px)}" y="{fmt(py)}"/>')
            geometry.append("</Array>")
        if source_point:
            geometry.append(
                f'<mxPoint x="{fmt(source_point[0])}" y="{fmt(source_point[1])}" as="sourcePoint"/>'
            )
        if target_point:
            geometry.append(
                f'<mxPoint x="{fmt(target_point[0])}" y="{fmt(target_point[1])}" as="targetPoint"/>'
            )
        geometry.append("</mxGeometry>")
        self.cells.append(f"<mxCell {' '.join(attrs)}>{''.join(geometry)}</mxCell>")
        return cell_id

    def build(self, page_width: int = 1400, page_height: int = 780) -> str:
        model = (
            f'<mxGraphModel dx="1543" dy="683" grid="0" gridSize="8" guides="1" tooltips="1" connect="1" '
            f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{page_width}" pageHeight="{page_height}" '
            'background="#FFFFFF" math="1" shadow="0">'
            '<root><mxCell id="0" /><mxCell id="1" parent="0" />'
            + "".join(self.cells)
            + "</root></mxGraphModel>"
        )
        return (
            '<mxfile host="Electron" agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/29.6.6 Chrome/144.0.7559.236 Electron/40.8.4 Safari/537.36" version="29.6.6">\n'
            '  <diagram name="Page-1" id="FJdx3XjqQJT51ps4pwYS">\n'
            f"    {model}\n"
            "  </diagram>\n"
            "</mxfile>\n"
        )


def edge_style(
    stroke: str = "#334155",
    width: float = 1.6,
    end: str = "block",
    start: str | None = None,
    dashed: bool = False,
    curved: bool = False,
    opacity: int | None = None,
) -> str:
    parts = [
        "edgeStyle=none",
        "html=1",
        f"strokeColor={stroke}",
        f"strokeWidth={width}",
        f"endArrow={end}",
        "endFill=1" if end != "none" else "endFill=0",
    ]
    if start:
        parts.append(f"startArrow={start}")
        parts.append("startFill=1")
    if dashed:
        parts.extend(["dashed=1", "dashPattern=6 4"])
    if curved:
        parts.extend(["curved=1", "rounded=0"])
    if opacity is not None:
        parts.append(f"opacity={opacity}")
    return ";".join(parts) + ";"


def rect_style(
    fill: str,
    stroke: str = "#CBD5E1",
    rounded: int = 1,
    arc: int = 18,
    width: float = 1.3,
    gradient: str | None = None,
    opacity: int | None = None,
) -> str:
    parts = [
        f"rounded={rounded}",
        f"arcSize={arc}",
        "html=1",
        "whiteSpace=wrap",
        f"fillColor={fill}",
        f"strokeColor={stroke}",
        f"strokeWidth={width}",
        "fontColor=#111827",
        "fontSize=14",
        "fontFamily=Arial",
    ]
    if gradient:
        parts.extend([f"gradientColor={gradient}", "gradientDirection=northWest"])
    if opacity is not None:
        parts.append(f"opacity={opacity}")
    return ";".join(parts) + ";"


def ellipse_style(
    fill: str,
    stroke: str = "#475569",
    width: float = 1.2,
    opacity: int | None = None,
) -> str:
    parts = [
        "ellipse",
        "html=1",
        "whiteSpace=wrap",
        f"fillColor={fill}",
        f"strokeColor={stroke}",
        f"strokeWidth={width}",
    ]
    if opacity is not None:
        parts.append(f"opacity={opacity}")
    return ";".join(parts) + ";"


def part_conc_ellipse_style(
    fill: str,
    stroke: str = "#3F6B91",
    width: float = 1.0,
    start_angle: float = 0.0,
    end_angle: float = 0.1,
    arc_width: float = 0.6,
) -> str:
    parts = [
        "shape=mxgraph.basic.partConcEllipse",
        "html=1",
        "whiteSpace=wrap",
        f"fillColor={fill}",
        f"strokeColor={stroke}",
        f"strokeWidth={width}",
        f"startAngle={start_angle:.4f}",
        f"endAngle={end_angle:.4f}",
        f"arcWidth={arc_width:.4f}",
    ]
    return ";".join(parts) + ";"


def cylinder_style(
    fill: str,
    stroke: str = "#6B7280",
    width: float = 1.2,
    opacity: int | None = None,
    gradient: str | None = None,
) -> str:
    parts = [
        "shape=cylinder3",
        "boundedLbl=1",
        "backgroundOutline=1",
        "size=15",
        "html=1",
        f"fillColor={fill}",
        f"strokeColor={stroke}",
        f"strokeWidth={width}",
    ]
    if gradient:
        parts.extend([f"gradientColor={gradient}", "gradientDirection=east"])
    if opacity is not None:
        parts.append(f"opacity={opacity}")
    return ";".join(parts) + ";"


def line_to_ellipse(
    cx: float, cy: float, rx: float, ry: float, angle_deg: float
) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    return cx + rx * math.cos(rad), cy + ry * math.sin(rad)


def crop_box(src: Path, box: tuple[int, int, int, int], dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.crop(box).save(dst)
    return dst


def crop_subject_centered(
    src: Path,
    dst: Path,
    target_ratio: float = 1.30,
    padding: int = 18,
    white_threshold: int = 245,
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src).convert("RGBA") as img:
        w, h = img.size
        minx, miny, maxx, maxy = w, h, 0, 0
        found = False
        for y in range(h):
            for x in range(w):
                r, g, b, a = img.getpixel((x, y))
                if a > 10 and not (
                    r > white_threshold and g > white_threshold and b > white_threshold
                ):
                    found = True
                    minx = min(minx, x)
                    miny = min(miny, y)
                    maxx = max(maxx, x)
                    maxy = max(maxy, y)
        if not found:
            img.save(dst)
            return dst

        left = max(0, minx - padding)
        top = max(0, miny - padding)
        right = min(w, maxx + padding)
        bottom = min(h, maxy + padding)

        crop_w = right - left
        crop_h = bottom - top
        current_ratio = crop_w / crop_h if crop_h else target_ratio

        cx = (left + right) / 2
        cy = (top + bottom) / 2

        if current_ratio > target_ratio:
            new_h = crop_w / target_ratio
            crop_h = min(h, new_h)
        else:
            new_w = crop_h * target_ratio
            crop_w = min(w, new_w)

        left = max(0, int(round(cx - crop_w / 2)))
        top = max(0, int(round(cy - crop_h / 2)))
        right = min(w, int(round(left + crop_w)))
        bottom = min(h, int(round(top + crop_h)))

        # Re-clamp if rounding pushed out of bounds.
        left = max(0, right - int(round(crop_w)))
        top = max(0, bottom - int(round(crop_h)))

        img.crop((left, top, right, bottom)).save(dst)
    return dst


def prepare_real_image_assets() -> dict[str, Path]:
    panel_b_icons = generate_panel_b_icons(OUT_DIR / "panel_b_icons")
    assets = {
        "comsol": crop_box(
            OUT_DIR / "pdf_refs" / "si_p2-02.png",
            (720, 110, 1095, 435),
            ASSET_DIR / "comsol_disk_crop.png",
        ),
        "mechanism": crop_box(
            REPO_ROOT
            / "figures_publication"
            / "main"
            / "fig02_mechanism_landscape.png",
            (1180, 70, 1990, 760),
            ASSET_DIR / "mechanism_landscape_crop.png",
        ),
        "parity": crop_box(
            REPO_ROOT
            / "figures_publication"
            / "main"
            / "fig05_unseen_structural_dielectric_validation.png",
            (1440, 65, 2140, 825),
            ASSET_DIR / "parity_crop.png",
        ),
        "icon_mechanism_landscape": OUT_DIR / "fig1_icon_mechanism_landscape.png",
        "icon_design_window": OUT_DIR / "fig1_icon_design_window.png",
        "icon_safe_region": OUT_DIR / "fig1_icon_safe_region.png",
        "icon_mechanism_landscape_svg": OUT_DIR / "fig1_icon_mechanism_landscape.svg",
        "icon_design_window_svg": OUT_DIR / "fig1_icon_design_window.svg",
        "icon_safe_region_svg": OUT_DIR / "fig1_icon_safe_region.svg",
        "fem_start": crop_subject_centered(
            OUT_DIR / "FEM_start.png",
            ASSET_DIR / "FEM_start_core.png",
            target_ratio=1.30,
            padding=18,
        ),
        "fem_end": crop_subject_centered(
            OUT_DIR / "FEM_end.png",
            ASSET_DIR / "FEM_end_core.png",
            target_ratio=1.30,
            padding=18,
        ),
    }
    assets.update(panel_b_icons)
    return assets


def image_style_from_path(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(suffix, "image/png")
    return (
        "shape=image;html=1;verticalAlign=middle;align=center;"
        f"imageAspect=0;aspect=fixed;image=data:{mime};base64,{encoded};"
        "strokeColor=none;fillColor=none;"
    )


def vertex_cell_xml(
    cell_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    style: str,
    value: str = "",
) -> str:
    return (
        f'        <mxCell id="{cell_id}" parent="1" style="{escape(style)}" '
        f'value="{escape(value)}" vertex="1">\n'
        f'          <mxGeometry height="{fmt(h)}" width="{fmt(w)}" '
        f'x="{fmt(x)}" y="{fmt(y)}" as="geometry" />\n'
        "        </mxCell>\n"
    )


def append_root_vertex(
    mxroot: ET.Element,
    cell_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    style: str,
    value: str = "",
) -> ET.Element:
    cell = ET.SubElement(
        mxroot,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": style,
            "vertex": "1",
            "parent": "1",
        },
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {
            "x": fmt(x),
            "y": fmt(y),
            "width": fmt(w),
            "height": fmt(h),
            "as": "geometry",
        },
    )
    return cell


def append_root_edge(
    mxroot: ET.Element,
    cell_id: str,
    style: str,
    source_point: tuple[float, float],
    target_point: tuple[float, float],
    points: list[tuple[float, float]] | None = None,
) -> ET.Element:
    cell = ET.SubElement(
        mxroot,
        "mxCell",
        {
            "id": cell_id,
            "value": "",
            "style": style,
            "edge": "1",
            "parent": "1",
        },
    )
    geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if points:
        arr = ET.SubElement(geom, "Array", {"as": "points"})
        for px, py in points:
            ET.SubElement(arr, "mxPoint", {"x": fmt(px), "y": fmt(py)})
    ET.SubElement(
        geom,
        "mxPoint",
        {"x": fmt(source_point[0]), "y": fmt(source_point[1]), "as": "sourcePoint"},
    )
    ET.SubElement(
        geom,
        "mxPoint",
        {"x": fmt(target_point[0]), "y": fmt(target_point[1]), "as": "targetPoint"},
    )
    return cell


def build_outcomes_panel_override(assets: dict[str, Path]) -> str:
    panel_style = rect_style(
        "#F7FAFD",
        stroke="#DCE6F0",
        gradient="#EDF3F8",
        rounded=1,
        arc=18,
        width=1.0,
    )
    panel_title_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontSize=20;fontStyle=1;fontFamily=Arial;"
        "fontColor=#111827;whiteSpace=wrap;"
    )
    text_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontSize=15;fontStyle=0;fontFamily=Arial;"
        "fontColor=#111827;whiteSpace=wrap;"
    )
    card_style = rect_style("#FBFCFE", stroke="#C9D6E0", rounded=1, arc=14, width=1.0)
    block = []
    panel_x = 1124
    title_x = 1138
    title_w = 208
    card_x = 1151
    card_w = 182
    card_h = 112
    image_w = 152
    image_h = 100
    image_x = card_x + (card_w - image_w) / 2
    block.append(vertex_cell_xml("155", panel_x, 54, 236, 560, panel_style))
    block.append(
        vertex_cell_xml(
            "156",
            panel_x + 12,
            70,
            212,
            34,
            panel_title_style,
            "Mechanism-aware design outputs",
        )
    )
    sections = [
        ("157", "158", "159", "Mechanism landscape", 138, 152, "icon_mechanism_landscape"),
        ("160", "161", "162", "Design window", 292, 306, "icon_design_window"),
        ("163", "164", "165", "Safe region", 446, 460, "icon_safe_region"),
    ]
    for title_id, card_id, image_id, title, title_y, card_y, icon_key in sections:
        block.append(vertex_cell_xml(title_id, title_x, title_y, title_w, 20, text_style, title))
        block.append(vertex_cell_xml(card_id, card_x, card_y, card_w, card_h, card_style))
        if assets.get(icon_key) and assets[icon_key].exists():
            block.append(
                vertex_cell_xml(
                    image_id,
                    image_x,
                    card_y + 8,
                    image_w,
                    96,
                    image_style_from_path(assets[icon_key]),
                )
            )
    return "".join(block)


def apply_outcomes_panel_override(xml: str, assets: dict[str, Path]) -> str:
    start = xml.index('        <mxCell id="155"')
    end = xml.index('        <mxCell id="169"')
    return xml[:start] + build_outcomes_panel_override(assets) + xml[end:]


def add_charge_row(
    b: DrawioBuilder,
    x0: float,
    x1: float,
    y: float,
    sign: str,
    color: str,
    count: int = 7,
    size: int = 11,
) -> None:
    if count <= 1:
        xs = [(x0 + x1) / 2]
    else:
        xs = [x0 + i * (x1 - x0) / (count - 1) for i in range(count)]
    for xx in xs:
        b.add_text(xx - 6, y - 8, 12, 12, sign, size=size, bold=True, color=color)


def add_sector_ring(
    b: DrawioBuilder,
    cx: float,
    cy: float,
    rx_out: float,
    ry_out: float,
    rx_in: float,
    ry_in: float,
    sectors: int,
    fill: str,
    stroke: str = "#F8FBFE",
    gap_deg: float = 8.0,
    rotation_deg: float = 0.0,
) -> None:
    step = 360.0 / sectors
    for i in range(sectors):
        a0 = i * step + gap_deg / 2 + rotation_deg
        a1 = (i + 1) * step - gap_deg / 2 + rotation_deg
        b.add_vertex(
            cx - rx_out,
            cy - ry_out,
            2 * rx_out,
            2 * ry_out,
            part_conc_ellipse_style(
                fill,
                stroke=stroke,
                width=1.0,
                start_angle=(a0 % 360.0) / 360.0,
                end_angle=(a1 % 360.0) / 360.0,
                arc_width=0.70,
            ),
        )


def add_teng_state(
    b: DrawioBuilder,
    x: float,
    y: float,
    title: str,
    geom: TENGGeometry,
    separated: bool = False,
    aligned_height: bool = False,
    show_layer_labels: bool = True,
    show_geometry_annotations: bool = True,
    show_surface_charges: bool = True,
    title_size: int = 14,
) -> None:
    # palette
    top_fill = "#F3F6FA"
    top_side = "#E8EEF5"
    diel_fill = "#D9E9F5"
    diel_side = "#BFD4E3"
    bot_fill = "#E7ECF2"
    bot_side = "#C9D1D9"
    stroke = "#334155"

    top_sector_fill = "#2C7FB8"
    bottom_sector_fill = "#94A3B8"

    # triboelectric interface charges
    neg_charge = "#3B82C4"  # dielectric surface charge
    pos_charge = "#E64B35"  # opposite interface charge

    # state logic
    if aligned_height:
        gap = geom.gap
        upper_shift = -10
    else:
        gap = geom.gap if separated else 2
        upper_shift = -10 if separated else 0

    # very important:
    # contacted -> top/bottom sectors aligned
    # separated -> top sectors rotated by half-sector to indicate non-overlap / end state
    sector_offset = 0.0 if not separated else 180.0 / geom.n_sectors

    # local anchor
    cx = x + 88
    top_face_y = y + 58 + upper_shift
    diel_y = top_face_y + geom.top_thickness + 6
    bottom_y = diel_y + geom.diel_thickness + gap + 20

    # title
    b.add_text(x + 10, y + 6, 156, 20, title, size=title_size, bold=True)

    # -------------------------
    # bottom electrode body
    # -------------------------
    b.add_vertex(
        cx - geom.rx,
        bottom_y,
        2 * geom.rx,
        geom.bottom_thickness + 2 * geom.ry,
        cylinder_style(bot_fill, stroke="#7B8794", width=1.0, gradient=bot_side),
    )

    # bottom sector electrodes on top face
    add_sector_ring(
        b,
        cx,
        bottom_y + geom.ry - 2,
        geom.rx - 8,
        geom.ry - 4,
        geom.hub_rx,
        geom.hub_ry,
        geom.n_sectors,
        bottom_sector_fill,
        stroke="#F8FBFE",
        rotation_deg=0.0,
    )

    # -------------------------
    # dielectric layer
    # -------------------------
    b.add_vertex(
        cx - geom.rx + 4,
        diel_y,
        2 * (geom.rx - 4),
        geom.diel_thickness + 2 * geom.ry,
        cylinder_style(diel_fill, stroke="#6E9FBE", width=1.0, gradient=diel_side),
    )

    # -------------------------
    # top electrode body
    # -------------------------
    b.add_vertex(
        cx - geom.rx + 2,
        top_face_y,
        2 * (geom.rx - 2),
        geom.top_thickness + 2 * geom.ry,
        cylinder_style(top_fill, stroke="#7B8794", width=1.0, gradient=top_side),
    )

    # top sector electrodes
    add_sector_ring(
        b,
        cx,
        top_face_y + geom.ry - 2,
        geom.rx - 10,
        geom.ry - 4,
        geom.hub_rx,
        geom.hub_ry,
        geom.n_sectors,
        top_sector_fill,
        stroke="#F8FBFE",
        rotation_deg=sector_offset,
    )

    # center hub
    b.add_vertex(
        cx - geom.hub_rx,
        top_face_y + geom.ry - geom.hub_ry - 2,
        2 * geom.hub_rx,
        2 * geom.hub_ry,
        ellipse_style("#F8FBFE", stroke="#64748B", width=1.0),
    )

    # -------------------------
    # air gap (empty space only)
    # -------------------------
    if separated:
        pass

    # -------------------------
    # n arrow
    # -------------------------
    if show_geometry_annotations:
        arc_start = line_to_ellipse(
            cx, top_face_y + geom.ry - 2, geom.rx - 4, geom.ry + 10, 205
        )
        arc_end = line_to_ellipse(
            cx, top_face_y + geom.ry - 2, geom.rx - 4, geom.ry + 10, -25
        )
        b.add_edge(
            edge_style(
                stroke="#111827", width=1.0, end="classic", curved=True
            ),
            source_point=arc_start,
            target_point=arc_end,
            points=[(cx, top_face_y - 18)],
        )
        b.add_text(cx - 8, top_face_y - 28, 20, 16, "n", size=14, italic=True)

    # -------------------------
    # labels
    # -------------------------
    if show_layer_labels:
        b.add_text(
            x + 154,
            top_face_y + 4,
            70,
            16,
            "Top electrode",
            size=11,
            align="left",
            color="#374151",
        )
        b.add_text(
            x + 154,
            diel_y + 8,
            76,
            16,
            "Dielectric",
            size=11,
            align="left",
            color="#374151",
        )
        b.add_text(
            x + 154,
            bottom_y + 6,
            84,
            16,
            "Bottom electrode",
            size=11,
            align="left",
            color="#374151",
        )
    if show_geometry_annotations:
        b.add_text(cx - 5, top_face_y + 44, 20, 14, "ε", size=13, italic=True)

    if show_geometry_annotations:
        dimx = x + 140
        b.add_text(
            dimx + 8,
            top_face_y + 30,
            28,
            14,
            "h/R",
            size=11,
            align="left",
        )

    if show_geometry_annotations:
        dimx2 = x + 128
        dim_style = (
            edge_style(stroke="#6B7280", width=0.9, end="classic", start="classic")
            + "startSize=1;endSize=1;"
        )
        b.add_edge(
            dim_style,
            cell_id="NpRr0kzRW96YMu1P_3pb-194",
            source_point=(x + 144.23, y + 78),
            target_point=(x + 144, y + 99),
        )
        b.add_edge(
            dim_style,
            source_point=(dimx2, y + 102),
            target_point=(dimx2, y + 126),
        )
        b.add_text(
            x + 134,
            top_face_y + 58,
            28,
            14,
            "d/R",
            size=11,
            align="left",
        )
        if show_layer_labels:
            b.add_text(
                x + 154,
                (diel_y + bottom_y) / 2 - 2,
                54,
                14,
                "Air gap",
                size=11,
                align="left",
                color="#374151",
            )

    # -------------------------
    # scientifically stricter charge depiction
    # -------------------------
    # triboelectric surface charges should reside on the facing interface:
    # - dielectric lower surface
    # - lower electrode upper surface (or lower friction partner surface)
    x0 = cx - 34
    x1 = cx + 34

    if show_surface_charges:
        add_charge_row(
            b,
            x0,
            x1,
            diel_y + geom.diel_thickness + geom.ry - 6,
            "−",
            neg_charge,
            count=5,
            size=10,
        )
        add_charge_row(
            b,
            x0,
            x1,
            bottom_y + geom.ry + 4,
            "+",
            pos_charge,
            count=5,
            size=10,
        )

def add_delta_v_label(
    b: DrawioBuilder, x: float, y: float, value: str, align: str = "left"
) -> None:
    b.add_text(
        x,
        y,
        72,
        16,
        value,
        size=11,
        italic=True,
        align=align,
        color="#374151",
    )


def add_external_circuit(
    b: DrawioBuilder,
    x: float,
    y: float,
    geom: TENGGeometry,
) -> None:
    cx = x + 88
    top_face_y = y + 48
    diel_y = top_face_y + geom.top_thickness + 6
    bottom_y = diel_y + geom.diel_thickness + geom.gap + 20

    top_anchor = (cx + geom.rx - 2, top_face_y + geom.ry + 2)
    bottom_anchor = (cx + geom.rx, bottom_y + geom.ry)
    load_x = x + 153
    load_y = y + 87
    load_w = 11
    load_h = 27
    load_top = (load_x + 7, load_y + 2)
    load_bottom = (load_x + 2, load_y + load_h - 2)
    bend_x = load_x + 19

    b.add_edge(
        edge_style(stroke="#475569", width=1.8, end="none"),
        source_point=top_anchor,
        target_point=load_top,
        points=[(bend_x, top_anchor[1]), (bend_x, load_top[1])],
    )
    b.add_edge(
        edge_style(stroke="#475569", width=1.8, end="none"),
        source_point=bottom_anchor,
        target_point=load_bottom,
        points=[(bend_x, bottom_anchor[1]), (bend_x, load_bottom[1])],
    )
    b.add_vertex(
        load_x,
        load_y,
        load_w,
        load_h,
        "rounded=1;arcSize=6;html=1;whiteSpace=wrap;fillColor=#FFFFFF;strokeColor=#475569;"
        "strokeWidth=1.1;fontColor=#111827;fontSize=6;fontFamily=Arial;textDirection=vertical-lr;"
        "autosizeText=1;align=left;",
        "Load",
    )

    # Electron flow direction: lower electrode -> load -> upper electrode.
    b.add_edge(
        edge_style(stroke="#475569", width=2.2, end="blockThin"),
        source_point=(387, bottom_anchor[1]),
        target_point=(397, bottom_anchor[1]),
    )
    b.add_edge(
        edge_style(stroke="#475569", width=2.2, end="blockThin"),
        source_point=(396, top_anchor[1]),
        target_point=(386, top_anchor[1]),
    )
    b.add_text(
        x + 144,
        y + 42,
        34,
        18,
        "e<sup>-</sup> flow",
        size=9,
        italic=True,
        align="left",
        color="#374151",
    )


def add_teng_layer_legend(b: DrawioBuilder, x: float, y: float) -> None:
    top_fill = "#F3F6FA"
    diel_fill = "#D9E9F5"
    bot_fill = "#E7ECF2"

    b.add_vertex(
        x,
        y,
        328,
        50,
        rect_style("#FFFFFF", stroke="#D7E1EB", rounded=1, arc=12, width=1.0),
    )
    items = [
        (top_fill, "#7B8794", "Top electrode", x + 26, y + 14),
        (diel_fill, "#6E9FBE", "Dielectric layer", x + 182, y + 14),
        (bot_fill, "#7B8794", "Bottom electrode", x + 26, y + 30),
    ]
    for fill, stroke, label, xx, yy in items:
        b.add_vertex(
            xx,
            yy,
            16,
            10,
            rect_style(fill, stroke=stroke, rounded=1, arc=4, width=1.0),
        )
        b.add_text(
            xx + 24,
            yy - 5,
            124,
            20,
            label,
            size=11,
            align="left",
            color="#374151",
        )

    b.add_edge(
        edge_style(stroke="#C5D1DD", width=1.0, end="none"),
        source_point=(x + 184, y + 31),
        target_point=(x + 200, y + 31),
    )
    b.add_edge(
        edge_style(stroke="#C5D1DD", width=1.0, end="none"),
        source_point=(x + 184, y + 37),
        target_point=(x + 200, y + 37),
    )
    b.add_text(
        x + 208,
        y + 21,
        92,
        20,
        "Air gap",
        size=11,
        align="left",
        color="#374151",
    )


def add_teng_panel(
    b: DrawioBuilder, x: float, y: float, assets: dict[str, Path]
) -> None:
    b.add_vertex(
        x,
        y,
        360,
        560,
        rect_style(
            "#F7FAFD",
            stroke="#DCE6F0",
            gradient="#EDF3F8",
            rounded=1,
            arc=18,
            width=1.0,
        ),
    )
    b.add_text(
        x + 14,
        y + 14,
        332,
        34,
        "Disk TENG geometry & states",
        size=22,
        bold=True,
        align="center",
    )

    geom = TENGGeometry(
        n_sectors=8,
        rx=56,
        ry=16,
        hub_rx=11,
        hub_ry=4,
        top_thickness=10,
        diel_thickness=10,
        gap=18,
        bottom_thickness=10,
    )

    add_teng_state(
        b,
        x + 18,
        y + 88,
        "Maximum overlap",
        geom,
        separated=False,
        aligned_height=True,
        show_layer_labels=False,
        show_geometry_annotations=True,
        show_surface_charges=True,
        title_size=12,
    )
    add_delta_v_label(b, x + 70, y + 248, "low ΔV", align="center")

    add_teng_state(
        b,
        x + 182,
        y + 88,
        "Maximum misalignment",
        geom,
        separated=True,
        aligned_height=True,
        show_layer_labels=False,
        show_geometry_annotations=False,
        show_surface_charges=True,
        title_size=12,
    )
    add_external_circuit(b, x + 182, y + 88, geom)
    add_delta_v_label(b, x + 234, y + 248, "high ΔV", align="center")

    b.add_edge(
        edge_style(stroke="#94A3B8", width=2.0, end="blockThin"),
        source_point=(x + 170, y + 222),
        target_point=(x + 192, y + 222),
    )

    add_teng_layer_legend(b, x + 10, y + 370)

    if assets.get("comsol") and assets["comsol"].exists():
        thumb_style = rect_style(
            "#FFFFFF", stroke="#CBD5E1", rounded=1, arc=12, width=1.0
        )
        b.add_text(
            x + 96,
            y + 438,
            168,
            16,
            "Representative FEM map",
            size=12,
            bold=True,
            align="center",
            color="#374151",
        )
        b.add_vertex(x + 106, y + 456, 148, 52, thumb_style)
        b.add_image(x + 112, y + 460, 136, 44, assets["comsol"])


def add_encoder_watermark(
    b: DrawioBuilder, x: float, y: float, w: float, h: float
) -> None:
    gear_cx = x + w * 0.72
    gear_cy = y + h * 0.56
    outer = 84
    inner = 34

    b.add_vertex(
        gear_cx - outer / 2,
        gear_cy - outer / 2,
        outer,
        outer,
        ellipse_style("#E7EEF6", stroke="#C6D2DF", width=1.0, opacity=12),
    )
    b.add_vertex(
        gear_cx - inner / 2,
        gear_cy - inner / 2,
        inner,
        inner,
        ellipse_style("#F3F6FA", stroke="#C6D2DF", width=0.9, opacity=18),
    )

    teeth = [
        (0, -50, 12, 18),
        (35, -35, 18, 12),
        (50, 0, 12, 18),
        (35, 35, 18, 12),
        (0, 50, 12, 18),
        (-35, 35, 18, 12),
        (-50, 0, 12, 18),
        (-35, -35, 18, 12),
    ]
    for dx, dy, tw, th in teeth:
        b.add_vertex(
            gear_cx + dx - tw / 2,
            gear_cy + dy - th / 2,
            tw,
            th,
            rect_style("#DCE6F1", stroke="#C6D2DF", rounded=1, arc=30, width=0.8, opacity=12),
        )


def add_balance_icon(b: DrawioBuilder, x: float, y: float) -> None:
    stroke = "#6B7280"
    width = 1.1
    b.add_edge(
        edge_style(stroke=stroke, width=width, end="none"),
        source_point=(x + 12, y + 4),
        target_point=(x + 12, y + 21),
    )
    b.add_edge(
        edge_style(stroke=stroke, width=width, end="none"),
        source_point=(x + 2, y + 9),
        target_point=(x + 22, y + 9),
    )
    b.add_edge(
        edge_style(stroke=stroke, width=width, end="none"),
        source_point=(x + 6, y + 9),
        target_point=(x + 6, y + 15),
    )
    b.add_edge(
        edge_style(stroke=stroke, width=width, end="none"),
        source_point=(x + 18, y + 9),
        target_point=(x + 18, y + 15),
    )
    b.add_edge(
        edge_style(stroke=stroke, width=width, end="none"),
        source_point=(x + 2, y + 16),
        target_point=(x + 10, y + 16),
    )
    b.add_edge(
        edge_style(stroke=stroke, width=width, end="none"),
        source_point=(x + 14, y + 16),
        target_point=(x + 22, y + 16),
    )
    b.add_edge(
        edge_style(stroke=stroke, width=width, end="none"),
        source_point=(x + 8, y + 22),
        target_point=(x + 16, y + 22),
    )


def add_center_panel(b: DrawioBuilder, x: float, y: float) -> None:
    pw, ph = 580, 560
    b.add_vertex(
        x, y, pw, ph,
        rect_style("#F7FAFD", stroke="#DCE6F0", gradient="#EDF3F8",
                    rounded=1, arc=18, width=1.0),
    )
    b.add_text(x + 34, y + 14, 512, 32,
               "Physics-consistent multitask surrogate", size=22, bold=True)

    # =====================================================================
    # LAYER 1: Data inputs
    # =====================================================================
    b.add_text(x + 36, y + 58, 110, 18, "Data inputs",
               size=14, bold=True, align="left", color="#374151")
    token_y = y + 82
    token_xs = [x + 82, x + 192, x + 302, x + 412]
    token_labels = ["n", "h/R", "d/R", "ε"]
    token_w, token_h = 64, 24
    for xx, label in zip(token_xs, token_labels):
        b.add_vertex(
            xx,
            token_y,
            token_w,
            token_h,
            rect_style("#EDF3F9", stroke="#A8B7C8", rounded=1, arc=10, width=1.0),
            value=f"<span style='font-size: 13px; font-style: italic;'>{label}</span>",
        )

    # Downstream layout constants reused by the encoder and task-head layers.
    qsc_color = "#E46E49"
    cap_color = "#4D86B3"
    direct_color = "#7A5C99"
    direct_x = x + 85
    direct_y = y + 342
    direct_w = 146
    direct_h = 28
    node_w = 60
    node_h = 26
    q_x = x + 334
    q_y = y + 342
    c_x = x + 446
    c_y = y + 339
    formula_x = x + 334
    formula_y = y + 395
    formula_w = 171
    formula_h = 46
    phys_x = x + 349.04
    phys_y = y + 456
    phys_w = 148
    phys_h = 28
    cons_x = x + 170
    cons_y = y + 517
    cons_w = 240
    cons_h = 32

    # =====================================================================
    # LAYER 2: Shared encoder
    # =====================================================================
    enc_x = x + 74
    enc_y = y + 126
    enc_w = 432
    enc_h = 158
    b.add_vertex(enc_x, enc_y, enc_w, enc_h,
                 rect_style("#F3F6FA", stroke="#94A3B8",
                            rounded=1, arc=12, width=1.6, opacity=82))
    add_encoder_watermark(b, enc_x, enc_y, enc_w, enc_h)
    b.add_text(
        enc_x + 16,
        enc_y + 10,
        170,
        20,
        "Shared encoder",
        size=17,
        bold=True,
        align="left",
        color="#1F2937",
    )
    b.add_text(
        enc_x + 16,
        enc_y + 30,
        130,
        14,
        "Transformer-based",
        size=11,
        italic=True,
        align="left",
        color="#64748B",
    )

    inp_w = 22
    inp_h = 10
    inp_y = enc_y + 56
    inp_centers = [xx + token_w / 2 for xx in token_xs]
    for cx in inp_centers:
        b.add_vertex(
            cx - inp_w / 2,
            inp_y,
            inp_w,
            inp_h,
            rect_style("#E7EEF6", stroke="#A8B7C8", rounded=1, arc=24, width=1.0),
        )

    fuse_x = enc_x + 140
    fuse_y = enc_y + 86
    fuse_w = 152
    fuse_h = 38
    b.add_vertex(fuse_x, fuse_y, fuse_w, fuse_h,
                 rect_style("#D5DCEA", stroke="#8B9DC3",
                            rounded=1, arc=26, width=1.2))
    b.add_text(fuse_x + 4, fuse_y + 9, fuse_w - 8, 20,
               "Self-attention fusion",
               size=10, bold=True, align="center", color="#4A5568")

    latent_centers = [direct_x + direct_w / 2, q_x + node_w / 2, c_x + node_w / 2]
    latent_y = enc_y + 118
    latent_r = 18
    for cx in latent_centers:
        b.add_vertex(
            cx - latent_r / 2,
            latent_y,
            latent_r,
            latent_r,
            ellipse_style("#7F8FA6", stroke="#5C6E82", width=1.0),
        )

    # Inputs -> encoder
    for xx, cx in zip(token_xs, inp_centers):
        b.add_edge(
            edge_style(stroke="#8FA6B9", width=1.0, end="none"),
            source_point=(xx + token_w / 2, token_y + token_h),
            target_point=(cx, inp_y + inp_h / 2),
        )

    fusion_targets = [fuse_x + 18, fuse_x + 52, fuse_x + 96, fuse_x + 130]
    for cx, tx in zip(inp_centers, fusion_targets):
        b.add_edge(
            edge_style(stroke="#CDD6DE", width=0.8, end="none"),
            source_point=(cx, inp_y + inp_h / 2),
            target_point=(tx, fuse_y),
            points=[(cx, fuse_y - 12), (tx, fuse_y - 12)],
        )

    for cx in latent_centers:
        b.add_edge(
            edge_style(stroke="#C2CCD7", width=0.9, end="none"),
            source_point=(fuse_x + fuse_w / 2, fuse_y + fuse_h),
            target_point=(cx, latent_y + latent_r / 2),
            points=[(fuse_x + fuse_w / 2, latent_y - 12), (cx, latent_y - 12)],
        )

    b.add_text(enc_x + 140, enc_y + enc_h - 20, 136, 16,
               "Feed-forward ×2", size=11, align="center", color="#64748B")

    # =====================================================================
    # LAYER 3: Split into data and physics branches
    # =====================================================================
    b.add_text(x + 36, y + 288, 100, 18, "Task heads",
               size=14, bold=True, align="left", color="#374151")
    b.add_text(
        x + 92,
        y + 314,
        132,
        14,
        "Data-driven branch",
        size=10,
        bold=True,
        align="center",
        color="#6D5A8E",
    )
    b.add_text(
        x + 366,
        y + 314,
        132,
        14,
        "Physics branch",
        size=10,
        bold=True,
        align="center",
        color="#355E86",
    )

    direct_id = b.add_vertex(
        direct_x,
        direct_y,
        direct_w,
        direct_h,
        rect_style("#EFE8F7", stroke="#7A5C99", rounded=1, arc=22, width=1.3),
        value="<span style='color: rgb(122, 92, 153); font-size: 12px; font-style: italic;'>FOMS_direct</span>",
    )
    q_id = b.add_vertex(
        q_x,
        q_y,
        node_w,
        node_h,
        rect_style("#FBE9E3", stroke="#D36A48", rounded=1, arc=20, width=1.2),
        value="<span style='color: rgb(228, 110, 73); font-size: 12px; font-style: italic;'>Q</span><sub style='color: rgb(228, 110, 73); font-style: italic;'>sc</sub>",
    )
    c_id = b.add_vertex(
        c_x,
        c_y,
        node_w,
        node_h,
        rect_style("#E6F0F9", stroke="#5C88B3", rounded=1, arc=20, width=1.2),
        value="<span style='color: rgb(77, 134, 179); font-size: 12px; font-style: italic;'>C</span><sup style='color: rgb(77, 134, 179); font-style: italic;'>−1</sup>",
    )
    formula_id = b.add_vertex(
        formula_x,
        formula_y,
        formula_w,
        formula_h,
        rect_style("#DCEEFF", stroke="#5C88B3", rounded=1, arc=12, width=2.0),
        value="<span style='color: rgb(53, 94, 134); font-size: 10px; font-weight: 700;'>Physical reconstruction<br><span style='color: rgb(36, 76, 115); font-size: 13px;'>FOM ∝ Q² × C⁻¹</span></span>",
    )
    phys_id = b.add_vertex(
        phys_x,
        phys_y,
        148,
        28,
        rect_style("#E1EDF8", stroke="#355E86", rounded=1, arc=22, width=1.3),
        value="<span style='color: rgb(53, 94, 134); font-size: 12px; font-style: italic;'>FOMS_phys</span>",
    )
    cons_id = b.add_vertex(
        cons_x,
        cons_y,
        cons_w,
        cons_h,
        rect_style("#F3F5F7", stroke="#9AA4AF", rounded=1, arc=16, width=1.2),
        value="<span style='color: rgb(75, 85, 99); font-size: 10px; font-weight: 700;'>L</span><sub style='color: rgb(75, 85, 99); font-weight: 700;'>consistency&nbsp;</sub><span style='color: rgb(75, 85, 99); font-size: 11px; font-style: italic;'>FOMS_direct ≈ FOMS_phys</span>",
    )

    b.add_edge(
        edge_style(stroke="#BCC6D1", width=1.1, end="blockThin"),
        source_point=(latent_centers[0], latent_y + latent_r),
        target_point=(direct_x + direct_w / 2, direct_y),
    )
    b.add_edge(
        edge_style(stroke="#BCC6D1", width=1.0, end="blockThin"),
        source_point=(latent_centers[1], latent_y + latent_r),
        target_point=(q_x + node_w / 2, q_y),
    )
    b.add_edge(
        edge_style(stroke="#BCC6D1", width=1.0, end="blockThin"),
        source_point=(latent_centers[2], latent_y + latent_r),
        target_point=(c_x + node_w / 2, c_y),
    )
    b.add_edge(
        "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#E46E49;strokeWidth=1.9;"
        "endArrow=blockThin;endFill=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
        "entryX=0.25;entryY=0;entryDx=0;entryDy=0;",
        source=q_id,
        target=formula_id,
    )
    b.add_edge(
        "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#4D86B3;strokeWidth=1.9;"
        "endArrow=blockThin;endFill=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
        "entryX=0.75;entryY=0;entryDx=0;entryDy=0;",
        source=c_id,
        target=formula_id,
    )
    b.add_edge(
        "html=1;strokeColor=#4A79A8;strokeWidth=2.0;endArrow=blockThin;endFill=1;"
        "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.482;entryY=0.043;"
        "entryDx=0;entryDy=0;entryPerimeter=0;",
        source=formula_id,
        target=phys_id,
    )
    b.add_edge(
        "edgeStyle=none;html=1;strokeColor=#8A94A3;strokeWidth=1.3;"
        "endArrow=blockThin;endFill=1;startArrow=blockThin;startFill=1;"
        "dashed=1;dashPattern=6 4;exitX=0.5;exitY=1;exitDx=0;exitDy=0;",
        source=direct_id,
        source_point=(direct_x + direct_w / 2, direct_y + direct_h),
        target_point=(x + 212, y + 514),
        points=[(direct_x + direct_w / 2, y + 502), (x + 212, y + 502)],
    )
    b.add_edge(
        "edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#8A94A3;strokeWidth=1.3;"
        "endArrow=blockThin;endFill=1;startArrow=blockThin;startFill=1;"
        "dashed=1;dashPattern=6 4;exitX=0.44;exitY=1.029;exitDx=0;exitDy=0;"
        "exitPerimeter=0;entryX=0.795;entryY=-0.018;entryDx=0;entryDy=0;entryPerimeter=0;",
        source=phys_id,
        target=cons_id,
    )


def add_outcomes_panel(
    b: DrawioBuilder, x: float, y: float, assets: dict[str, Path]
) -> None:
    b.add_vertex(
        x,
        y,
        236,
        560,
        rect_style(
            "#F7FAFD",
            stroke="#DCE6F0",
            gradient="#EDF3F8",
            rounded=1,
            arc=18,
            width=1.0,
        ),
    )
    b.add_text(
        x + 12, y + 16, 212, 34, "Mechanism-aware design outputs", size=20, bold=True
    )

    card = rect_style("#FBFCFE", stroke="#C9D6E0", rounded=1, arc=14, width=1.0)
    card_x = x + 31
    card_w = 174
    card_h = 106
    img_w = 144
    img_h = 96
    img_x = card_x + (card_w - img_w) / 2

    sections = [
        ("Mechanism landscape", y + 126, y + 146, "icon_mechanism_landscape"),
        ("Design window", y + 282, y + 302, "icon_design_window"),
        ("Safe region", y + 438, y + 458, "icon_safe_region"),
    ]
    for title, title_y, card_y, icon_key in sections:
        b.add_text(x + 14, title_y, 208, 20, title, size=15)
        b.add_vertex(card_x, card_y, card_w, card_h, card)
        if assets.get(icon_key) and assets[icon_key].exists():
            b.add_image(img_x, card_y + 5, img_w, img_h, assets[icon_key])


def add_interpanel_arrow(
    b: DrawioBuilder, x1: float, y1: float, x2: float, y2: float
) -> None:
    b.add_edge(
        edge_style(stroke="#D6E1EA", width=18, end="blockThin", opacity=58),
        source_point=(x1, y1),
        target_point=(x2, y2),
    )


def add_bottom_strip(
    b: DrawioBuilder, x: float, y: float, assets: dict[str, Path]
) -> None:
    title_w = 180.0
    card_w = 180.0
    card_h = 82.0
    card_y = y + 30.0
    step = 256.0
    xs = [x + step * i for i in range(6)]

    for idx, (title, xx) in enumerate(zip(PANEL_B_TITLES, xs)):
        b.add_text(xx, y, title_w, 24, title, size=15, align="center")
        if idx < len(xs) - 1:
            b.add_edge(
                edge_style(stroke="#334155", width=1.4, end="block"),
                source_point=(xx + card_w, card_y + card_h / 2),
                target_point=(xs[idx + 1], card_y + card_h / 2),
            )

    icon_w = 96.0
    icon_h = 72.0
    icon_y = card_y + (card_h - icon_h) / 2
    for xx, icon_key in zip(xs, PANEL_B_ICON_KEYS):
        icon_path = assets.get(icon_key)
        if icon_path and icon_path.exists():
            b.add_image(xx + (card_w - icon_w) / 2, icon_y, icon_w, icon_h, icon_path)


def build_from_backup() -> str:
    return load_baseline_xml()


def apply_a_panel_layout_override(xml: str) -> str:
    root = ET.fromstring(xml)
    model = root.find(".//mxGraphModel")
    mxroot = root.find(".//root")
    if mxroot is None or model is None:
        return xml

    cells_by_id = {cell.attrib.get("id"): cell for cell in mxroot.findall("mxCell")}

    panel_positions = {
        "8RxiAfdrNtuPFcRHXn3M-217": (60.0, 60.0, 400.0, 622.0),   # left panel, isotropic scale from backup
        "8RxiAfdrNtuPFcRHXn3M-216": (520.0, 60.0, 820.0, 820.0),  # center panel
        "8RxiAfdrNtuPFcRHXn3M-215": (1400.0, 60.0, 340.0, 800.0), # right panel
    }
    scale_map: dict[str, tuple[float, float]] = {}
    for cell_id, (_, _, new_w, new_h) in panel_positions.items():
        cell = cells_by_id.get(cell_id)
        if cell is None:
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        old_w = float(geom.attrib.get("width", new_w))
        old_h = float(geom.attrib.get("height", new_h))
        if cell_id == "8RxiAfdrNtuPFcRHXn3M-217":
            s = new_w / old_w if old_w else 1.0
            scale_map[cell_id] = (s, s)
        else:
            scale_map[cell_id] = (
                new_w / old_w if old_w else 1.0,
                new_h / old_h if old_h else 1.0,
            )

    def top_group_id(cell_id: str) -> str | None:
        current = cell_id
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            cell = cells_by_id.get(current)
            if cell is None:
                return None
            parent = cell.attrib.get("parent")
            if parent in panel_positions:
                return parent
            current = parent or ""
        return None

    for cell in mxroot.findall("mxCell"):
        cid = cell.attrib.get("id", "")
        if cid in panel_positions:
            continue
        top_id = top_group_id(cid)
        if top_id is None:
            continue
        sx, sy = scale_map[top_id]
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        for attr in ("x", "width"):
            if attr in geom.attrib:
                geom.set(attr, fmt(float(geom.attrib[attr]) * sx))
        for attr in ("y", "height"):
            if attr in geom.attrib:
                geom.set(attr, fmt(float(geom.attrib[attr]) * sy))
        for pt in geom.findall(".//mxPoint"):
            if "x" in pt.attrib:
                pt.set("x", fmt(float(pt.attrib["x"]) * sx))
            if "y" in pt.attrib:
                pt.set("y", fmt(float(pt.attrib["y"]) * sy))

    for cell_id, (x, y, w, h) in panel_positions.items():
        cell = cells_by_id.get(cell_id)
        if cell is None:
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        geom.set("x", fmt(x))
        geom.set("y", fmt(y))
        geom.set("width", fmt(w))
        geom.set("height", fmt(h))

    geometry_overrides = {
        "4": {"x": 14.0, "width": 370.0},
        "96": {"x": 0.0, "y": 0.0, "width": 820.0, "height": 820.0},
        "97": {"x": 20.0, "width": 780.0},
        "155": {"width": 340.0, "height": 800.0},
        "156": {"x": 16.0, "width": 308.0},
    }
    for cell_id, attrs in geometry_overrides.items():
        cell = cells_by_id.get(cell_id)
        if cell is None:
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        for key, value in attrs.items():
            geom.set(key, fmt(value))

    arrow_specs = {
        "95": ((470.0, 371.0), (510.0, 371.0)),
        "154": ((1352.0, 460.0), (1392.0, 460.0)),
    }
    for cell_id, (source_point, target_point) in arrow_specs.items():
        cell = cells_by_id.get(cell_id)
        if cell is None:
            continue
        cell.set(
            "style",
            "edgeStyle=none;html=1;strokeColor=#CBD5E1;strokeWidth=4;endArrow=blockThin;endFill=1;opacity=38;",
        )
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        for tag, point in (("sourcePoint", source_point), ("targetPoint", target_point)):
            pt = geom.find(f"mxPoint[@as='{tag}']")
            if pt is None:
                pt = ET.SubElement(geom, "mxPoint", {"as": tag})
            pt.set("x", fmt(point[0]))
            pt.set("y", fmt(point[1]))

    for cell_id in ("159", "162"):
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            mxroot.remove(cell)

    model.set("pageWidth", "1800")
    model.set("pageHeight", "960")
    return ET.tostring(root, encoding="unicode")


def apply_b_panel_layout_override(xml: str, assets: dict[str, Path]) -> str:
    root = ET.fromstring(xml)
    mxroot = root.find(".//root")
    if mxroot is None:
        return xml

    cells = list(mxroot.findall("mxCell"))
    for cell in cells:
        cid = cell.attrib.get("id", "")
        if cid.startswith("bstrip-"):
            mxroot.remove(cell)
            continue
        if cid.isdigit() and 170 <= int(cid) <= 206:
            mxroot.remove(cell)

    title_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontSize=15;fontStyle=0;fontFamily=Arial;"
        "fontColor=#111827;whiteSpace=wrap;"
    )
    small_text_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontSize=12;fontStyle=0;fontFamily=Arial;"
        "fontColor=#334155;whiteSpace=wrap;"
    )
    formula_text_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontSize=11;fontStyle=0;fontFamily=Arial;"
        "fontColor=#243B53;whiteSpace=wrap;"
    )
    box_style = rect_style("#F8FBFD", stroke="#CBD5E1", rounded=1, arc=10, width=1.0)
    card_style = rect_style("#FFFFFF", stroke="#D7E1EA", rounded=1, arc=8, width=0.9)
    capsule_style = rect_style("#F5F8FB", stroke="#C5D1DB", rounded=1, arc=14, width=0.9)
    check_style = ellipse_style("#E8F7EE", stroke="#68A27D", width=1.0)
    q_capsule_style = (
        "rounded=1;arcSize=20;html=1;whiteSpace=wrap;fillColor=#FBE9E3;"
        "strokeColor=#D36A48;strokeWidth=1.2;fontColor=#111827;fontSize=14;fontFamily=Arial;"
    )
    c_capsule_style = (
        "rounded=1;arcSize=20;html=1;whiteSpace=wrap;fillColor=#E6F0F9;"
        "strokeColor=#5C88B3;strokeWidth=1.2;fontColor=#111827;fontSize=14;fontFamily=Arial;"
    )
    fom_style = (
        "rounded=1;arcSize=22;html=1;whiteSpace=wrap;fillColor=#EFE8F7;"
        "strokeColor=#7A5C99;strokeWidth=1.3;fontColor=#111827;fontSize=14;fontFamily=Arial;"
    )

    card_y = 684.0
    title_w = 180.0
    card_w = 180.0
    card_h = 82.0
    step = 256.0
    xs = [70.0 + step * i for i in range(6)]
    append_root_vertex(
        mxroot,
        "bstrip-panel-bg",
        50,
        674,
        1488,
        112,
        "rounded=1;arcSize=10;html=1;whiteSpace=wrap;fillColor=#FAFAFA;strokeColor=none;strokeWidth=0;",
    )
    title_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontSize=8;fontStyle=0;fontFamily=Arial;"
        "fontColor=#374151;whiteSpace=wrap;"
    )
    icon_w = 112.0
    icon_h = 78.0
    icon_y = 682.0
    title_y = icon_y + icon_h + 8
    positions: list[tuple[float, float]] = []

    for idx, xx in enumerate(xs, start=1):
        ix = xx + (card_w - icon_w) / 2
        positions.append((ix, icon_y))
        append_root_vertex(
            mxroot,
            f"bstrip-title-{idx}",
            xx,
            title_y,
            title_w,
            10,
            title_style,
            PANEL_B_TITLES[idx - 1],
        )

    for idx in range(5):
        x1 = positions[idx][0] + icon_w + 6
        x2 = positions[idx + 1][0] - 6
        y = icon_y + icon_h / 2
        append_root_edge(
            mxroot,
            f"bstrip-arrow-{idx+1}",
            "edgeStyle=none;html=1;strokeColor=#94A3B8;strokeWidth=0.8;endArrow=open;endFill=0;",
            (x1, y),
            (x2, y),
        )

    for idx, icon_key in enumerate(PANEL_B_ICON_KEYS, start=1):
        ix, iy = positions[idx - 1]
        icon_path = assets.get(icon_key)
        if icon_path and icon_path.exists():
            append_root_vertex(
                mxroot,
                f"bstrip-icon-{idx}",
                ix,
                iy,
                icon_w,
                icon_h,
                image_style_from_path(icon_path),
            )

    return ET.tostring(root, encoding="unicode")


def apply_only_panel_a_layout(xml: str) -> str:
    root = ET.fromstring(xml)
    model = root.find(".//mxGraphModel")
    mxroot = root.find(".//root")
    if model is None or mxroot is None:
        return xml

    for cell in list(mxroot.findall("mxCell")):
        cid = cell.attrib.get("id", "")
        if cid == "2":
            mxroot.remove(cell)
            continue
        if cid.startswith("bstrip-"):
            mxroot.remove(cell)
            continue
        if cid.isdigit() and 169 <= int(cid) <= 206:
            mxroot.remove(cell)

    model.set("pageWidth", "1800")
    model.set("pageHeight", "960")
    return ET.tostring(root, encoding="unicode")


def apply_panel_a_text_replacements(xml: str) -> str:
    root = ET.fromstring(xml)
    mxroot = root.find(".//root")
    if mxroot is None:
        return xml

    cells_by_id = {cell.attrib.get("id"): cell for cell in mxroot.findall("mxCell")}
    replacements = {
        "4": "Disk TENG geometry and operating states",
        "97": "Physics-consistent multitask surrogate",
        "98": "Four geometric inputs",
        "114": "Shared transformer backbone",
        "120": "Self-attention encoding",
        "137": "Prediction heads",
        "140": (
            '<span style="color: rgb(122, 92, 153); font-size: 12px; font-style: italic;">FOM</span>'
            '<span style="color: rgb(122, 92, 153); font-style: italic; font-size: 11px;"><sub>direct</sub></span>'
        ),
        "141": (
            '<span style="color: rgb(228, 110, 73); font-size: 12px; font-style: italic;">Q</span>'
            '<span style="color: rgb(228, 110, 73); font-style: italic; font-size: 11px;"><sub>sc</sub></span>'
        ),
        "142": (
            '<span style="color: rgb(77, 134, 179); font-size: 12px; font-style: italic;">C</span>'
            '<span style="color: rgb(77, 134, 179); font-style: italic; font-size: 11px;"><sub>sum</sub></span>'
            '<span style="color: rgb(77, 134, 179); font-style: italic; font-size: 12px;"><sup>-1</sup></span>'
        ),
        "143": (
            '<span style="color: rgb(53, 94, 134); font-size: 10px; font-weight: 700;">Physics-based reconstruction'
            '<br><span style="color: rgb(36, 76, 115); font-size: 13px;">FOM<sub>phys</sub> ∝ '
            'Q<sub>sc</sub><sup>2</sup> × C<sub>sum</sub><sup>-1</sup></span></span>'
        ),
        "144": (
            '<i><font face="Arial" style="color: rgb(53, 94, 134); font-size: 12px;">'
            'FOM<sub>phys</sub></font></i>'
        ),
        "145": (
            '<span style="color: rgb(75, 85, 99); font-weight: 700;">Consistency loss: '
            'L<sub>cons</sub></span><br><span style="color: rgb(75, 85, 99); font-size: 11px;">'
            '(FOM<sub>direct</sub>, FOM<sub>phys</sub>)</span>'
        ),
        "156": "Mechanism-informed design outputs",
        "jp7RGeU-hvWb_L_M1MVX-218": "Design window",
        "jp7RGeU-hvWb_L_M1MVX-223": "Robust design region",
    }
    for cell_id, value in replacements.items():
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            cell.set("value", value)

    geometry_overrides = {
        "98": {"width": 182.0},
        "137": {"width": 124.0},
        "143": {"x": 300.0, "width": 182.0},
        "145": {"x": 145.0, "width": 280.0, "height": 38.0},
        "156": {"y": 14.0, "height": 42.0},
    }
    for cell_id, attrs in geometry_overrides.items():
        cell = cells_by_id.get(cell_id)
        if cell is None:
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        for key, value in attrs.items():
            geom.set(key, fmt(value))

    return ET.tostring(root, encoding="unicode")


def apply_left_panel_micro_tweak(xml: str, assets: dict[str, Path] | None = None) -> str:
    root = ET.fromstring(xml)
    mxroot = root.find(".//root")
    if mxroot is None:
        return xml

    for cell in list(mxroot.findall("mxCell")):
        cid = cell.attrib.get("id", "")
        if cid in {"lp-transition-arrow", "lp-transition-label", "lp-fem-start", "lp-fem-end"}:
            mxroot.remove(cell)

    cells_by_id = {cell.attrib.get("id"): cell for cell in mxroot.findall("mxCell")}

    geometry_overrides = {
        "5JC-8PAxv0zjpvQebs5j-217": {"x": 166.78, "y": 160.0, "width": 184.44},
        "5JC-8PAxv0zjpvQebs5j-218": {"x": 166.78, "y": 382.0, "width": 184.44},
        "5JC-8PAxv0zjpvQebs5j-215": {"x": 11.11, "width": 162.22},
        "43": {"x": 5.55, "width": 173.33},
        "5JC-8PAxv0zjpvQebs5j-216": {"x": 90.0, "y": 574.0, "width": 368.0},
        "92": {"x": 146.0, "y": 676.0, "width": 256.0, "height": 18.0},
        "93": {"x": 214.0, "y": 700.0, "width": 120.0, "height": 120.0},
    }
    for cell_id, attrs in geometry_overrides.items():
        cell = cells_by_id.get(cell_id)
        if cell is None:
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        for key, value in attrs.items():
            geom.set(key, fmt(value))

    start_path = assets.get("fem_start") if assets else None
    end_path = assets.get("fem_end") if assets else None
    box = cells_by_id.get("93")
    box_geom = box.find("mxGeometry") if box is not None else None
    if box_geom is not None:
        box_x = float(box_geom.attrib.get("x", "214"))
        box_y = float(box_geom.attrib.get("y", "700"))
        box_w = float(box_geom.attrib.get("width", "120"))
        box_h = float(box_geom.attrib.get("height", "120"))
        pad_x = 6.0
        pad_y = 8.0
        gap_x = 4.0
        thumb_w = (box_w - 2 * pad_x - gap_x) / 2
        thumb_h = box_h - 2 * pad_y
        if start_path and start_path.exists():
            append_root_vertex(
                mxroot,
                "lp-fem-start",
                box_x + pad_x,
                box_y + pad_y,
                thumb_w,
                thumb_h,
                image_style_from_path(start_path),
            )
        if end_path and end_path.exists():
            append_root_vertex(
                mxroot,
                "lp-fem-end",
                box_x + pad_x + thumb_w + gap_x,
                box_y + pad_y,
                thumb_w,
                thumb_h,
                image_style_from_path(end_path),
            )

    append_root_edge(
        mxroot,
        "lp-transition-arrow",
        edge_style(stroke="#94A3B8", width=1.8, end="blockThin"),
        (258.0, 334.0),
        (258.0, 376.0),
    )
    append_root_vertex(
        mxroot,
        "lp-transition-label",
        270.0,
        344.0,
        42.0,
        18.0,
        "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
        "fontSize=12;fontStyle=2;fontFamily=Arial;fontColor=#475569;whiteSpace=wrap;",
        "ΔV",
    )

    return ET.tostring(root, encoding="unicode")


def apply_final_text_and_fem_labels(xml: str) -> str:
    root = ET.fromstring(xml)
    mxroot = root.find(".//root")
    if mxroot is None:
        return xml

    for cell in list(mxroot.findall("mxCell")):
        if cell.attrib.get("id") in {"lp-fem-label-start", "lp-fem-label-end"}:
            mxroot.remove(cell)

    cells_by_id = {cell.attrib.get("id"): cell for cell in mxroot.findall("mxCell")}
    replacements = {
        "4": "Disk TENG geometry and representative electrostatic states",
        "98": "Four structural inputs",
        "114": "Shared transformer encoder",
        "92": "Representative electrostatic FEM snapshots",
    }
    for cell_id, value in replacements.items():
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            cell.set("value", value)

    fem_start = cells_by_id.get("lp-fem-start")
    fem_end = cells_by_id.get("lp-fem-end")
    if fem_start is None or fem_end is None:
        return ET.tostring(root, encoding="unicode")

    parent = fem_start.attrib.get("parent", "1")
    start_geom = fem_start.find("mxGeometry")
    end_geom = fem_end.find("mxGeometry")
    if start_geom is None or end_geom is None:
        return ET.tostring(root, encoding="unicode")

    label_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
        "fontSize=10;fontStyle=0;fontFamily=Arial;fontColor=#374151;whiteSpace=wrap;"
    )

    def append_child_text(cell_id: str, x: float, y: float, w: float, h: float, value: str) -> None:
        cell = ET.SubElement(
            mxroot,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": label_style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": fmt(x),
                "y": fmt(y),
                "width": fmt(w),
                "height": fmt(h),
                "as": "geometry",
            },
        )

    sx = float(start_geom.attrib.get("x", "0"))
    sy = float(start_geom.attrib.get("y", "0"))
    sw = float(start_geom.attrib.get("width", "0"))
    sh = float(start_geom.attrib.get("height", "0"))
    ex = float(end_geom.attrib.get("x", "0"))
    ey = float(end_geom.attrib.get("y", "0"))
    ew = float(end_geom.attrib.get("width", "0"))
    eh = float(end_geom.attrib.get("height", "0"))
    label_y = max(sy + sh, ey + eh) + 4.0

    append_child_text("lp-fem-label-start", sx, label_y, sw, 14.0, "Start state")
    append_child_text("lp-fem-label-end", ex, label_y, ew, 14.0, "End state")

    # Lower the positive charges slightly so they sit on the lower-electrode
    # cylindrical surface, matching the user's manual visual intent.
    plus_charge_ids = {"37", "38", "39", "40", "41", "69", "70", "71", "72", "73"}
    for cell_id in plus_charge_ids:
        cell = cells_by_id.get(cell_id)
        if cell is None:
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        y = float(geom.attrib.get("y", "0"))
        geom.set("y", fmt(y + 2.0))

    return ET.tostring(root, encoding="unicode")


def apply_expression_unification(xml: str) -> str:
    root = ET.fromstring(xml)
    mxroot = root.find(".//root")
    if mxroot is None:
        return xml

    cells_by_id = {cell.attrib.get("id"): cell for cell in mxroot.findall("mxCell")}

    replacements = {
        "140": (
            '<span style="color: rgb(122, 92, 153); font-size: 21px; font-style: italic;">FOM</span>'
            '<sub><span style="color: rgb(122, 92, 153); font-size: 17px;">'
            '<i>S</i>,direct</span></sub>'
        ),
        "141": (
            '<span style="color: rgb(228, 110, 73); font-size: 21px; font-style: italic;">Q</span>'
            '<sub><span style="color: rgb(228, 110, 73); font-size: 17px;">'
            '<i>sc</i>,MACRS</span></sub>'
        ),
        "142": (
            '<span style="color: rgb(77, 134, 179); font-size: 21px; font-style: italic;">C</span>'
            '<sup><span style="color: rgb(77, 134, 179); font-size: 17px;">-1</span></sup>'
            '<sub><span style="color: rgb(77, 134, 179); font-size: 17px;">sum</span></sub>'
        ),
        "143": (
            '<span style="color: rgb(53, 94, 134); font-size: 20px; font-weight: 700;">'
            'Physics-based reconstruction<br>'
            '<span style="color: rgb(36, 76, 115); font-size: 20px;">'
            'FOM<sub><i>S</i>,phys</sub> ∝ '
            'Q<sub><i>sc</i>,MACRS</sub><sup>2</sup> × '
            'C<sup>-1</sup><sub>sum</sub>'
            '</span></span>'
        ),
        "144": (
            '<span style="color: rgb(53, 94, 134); font-size: 21px; font-style: italic;">FOM</span>'
            '<sub><span style="color: rgb(53, 94, 134); font-size: 17px;">'
            '<i>S</i>,phys</span></sub>'
        ),
        "145": (
            '<span style="color: rgb(75, 85, 99); font-size: 20px; font-weight: 700;">'
            'Consistency loss: ℒ<sub>cons</sub></span><br>'
            '<span style="color: rgb(75, 85, 99); font-size: 18px;">'
            'MSE(FOM<sub><i>S</i>,direct</sub>, '
            'FOM<sub><i>S</i>,phys</sub>)'
            '</span>'
        ),
    }

    for cell_id, value in replacements.items():
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            cell.set("value", value)

    return ET.tostring(root, encoding="unicode")


def _split_style(style: str) -> tuple[list[str], dict[str, str], list[str]]:
    order: list[str] = []
    mapping: dict[str, str] = {}
    flags: list[str] = []
    for part in [item for item in style.split(";") if item]:
        if "=" not in part:
            flags.append(part)
            continue
        key, value = part.split("=", 1)
        if key not in mapping:
            order.append(key)
        mapping[key] = value
    return order, mapping, flags


def _join_style(order: list[str], mapping: dict[str, str], flags: list[str]) -> str:
    parts: list[str] = []
    for key in order:
        if key in mapping:
            parts.append(f"{key}={mapping[key]}")
    for key, value in mapping.items():
        if key not in order:
            parts.append(f"{key}={value}")
    parts.extend(flags)
    return ";".join(parts) + ";"


def _update_style(
    cell: ET.Element,
    updates: dict[str, str | int | float],
    remove: tuple[str, ...] = (),
) -> None:
    order, mapping, flags = _split_style(cell.attrib.get("style", ""))
    for key in remove:
        mapping.pop(key, None)
    for key, value in updates.items():
        mapping[key] = str(value)
    cell.set("style", _join_style(order, mapping, flags))


def _set_geom(cell: ET.Element, **attrs: float) -> None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return
    for key, value in attrs.items():
        geom.set(key, fmt(value))


def _recenter_geom(
    cell: ET.Element,
    new_w: float | None = None,
    new_h: float | None = None,
) -> None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return
    x = float(geom.attrib.get("x", "0"))
    y = float(geom.attrib.get("y", "0"))
    w = float(geom.attrib.get("width", "0"))
    h = float(geom.attrib.get("height", "0"))
    if new_w is not None:
        x += (w - new_w) / 2.0
        w = new_w
    if new_h is not None:
        y += (h - new_h) / 2.0
        h = new_h
    geom.set("x", fmt(x))
    geom.set("y", fmt(y))
    geom.set("width", fmt(w))
    geom.set("height", fmt(h))


def _set_edge_points(
    cell: ET.Element,
    source: tuple[float, float] | None = None,
    target: tuple[float, float] | None = None,
    points: list[tuple[float, float]] | None = None,
) -> None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return
    if points is not None:
        arr = geom.find("Array")
        if arr is not None:
            geom.remove(arr)
        arr = ET.SubElement(geom, "Array", {"as": "points"})
        for px, py in points:
            ET.SubElement(arr, "mxPoint", {"x": fmt(px), "y": fmt(py)})
    if source is not None:
        src = geom.find("mxPoint[@as='sourcePoint']")
        if src is not None:
            src.set("x", fmt(source[0]))
            src.set("y", fmt(source[1]))
    if target is not None:
        dst = geom.find("mxPoint[@as='targetPoint']")
        if dst is not None:
            dst.set("x", fmt(target[0]))
            dst.set("y", fmt(target[1]))


def apply_publication_overrides(xml: str) -> str:
    root = ET.fromstring(xml)
    model = root.find(".//mxGraphModel")
    mxroot = root.find(".//root")
    if mxroot is None:
        return xml

    cells_by_id = {cell.attrib.get("id"): cell for cell in mxroot.findall("mxCell")}
    preserve_manual_text_ids = {
        "4", "92", "97", "98", "114", "137", "156", "157",
        "jp7RGeU-hvWb_L_M1MVX-218", "jp7RGeU-hvWb_L_M1MVX-223",
        "5", "43", "84", "86", "88", "91",
        "27", "28", "29", "31", "LrGlqFZp42gKuIS671kT-220",
        "76", "79", "99", "100", "101", "102",
        "lp-transition-label", "lp-fem-label-start", "lp-fem-label-end",
        "140", "141", "142", "143", "144", "145",
    }

    def style(cell_id: str, **updates: str | int | float) -> None:
        if cell_id in preserve_manual_text_ids:
            return
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            _update_style(cell, updates)

    def style_remove(
        cell_id: str,
        updates: dict[str, str | int | float],
        remove: tuple[str, ...],
    ) -> None:
        if cell_id in preserve_manual_text_ids:
            return
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            _update_style(cell, updates, remove=remove)

    def geom(cell_id: str, **attrs: float) -> None:
        if cell_id in preserve_manual_text_ids:
            return
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            _set_geom(cell, **attrs)

    def recenter(cell_id: str, new_w: float | None = None, new_h: float | None = None) -> None:
        if cell_id in preserve_manual_text_ids:
            return
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            _recenter_geom(cell, new_w=new_w, new_h=new_h)

    def edge_style(cell_id: str, **updates: str | int | float) -> None:
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            _update_style(cell, updates)

    def edge_points(
        cell_id: str,
        source: tuple[float, float] | None = None,
        target: tuple[float, float] | None = None,
        points: list[tuple[float, float]] | None = None,
    ) -> None:
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            _set_edge_points(cell, source=source, target=target, points=points)

    for cell_id in ("3", "96", "155"):
        style(cell_id, strokeWidth=1.2)

    # Increase the three main panel heights to create a larger manual
    # refinement workspace without changing the current horizontal layout.
    geom("3", height=900.0)
    geom("8RxiAfdrNtuPFcRHXn3M-216", height=900.0)
    geom("96", height=900.0)
    geom("8RxiAfdrNtuPFcRHXn3M-215", height=900.0)
    geom("155", height=900.0)

    style("4", fontSize=28, fontStyle=1)
    style("97", fontSize=30, fontStyle=1)
    style("156", fontSize=28, fontStyle=1)
    geom("4", x=82.0, y=70.0, width=384.0, height=82.0)
    geom("97", y=14.0, height=52.0)
    geom("156", x=18.0, y=12.0, width=304.0, height=70.0)

    for cell_id in ("98", "137", "157", "jp7RGeU-hvWb_L_M1MVX-218", "jp7RGeU-hvWb_L_M1MVX-223"):
        style(cell_id, fontSize=22)
    style("92", fontSize=20, fontStyle=1)
    style("5", fontSize=18, fontStyle=1)
    style("43", fontSize=18, fontStyle=1)
    style("114", fontSize=20, fontStyle=1)
    style("120", fontSize=22, strokeWidth=1.3)
    geom("98", width=250.0, height=32.0)
    geom("137", width=180.0, height=32.0, y=434.0)
    geom("114", width=340.0, height=32.0)
    geom("92", x=128.0, y=690.0, width=300.0, height=28.0)
    geom("5", height=32.0)
    geom("43", x=3.0, width=178.0, height=44.0)
    geom("157", height=32.0)
    geom("jp7RGeU-hvWb_L_M1MVX-218", height=32.0)
    geom("jp7RGeU-hvWb_L_M1MVX-223", height=32.0)

    style("99", fontSize=20)
    style("100", fontSize=20)
    style("101", fontSize=20)
    style("102", fontSize=20)
    for cell_id, value in {
        "99": "<span style='font-size: 20px; font-style: italic;'>n</span>",
        "100": "<span style='font-size: 20px; font-style: italic;'>h/R</span>",
        "101": "<span style='font-size: 20px; font-style: italic;'>d/R</span>",
        "102": "<span style='font-size: 20px; font-style: italic;'>ε</span>",
    }.items():
        cell = cells_by_id.get(cell_id)
        if cell is not None:
            cell.set("value", value)
    for cell_id in ("99", "100", "101", "102"):
        recenter(cell_id, new_w=100.0, new_h=40.0)

    style("82", strokeWidth=1.2)
    geom("5JC-8PAxv0zjpvQebs5j-216", y=592.0, height=74.0)
    geom("82", height=74.0)
    for cell_id in ("84", "86", "88", "91"):
        style(cell_id, fontSize=18)
    geom("84", width=145.0, height=24.0, y=8.0)
    geom("86", width=138.0, height=24.0, y=8.0)
    geom("88", width=160.0, height=24.0, y=40.0)
    geom("91", width=118.0, height=24.0, y=28.0)

    geom("5JC-8PAxv0zjpvQebs5j-218", y=392.0, height=182.0)
    geom("5JC-8PAxv0zjpvQebs5j-215", y=56.0)
    geom("cuh1WUVGdzq4t4RnDBzX-215", y=714.0)

    for cell_id in ("27", "28", "29", "31", "LrGlqFZp42gKuIS671kT-220"):
        style(cell_id, fontSize=18)
    geom("29", width=44.0, height=20.0, y=78.0)
    geom("31", width=44.0, height=20.0, y=109.0)
    geom("LrGlqFZp42gKuIS671kT-220", width=32.0, height=20.0, y=36.0)

    style_remove(
        "76",
        {"fontSize": 18, "align": "center", "verticalAlign": "middle", "strokeWidth": 1.2},
        remove=("textDirection", "autosizeText"),
    )
    recenter("76", new_w=36.0, new_h=18.0)
    load_cell = cells_by_id.get("76")
    if load_cell is not None:
        geom_el = load_cell.find("mxGeometry")
        if geom_el is not None:
            load_x = float(geom_el.attrib.get("x", "0"))
            load_y = float(geom_el.attrib.get("y", "0"))
            load_w = float(geom_el.attrib.get("width", "0"))
            load_h = float(geom_el.attrib.get("height", "0"))
            load_cx = load_x + load_w / 2.0
            load_top = load_y
            load_bottom = load_y + load_h
            edge_points(
                "74",
                source=(122.22, 26.67),
                target=(load_cx, load_top),
                points=[(load_cx, 26.67), (load_cx, load_top)],
            )
            edge_points(
                "75",
                source=(124.44, 95.56),
                target=(load_cx, load_bottom),
                points=[(load_cx, 95.56), (load_cx, load_bottom)],
            )

    style("79", fontSize=18)
    geom("79", x=112.0, y=8.0, width=72.0, height=22.0)
    style("lp-transition-label", fontSize=18)
    geom("lp-transition-label", x=266.0, y=340.0, width=52.0, height=24.0)

    style("lp-fem-label-start", fontSize=18)
    style("lp-fem-label-end", fontSize=18)
    geom("lp-fem-label-start", height=20.0)
    geom("lp-fem-label-end", height=20.0)

    for cell_id in ("140", "141", "142", "143", "144"):
        style(cell_id, fontSize=20)
    style("143", strokeWidth=1.8)
    style("145", fontSize=18, strokeWidth=1.3, verticalAlign="middle")
    recenter("140", new_h=52.0)
    recenter("141", new_h=50.0)
    recenter("142", new_h=50.0)
    recenter("143", new_h=96.0)
    recenter("144", new_h=52.0)
    recenter("145", new_h=72.0)

    for cell_id in ("134", "135"):
        edge_style(cell_id, strokeWidth=1.2, strokeColor="#AEB7C2")
    for cell_id in ("146", "147", "148"):
        edge_style(cell_id, strokeWidth=1.3, strokeColor="#AEB7C2")
    for cell_id in ("152", "153"):
        edge_style(cell_id, strokeWidth=1.5, dashPattern="8 5")

    # Keep the two inter-panel arrows on a common horizontal axis.
    edge_points("95", source=(470.0, 460.0), target=(510.0, 460.0))
    edge_points("154", source=(1352.0, 460.0), target=(1392.0, 460.0))

    if model is not None:
        model.set("pageWidth", "1800")
        model.set("pageHeight", "1040")

    return ET.tostring(root, encoding="unicode")


def apply_right_panel_bbox_fix(xml: str) -> str:
    root = ET.fromstring(xml)
    mxroot = root.find(".//root")
    if mxroot is None:
        return xml

    cells_by_id = {cell.attrib.get("id"): cell for cell in mxroot.findall("mxCell")}
    right_group = cells_by_id.get("8RxiAfdrNtuPFcRHXn3M-215")
    if right_group is not None:
        geom = right_group.find("mxGeometry")
        if geom is not None:
            # The manual file kept the right-panel wrapper much wider than its
            # visible content, which creates a large blank area on the right.
            # Keep the current x/y and internal layout, but shrink the wrapper
            # to the actual occupied width of the panel content.
            geom.set("width", fmt(344.0))

    return ET.tostring(root, encoding="unicode")


def apply_double_column_canvas(xml: str) -> str:
    root = ET.fromstring(xml)
    model = root.find(".//mxGraphModel")
    mxroot = root.find(".//root")
    if model is None or mxroot is None:
        return xml

    cells = {cell.attrib.get("id"): cell for cell in mxroot.findall("mxCell")}
    panel_ids = ("3", "8RxiAfdrNtuPFcRHXn3M-216", "8RxiAfdrNtuPFcRHXn3M-215")
    panel_boxes: list[tuple[float, float, float, float]] = []
    for cell_id in panel_ids:
        cell = cells.get(cell_id)
        if cell is None:
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        x = float(geom.attrib.get("x", "0"))
        y = float(geom.attrib.get("y", "0"))
        w = float(geom.attrib.get("width", "0"))
        h = float(geom.attrib.get("height", "0"))
        panel_boxes.append((x, y, w, h))
    if not panel_boxes:
        return xml

    src_x = min(x for x, _, _, _ in panel_boxes)
    src_y = min(y for _, y, _, _ in panel_boxes)
    src_max_x = max(x + w for x, _, w, _ in panel_boxes)
    src_max_y = max(y + h for _, y, _, h in panel_boxes)
    src_w = src_max_x - src_x
    src_h = src_max_y - src_y

    page_w = 1830.0
    page_h = 980.0
    top_bottom_margin = 50.0
    scale = min((page_h - 2 * top_bottom_margin) / src_h, 1.0)
    target_w = src_w * scale
    target_x = (page_w - target_w) / 2.0
    target_y = top_bottom_margin

    children_by_parent: dict[str, list[ET.Element]] = {}
    for cell in mxroot.findall("mxCell"):
        parent = cell.attrib.get("parent")
        if not parent:
            continue
        children_by_parent.setdefault(parent, []).append(cell)

    def transform_value(value: float, root_level: bool, is_x: bool) -> float:
        if root_level:
            origin = src_x if is_x else src_y
            target = target_x if is_x else target_y
            return target + (value - origin) * scale
        return value * scale

    def transform_points(node: ET.Element, root_level: bool) -> None:
        for point in node.findall(".//mxPoint"):
            if "x" in point.attrib:
                point.set("x", fmt(transform_value(float(point.attrib["x"]), root_level, True)))
            if "y" in point.attrib:
                point.set("y", fmt(transform_value(float(point.attrib["y"]), root_level, False)))

    def transform_cell(cell: ET.Element, root_level: bool) -> None:
        geom = cell.find("mxGeometry")
        if geom is not None:
            if cell.attrib.get("vertex") == "1":
                if "x" in geom.attrib:
                    geom.set("x", fmt(transform_value(float(geom.attrib["x"]), root_level, True)))
                if "y" in geom.attrib:
                    geom.set("y", fmt(transform_value(float(geom.attrib["y"]), root_level, False)))
                if "width" in geom.attrib:
                    geom.set("width", fmt(float(geom.attrib["width"]) * scale))
                if "height" in geom.attrib:
                    geom.set("height", fmt(float(geom.attrib["height"]) * scale))
            elif cell.attrib.get("edge") == "1":
                transform_points(geom, root_level)
                for attr in ("x", "y", "width", "height"):
                    if attr in geom.attrib:
                        value = float(geom.attrib[attr])
                        if attr in ("x", "width"):
                            geom.set(attr, fmt(transform_value(value, root_level, True) if attr == "x" else value * scale))
                        else:
                            geom.set(attr, fmt(transform_value(value, root_level, False) if attr == "y" else value * scale))

        for child in children_by_parent.get(cell.attrib.get("id", ""), []):
            transform_cell(child, False)

    for cell in children_by_parent.get("1", []):
        transform_cell(cell, True)

    model.set("pageWidth", fmt(page_w))
    model.set("pageHeight", fmt(page_h))
    return ET.tostring(root, encoding="unicode")


def build_procedural_figure() -> str:
    assets = prepare_real_image_assets()
    b = DrawioBuilder()

    b.add_text(12, 8, 48, 36, "(a)", size=30, bold=True, align="left", vertical="top")
    add_teng_panel(b, 52, 54, assets)
    add_interpanel_arrow(b, 418, 334, 474, 334)
    add_center_panel(b, 478, 54)
    add_interpanel_arrow(b, 1060, 334, 1106, 334)
    add_outcomes_panel(b, 1110, 54, assets)

    b.add_text(12, 644, 48, 36, "(b)", size=30, bold=True, align="left", vertical="top")
    add_bottom_strip(b, 50, 662, assets)

    return b.build()


def build_figure() -> str:
    return load_baseline_xml()


def main() -> None:
    xml = build_figure()
    (OUT_DIR / "fig01_workflow_editable_integrated.drawio").write_text(
        xml, encoding="utf-8"
    )
    (OUT_DIR / "fig01_workflow_editable_integrated.xml").write_text(
        xml, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
