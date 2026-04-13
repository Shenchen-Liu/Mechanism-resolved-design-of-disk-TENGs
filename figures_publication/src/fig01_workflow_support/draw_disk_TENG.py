from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


OUT_DIR = Path(__file__).resolve().parent


class SVGGroup:
    def __init__(self):
        self.element = ET.Element("g")

    def add(self, child):
        target = child.element if hasattr(child, "element") else child
        self.element.append(target)


class SVGDrawing:
    def __init__(self, filename: str, size=("1200px", "520px")):
        self.filename = str(filename)
        self.root = ET.Element(
            "svg",
            {
                "xmlns": "http://www.w3.org/2000/svg",
                "version": "1.1",
                "width": str(size[0]),
                "height": str(size[1]),
                "viewBox": "0 0 1200 520",
            },
        )

    def add(self, child):
        target = child.element if hasattr(child, "element") else child
        self.root.append(target)

    def g(self):
        return SVGGroup()

    def text(self, text, insert, **attrs):
        x, y = insert
        elem = ET.Element(
            "text",
            {
                "x": f"{x}",
                "y": f"{y}",
                **self._normalize_attrs(attrs),
            },
        )
        elem.text = str(text)
        return elem

    def path(self, d, **attrs):
        return ET.Element("path", {"d": d, **self._normalize_attrs(attrs)})

    def ellipse(self, center, r, **attrs):
        cx, cy = center
        rx, ry = r
        return ET.Element(
            "ellipse",
            {
                "cx": f"{cx}",
                "cy": f"{cy}",
                "rx": f"{rx}",
                "ry": f"{ry}",
                **self._normalize_attrs(attrs),
            },
        )

    def line(self, start, end, **attrs):
        x1, y1 = start
        x2, y2 = end
        return ET.Element(
            "line",
            {
                "x1": f"{x1}",
                "y1": f"{y1}",
                "x2": f"{x2}",
                "y2": f"{y2}",
                **self._normalize_attrs(attrs),
            },
        )

    def save(self):
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space="  ")
        tree.write(self.filename, encoding="utf-8", xml_declaration=True)

    def _normalize_attrs(self, attrs):
        normalized = {}
        for key, value in attrs.items():
            name = key.rstrip("_").replace("_", "-")
            if name == "text-anchor":
                normalized["text-anchor"] = str(value)
            else:
                normalized[name] = str(value)
        return normalized


@dataclass
class TENGGeometry:
    n_sectors: int = 8
    r_outer: float = 110
    r_inner: float = 26
    rx: float = 110
    ry: float = 34
    top_thickness: float = 18
    diel_thickness: float = 16
    gap: float = 22
    bottom_thickness: float = 18


class DiskTENGSVG:
    def __init__(self, filename: str | Path = OUT_DIR / "disk_teng_states.svg"):
        self.dwg = SVGDrawing(str(filename), size=("1200px", "520px"))
        self.colors = {
            "bg": "#FFFFFF",
            "stroke": "#334155",
            "text": "#111827",
            "muted": "#6B7280",
            "top": "#2C7FB8",
            "top_side": "#225F89",
            "diel": "#D9E9F5",
            "diel_side": "#BFD4E3",
            "bottom": "#E7ECF2",
            "bottom_side": "#C9D1D9",
            "plus": "#E64B35",
            "minus": "#3B82C4",
            "field": "#94A3B8",
        }

    # ---------- 基础几何 ----------
    def ellipse_point(self, cx, cy, rx, ry, angle_deg):
        a = math.radians(angle_deg)
        return cx + rx * math.cos(a), cy + ry * math.sin(a)

    def annular_sector_path(self, cx, cy, rx_out, ry_out, rx_in, ry_in, a0, a1):
        # SVG path for elliptical annular sector
        x0o, y0o = self.ellipse_point(cx, cy, rx_out, ry_out, a0)
        x1o, y1o = self.ellipse_point(cx, cy, rx_out, ry_out, a1)
        x1i, y1i = self.ellipse_point(cx, cy, rx_in, ry_in, a1)
        x0i, y0i = self.ellipse_point(cx, cy, rx_in, ry_in, a0)
        large = 1 if (a1 - a0) % 360 > 180 else 0

        d = [
            f"M {x0o:.2f},{y0o:.2f}",
            f"A {rx_out:.2f},{ry_out:.2f} 0 {large} 1 {x1o:.2f},{y1o:.2f}",
            f"L {x1i:.2f},{y1i:.2f}",
            f"A {rx_in:.2f},{ry_in:.2f} 0 {large} 0 {x0i:.2f},{y0i:.2f}",
            "Z",
        ]
        return " ".join(d)

    def add_label(
        self,
        text,
        x,
        y,
        size=16,
        weight="normal",
        italic=False,
        anchor="middle",
        color=None,
    ):
        self.dwg.add(
            self.dwg.text(
                text,
                insert=(x, y),
                text_anchor=anchor,
                font_size=size,
                font_family="Arial",
                font_weight=weight,
                font_style="italic" if italic else "normal",
                fill=color or self.colors["text"],
            )
        )

    def draw_disk_layer(
        self, group, cx, cy, rx, ry, thickness, top_fill, side_fill, stroke
    ):
        # side
        group.add(
            self.dwg.path(
                d=(
                    f"M {cx-rx},{cy} "
                    f"L {cx-rx},{cy+thickness} "
                    f"A {rx},{ry} 0 0 0 {cx+rx},{cy+thickness} "
                    f"L {cx+rx},{cy} "
                    f"A {rx},{ry} 0 0 1 {cx-rx},{cy} Z"
                ),
                fill=side_fill,
                stroke=stroke,
                stroke_width=1.2,
            )
        )
        # top
        group.add(
            self.dwg.ellipse(
                center=(cx, cy),
                r=(rx, ry),
                fill=top_fill,
                stroke=stroke,
                stroke_width=1.2,
            )
        )

    def draw_sector_electrodes(
        self, group, cx, cy, rx_out, ry_out, rx_in, ry_in, n, color
    ):
        gap = 8  # degree
        step = 360 / n
        for i in range(n):
            a0 = i * step + gap / 2
            a1 = (i + 1) * step - gap / 2
            path_d = self.annular_sector_path(
                cx, cy, rx_out, ry_out, rx_in, ry_in, a0, a1
            )
            group.add(
                self.dwg.path(
                    d=path_d,
                    fill=color,
                    stroke="#F8FBFE",
                    stroke_width=1.5,
                )
            )

    def draw_charge_marks(
        self, group, x_start, x_end, y, sign="+", color="#E64B35", count=8
    ):
        xs = [x_start + i * (x_end - x_start) / (count - 1) for i in range(count)]
        for x in xs:
            group.add(
                self.dwg.text(
                    sign,
                    insert=(x, y),
                    text_anchor="middle",
                    font_size=14,
                    font_family="Arial",
                    font_weight="bold",
                    fill=color,
                )
            )

    def draw_arrow(self, group, x1, y1, x2, y2, color="#64748B", width=2.0):
        group.add(self.dwg.line((x1, y1), (x2, y2), stroke=color, stroke_width=width))
        ang = math.atan2(y2 - y1, x2 - x1)
        ah = 8
        a1 = ang + math.radians(150)
        a2 = ang - math.radians(150)
        p1 = (x2 + ah * math.cos(a1), y2 + ah * math.sin(a1))
        p2 = (x2 + ah * math.cos(a2), y2 + ah * math.sin(a2))
        group.add(self.dwg.line((x2, y2), p1, stroke=color, stroke_width=width))
        group.add(self.dwg.line((x2, y2), p2, stroke=color, stroke_width=width))

    # ---------- 单个状态 ----------
    def draw_state(
        self,
        origin_x,
        origin_y,
        title,
        separated=False,
        geom: TENGGeometry | None = None,
    ):
        if geom is None:
            geom = TENGGeometry()

        g = self.dwg.g()
        cx = origin_x + 180
        top_y = origin_y + 120

        # 状态参数
        gap = 0 if not separated else geom.gap
        upper_shift = 0 if not separated else -18

        # --- 下电极 ---
        bottom_top_y = (
            top_y + geom.top_thickness + geom.diel_thickness + gap + 44 + upper_shift
        )
        self.draw_disk_layer(
            g,
            cx,
            bottom_top_y,
            geom.rx,
            geom.ry,
            geom.bottom_thickness,
            self.colors["bottom"],
            self.colors["bottom_side"],
            self.colors["stroke"],
        )

        # --- 上结构：上电极 + 介质层 ---
        diel_top_y = top_y + geom.top_thickness + upper_shift
        self.draw_disk_layer(
            g,
            cx,
            diel_top_y,
            geom.rx,
            geom.ry,
            geom.diel_thickness,
            self.colors["diel"],
            self.colors["diel_side"],
            self.colors["stroke"],
        )

        top_elec_y = top_y + upper_shift
        self.draw_disk_layer(
            g,
            cx,
            top_elec_y,
            geom.rx,
            geom.ry,
            geom.top_thickness,
            "#F8FBFE",
            "#E6EDF5",
            self.colors["stroke"],
        )

        # 顶部扇区电极
        self.draw_sector_electrodes(
            g,
            cx,
            top_elec_y,
            geom.rx - 14,
            geom.ry - 6,
            geom.r_inner,
            10,
            geom.n_sectors,
            self.colors["top"],
        )

        # 中心 hub
        g.add(
            self.dwg.ellipse(
                center=(cx, top_elec_y),
                r=(geom.r_inner - 6, 8),
                fill="#F8FBFE",
                stroke=self.colors["stroke"],
                stroke_width=1.0,
            )
        )

        # air gap 边界
        if separated:
            y1 = diel_top_y + geom.diel_thickness
            y2 = bottom_top_y
            g.add(
                self.dwg.line(
                    (cx - geom.rx + 6, y1),
                    (cx + geom.rx - 6, y1),
                    stroke="#D6DEE8",
                    stroke_width=1.2,
                )
            )
            g.add(
                self.dwg.line(
                    (cx - geom.rx + 6, y2),
                    (cx + geom.rx - 6, y2),
                    stroke="#D6DEE8",
                    stroke_width=1.2,
                )
            )

        # 标题
        self.add_label(title, origin_x + 180, origin_y + 24, size=20, weight="bold")

        # 层标签
        self.add_label(
            "Top electrode", origin_x + 300, top_elec_y - 10, size=13, anchor="start"
        )
        self.add_label(
            "Dielectric layer", origin_x + 300, diel_top_y + 8, size=13, anchor="start"
        )
        self.add_label(
            "Bottom electrode",
            origin_x + 300,
            bottom_top_y + 8,
            size=13,
            anchor="start",
        )
        if separated:
            self.add_label(
                "Air gap",
                origin_x + 300,
                (diel_top_y + geom.diel_thickness + bottom_top_y) / 2 + 4,
                size=13,
                anchor="start",
            )

        # 参数标注
        self.add_label("n", cx, top_elec_y - 52, size=16, italic=True)
        self.draw_arrow(
            g,
            cx - 105,
            top_elec_y - 36,
            cx + 105,
            top_elec_y - 36,
            color="#374151",
            width=1.4,
        )

        self.add_label(
            "ε", cx + 126, diel_top_y + 10, size=16, italic=True, anchor="start"
        )

        if separated:
            # d/R
            xdim = cx + 118
            self.draw_arrow(
                g,
                xdim,
                diel_top_y + geom.diel_thickness,
                xdim,
                bottom_top_y,
                color="#6B7280",
                width=1.2,
            )
            self.add_label(
                "d/R",
                xdim + 8,
                (diel_top_y + geom.diel_thickness + bottom_top_y) / 2 + 4,
                size=14,
                anchor="start",
            )

        # h/R
        xdim2 = cx + 132
        self.draw_arrow(
            g,
            xdim2,
            diel_top_y,
            xdim2,
            diel_top_y + geom.diel_thickness,
            color="#6B7280",
            width=1.2,
        )
        self.add_label(
            "h/R",
            xdim2 + 8,
            diel_top_y + geom.diel_thickness / 2 + 4,
            size=14,
            anchor="start",
        )

        # 电荷与电压差
        interface_x0 = cx - 70
        interface_x1 = cx + 70

        if not separated:
            # 接触后界面起电
            self.draw_charge_marks(
                g,
                interface_x0,
                interface_x1,
                diel_top_y + geom.diel_thickness - 6,
                sign="−",
                color=self.colors["minus"],
                count=7,
            )
            self.draw_charge_marks(
                g,
                interface_x0,
                interface_x1,
                bottom_top_y + 14,
                sign="+",
                color=self.colors["plus"],
                count=7,
            )
            self.add_label(
                "Charge transfer",
                origin_x + 180,
                origin_y + 290,
                size=14,
                color="#374151",
            )
        else:
            self.draw_charge_marks(
                g,
                interface_x0,
                interface_x1,
                diel_top_y + geom.diel_thickness - 6,
                sign="−",
                color=self.colors["minus"],
                count=7,
            )
            self.draw_charge_marks(
                g,
                interface_x0,
                interface_x1,
                bottom_top_y + 14,
                sign="+",
                color=self.colors["plus"],
                count=7,
            )

            # 电场线 / 势差
            for dx in [-40, 0, 40]:
                self.draw_arrow(
                    g,
                    cx + dx,
                    bottom_top_y - 4,
                    cx + dx,
                    diel_top_y + geom.diel_thickness + 6,
                    color=self.colors["field"],
                    width=1.5,
                )
            self.add_label(
                "Potential difference",
                origin_x + 180,
                origin_y + 302,
                size=14,
                color="#374151",
            )
            self.add_label(
                "V",
                cx + 150,
                origin_y + 220,
                size=16,
                weight="bold",
                color=self.colors["plus"],
            )

        self.dwg.add(g)

    def save(self):
        self.dwg.save()


def main():
    geom = TENGGeometry(
        n_sectors=8,
        rx=108,
        ry=34,
        r_inner=24,
        top_thickness=18,
        diel_thickness=16,
        gap=24,
        bottom_thickness=18,
    )

    fig = DiskTENGSVG(OUT_DIR / "disk_teng_states.svg")

    # overall panel labels
    fig.add_label("(a)", 26, 30, size=24, weight="bold", anchor="start")
    fig.add_label("(b)", 626, 30, size=24, weight="bold", anchor="start")

    fig.draw_state(20, 40, "Fully contacted", separated=False, geom=geom)
    fig.draw_state(620, 40, "Fully separated", separated=True, geom=geom)

    # 状态转换箭头
    g = fig.dwg.g()
    fig.draw_arrow(g, 540, 240, 600, 240, color="#94A3B8", width=2.2)
    fig.dwg.add(g)

    fig.save()
    print(f"Saved: {OUT_DIR / 'disk_teng_states.svg'}")


if __name__ == "__main__":
    main()
