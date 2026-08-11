"""Render the per-kernel M sweep as light/dark SVGs for the README.

Reads results/kernel_m_sweep_20260812.csv and writes
docs/assets/m_sweep_{light,dark}.svg — a line chart of effective bandwidth
vs batch rows M for the three NVFP4 GEMM kernels, plus a strip showing
which kernel production dispatch uses in each band.

Palette: dataviz reference categorical slots 1/2/3, validated for both
surfaces (light #fcfcfb, dark #1a1a19).
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "results", "kernel_m_sweep_20260812.csv")
OUT_DIR = os.path.join(ROOT, "docs", "assets")

W, H = 880, 500
ML, MR, MT, MB = 74, 168, 74, 100         # margins
PX0, PX1 = ML, W - MR                      # plot x range
PY0, PY1 = MT, H - MB                      # plot y range (top, bottom)
YMAX = 700.0
STRIP_Y, STRIP_H = H - MB + 56, 15         # dispatch strip, below axis label

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  muted="#8a8880", grid="#e6e5e0",
                  series={"simt": "#2a78d6", "qpn": "#eb6834",
                          "wmma": "#1baf7a"}),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
                 muted="#8f8d84", grid="#302f2c",
                 series={"simt": "#3987e5", "qpn": "#d95926",
                         "wmma": "#199e70"}),
}
LABEL = {"simt": "SIMT", "qpn": "QPN m8n8k4", "wmma": "WMMA"}
FONT = ("system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',"
        "Arial,sans-serif")


def load():
    series = {"simt": {}, "qpn": {}, "wmma": {}}
    with open(CSV_PATH) as fh:
        for row in csv.reader(l for l in fh if not l.startswith("#")):
            if not row or row[0] == "M":
                continue
            m, kern, val = int(row[0]), row[1], row[2]
            if val != "unsupported":
                series[kern][m] = float(val)
    return series


def sx(m):
    return PX0 + (m - 1) / 15.0 * (PX1 - PX0)


def sy(v):
    return PY1 - (v / YMAX) * (PY1 - PY0)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, fill, size=13, anchor="start", weight="400", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'font-family="{FONT}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{extra}>{esc(s)}</text>')


def render(theme_name, series):
    t = THEMES[theme_name]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Effective bandwidth versus batch rows M for three '
         f'NVFP4 GEMM kernels on Tesla V100">',
         f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>']

    o.append(text(ML - 6, 32, "Effective bandwidth vs batch rows (M)",
                  t["ink"], 17, weight="600"))
    o.append(text(ML - 6, 52,
                  "W4A16 NVFP4 GEMM, aggregate over five Qwen3.6-27B TP4 "
                  "per-rank shapes · V100 memcpy ceiling 825 GB/s",
                  t["ink2"], 12.5))

    # y grid + labels
    for gv in range(0, int(YMAX) + 1, 100):
        y = sy(gv)
        o.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(text(PX0 - 10, y + 4, str(gv), t["muted"], 11.5,
                      anchor="end"))
    o.append(text(PX0 - 10, sy(YMAX) - 12, "GB/s", t["ink2"], 11.5,
                  anchor="end"))

    # x axis
    o.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" '
             f'stroke="{t["grid"]}" stroke-width="1.5"/>')
    for m in range(1, 17):
        o.append(text(sx(m), PY1 + 20, str(m), t["muted"], 11.5,
                      anchor="middle"))
    o.append(text((PX0 + PX1) / 2, PY1 + 40,
                  "M  (rows per GEMM — 1 = plain decode, k+1 = "
                  "speculative verify width)", t["ink2"], 12,
                  anchor="middle"))

    # production dispatch strip
    bx = sx(3.5)
    o.append(f'<rect x="{PX0}" y="{STRIP_Y}" width="{bx - PX0:.1f}" '
             f'height="{STRIP_H}" rx="3" fill="{t["series"]["simt"]}" '
             f'opacity="0.85"/>')
    o.append(f'<rect x="{bx + 2:.1f}" y="{STRIP_Y}" '
             f'width="{PX1 - bx - 2:.1f}" height="{STRIP_H}" rx="3" '
             f'fill="{t["series"]["qpn"]}" opacity="0.85"/>')
    o.append(text(PX0 - 10, STRIP_Y + 12, "dispatched", t["muted"], 11,
                  anchor="end"))
    o.append(text((PX0 + bx) / 2, STRIP_Y + 12, "SIMT", "#ffffff", 10.5,
                  anchor="middle", weight="600"))
    o.append(text((bx + PX1) / 2, STRIP_Y + 12, "QPN m8n8k4", "#ffffff",
                  10.5, anchor="middle", weight="600"))
    o.append(text(PX1 + 8, STRIP_Y + 12, "M>16 → WMMA", t["muted"], 11))

    # series lines
    for kern in ("wmma", "qpn", "simt"):
        pts = sorted(series[kern].items())
        d = " ".join(f'{"M" if i == 0 else "L"}{sx(m):.1f},{sy(v):.1f}'
                     for i, (m, v) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{t["series"][kern]}" '
                 f'stroke-width="2" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
        for m, v in pts:
            o.append(f'<circle cx="{sx(m):.1f}" cy="{sy(v):.1f}" r="4" '
                     f'fill="{t["series"][kern]}" stroke="{t["surface"]}" '
                     f'stroke-width="2"/>')

    # SIMT terminus — leader DOWN into the empty region below WMMA
    ex, ey = sx(8), sy(series["simt"][8])
    o.append(f'<line x1="{ex:.1f}" y1="{ey + 13:.1f}" x2="{ex:.1f}" '
             f'y2="{sy(150):.1f}" stroke="{t["muted"]}" stroke-width="1" '
             f'stroke-dasharray="3 3"/>')
    o.append(text(ex, sy(150) + 16, "SIMT stops here: compiled for M ≤ 8",
                  t["ink2"], 11.5, anchor="middle"))

    # crossover at M=3 — leader UP-RIGHT into the empty region above QPN
    cx, cy = sx(3), sy(475.8)
    o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="none" '
             f'stroke="{t["muted"]}" stroke-width="1.2"/>')
    o.append(f'<line x1="{cx + 8:.1f}" y1="{cy - 6:.1f}" '
             f'x2="{sx(4.4):.1f}" y2="{sy(612):.1f}" '
             f'stroke="{t["muted"]}" stroke-width="1"/>')
    o.append(text(sx(4.6), sy(612) + 4, "SIMT and QPN cross at M=3, by 0.25%",
                  t["ink2"], 11.5))

    # QPN two-tile step — leader UP into the same empty region
    s9 = sy(series["qpn"][9])
    o.append(f'<line x1="{sx(9.2):.1f}" y1="{s9 - 8:.1f}" '
             f'x2="{sx(10.2):.1f}" y2="{sy(520):.1f}" '
             f'stroke="{t["muted"]}" stroke-width="1"/>')
    o.append(text(sx(10.4), sy(520) + 4, "M ≥ 9 pads to two 8-row tiles",
                  t["ink2"], 11.5))

    # direct labels (ink text + colored dash carries identity)
    for kern in ("qpn", "wmma"):
        y = sy(series[kern][16])
        o.append(f'<line x1="{PX1 + 10}" y1="{y:.1f}" x2="{PX1 + 24}" '
                 f'y2="{y:.1f}" stroke="{t["series"][kern]}" '
                 f'stroke-width="3" stroke-linecap="round"/>')
        o.append(text(PX1 + 30, y + 4.5, LABEL[kern], t["ink"], 12.5,
                      weight="600"))
        o.append(text(PX1 + 30, y + 19, f"{series[kern][16]:.1f} GB/s",
                      t["muted"], 11))
    ly = sy(series["simt"][1]) - 22
    o.append(f'<line x1="{sx(1):.1f}" y1="{ly:.1f}" '
             f'x2="{sx(1.5):.1f}" y2="{ly:.1f}" '
             f'stroke="{t["series"]["simt"]}" stroke-width="3" '
             f'stroke-linecap="round"/>')
    o.append(text(sx(1.75), ly + 4.5, "SIMT", t["ink"], 12.5, weight="600"))
    o.append(text(sx(2.75), ly + 4.5,
                  f'{series["simt"][1]:.1f} GB/s at M=1 — 74% of ceiling',
                  t["muted"], 11))

    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    data = load()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in THEMES:
        path = os.path.join(OUT_DIR, f"m_sweep_{name}.svg")
        with open(path, "w") as fh:
            fh.write(render(name, data))
        print("wrote", path)
