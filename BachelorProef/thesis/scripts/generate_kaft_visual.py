#!/usr/bin/env python3
"""Generate a cover-only semi-supervised learning visual as SVG.

The image is intentionally separate from the thesis body figures. It is a
general, cover-friendly pseudo-labeling overview for manual use in the kaft.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
OUTPUT = THESIS_DIR / "figures" / "kaft_ssl_visual.svg"

W = 1920
H = 1080

INK = "#102033"
MUTED = "#5C6B78"
LINE = "#CAD6D2"
BLUE = "#174A7C"
BLUE_SOFT = "#EAF1F7"
TEAL = "#0F6B63"
TEAL_SOFT = "#E8F3F1"
GREEN = "#2F7D5B"
GOLD = "#D98B2B"
CLASS_BLUE = "#4979B6"
GREY = "#AEB9C3"
PANEL = "#FBFCFC"


def esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x: int, y: int, value: str, cls: str, anchor: str = "middle") -> str:
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{esc(value)}</text>'


def cylinder(x: int, y: int, color: str, scale: float = 1.0, cls: str = "") -> str:
    rx = 25 * scale
    ry = 8 * scale
    h = 38 * scale
    sw = 3 * scale
    return f"""
      <g class="{cls}">
        <rect x="{x - rx:.1f}" y="{y:.1f}" width="{2 * rx:.1f}" height="{h:.1f}" fill="#FFFFFF" stroke="{color}" stroke-width="{sw:.1f}" />
        <ellipse cx="{x}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="#FFFFFF" stroke="{color}" stroke-width="{sw:.1f}" />
        <ellipse cx="{x}" cy="{y + h:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="{sw:.1f}" />
        <path d="M {x - rx:.1f} {y + h * 0.42:.1f} C {x - rx / 2:.1f} {y + h * 0.55:.1f}, {x + rx / 2:.1f} {y + h * 0.55:.1f}, {x + rx:.1f} {y + h * 0.42:.1f}" fill="none" stroke="{color}" stroke-width="{sw * 0.7:.1f}" opacity="0.65" />
      </g>
    """


def data_grid(items: list[tuple[int, int, str]], scale: float = 1.0) -> str:
    return "\n".join(cylinder(x, y, color, scale) for x, y, color in items)


def neural_icon(x: int, y: int, stroke: str, fill: str) -> str:
    return f"""
      <g transform="translate({x} {y})" fill="none">
        <line x1="0" y1="52" x2="76" y2="10" stroke="{stroke}" stroke-width="5" />
        <line x1="0" y1="52" x2="76" y2="94" stroke="{stroke}" stroke-width="5" />
        <line x1="76" y1="10" x2="152" y2="52" stroke="{stroke}" stroke-width="5" />
        <line x1="76" y1="94" x2="152" y2="52" stroke="{stroke}" stroke-width="5" />
        <line x1="76" y1="10" x2="76" y2="94" stroke="{stroke}" stroke-width="5" opacity="0.45" />
        <circle cx="0" cy="52" r="13" fill="{fill}" stroke="#FFFFFF" stroke-width="4" />
        <circle cx="76" cy="10" r="13" fill="{fill}" stroke="#FFFFFF" stroke-width="4" />
        <circle cx="76" cy="94" r="13" fill="{fill}" stroke="#FFFFFF" stroke-width="4" />
        <circle cx="152" cy="52" r="13" fill="{fill}" stroke="#FFFFFF" stroke-width="4" />
      </g>
    """


def model_chip(x: int, y: int, w: int, h: int, fill: str, stroke: str, title: str, subtitle: str) -> str:
    pin_count = max(5, round(w / 40))
    pin_gap = (w - 72) / max(pin_count - 1, 1)
    pins_top = "\n".join(
        f'<rect class="pin" x="{x + 29 + i * pin_gap:.1f}" y="{y - 22}" width="14" height="21" rx="3" />'
        for i in range(pin_count)
    )
    pins_bottom = "\n".join(
        f'<rect class="pin" x="{x + 29 + i * pin_gap:.1f}" y="{y + h + 1}" width="14" height="21" rx="3" />'
        for i in range(pin_count)
    )
    return f"""
      <g class="chip shadow">
        {pins_top}
        {pins_bottom}
        <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="38" fill="{fill}" stroke="{stroke}" stroke-width="5" />
        {text(x + w // 2, y + 80, title, "chip-title")}
        {text(x + w // 2, y + 116, subtitle, "chip-sub")}
        {neural_icon(x + 78, y + 134, "rgba(255,255,255,0.62)", "rgba(255,255,255,0.92)")}
      </g>
    """


def svg() -> str:
    labeled = data_grid(
        [
            (170, 338, GREEN),
            (242, 338, GOLD),
            (314, 338, CLASS_BLUE),
            (206, 410, GREEN),
            (278, 410, GOLD),
        ],
        0.86,
    )
    unlabeled = data_grid(
        [
            (150, 650, GREY), (222, 650, GREY), (294, 650, GREY), (366, 650, GREY),
            (150, 724, GREY), (222, 724, GREY), (294, 724, GREY), (366, 724, GREY),
            (150, 798, GREY), (222, 798, GREY), (294, 798, GREY), (366, 798, GREY),
        ],
        0.78,
    )
    pseudo = data_grid(
        [
            (1200, 560, GREEN),
            (1278, 560, GOLD),
            (1356, 560, CLASS_BLUE),
            (1239, 640, GREEN),
            (1317, 640, GOLD),
        ],
        0.86,
    )
    return f"""<svg id="kaft-ssl-visual" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Semi-supervised learning pseudo-labeling overview">
  <style>
    text {{
      font-family: "Noto Sans", "DejaVu Sans", Helvetica, Arial, sans-serif;
      letter-spacing: 0;
    }}
    .canvas {{
      fill: #FFFFFF;
    }}
    .panel {{
      fill: {PANEL};
      stroke: {LINE};
      stroke-width: 4;
    }}
    .panel-label {{
      fill: {INK};
      font-size: 33px;
      font-weight: 850;
    }}
    .panel-sub {{
      fill: {MUTED};
      font-size: 20px;
      font-weight: 650;
    }}
    .chip-title {{
      fill: #FFFFFF;
      font-size: 42px;
      font-weight: 850;
    }}
    .chip-sub {{
      fill: rgba(255,255,255,0.82);
      font-size: 22px;
      font-weight: 700;
    }}
    .pseudo-title {{
      fill: {INK};
      font-size: 34px;
      font-weight: 850;
    }}
    .pseudo-sub {{
      fill: {MUTED};
      font-size: 20px;
      font-weight: 650;
    }}
    .arrow {{
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-width: 7;
    }}
    .arrow-blue {{ stroke: {BLUE}; }}
    .arrow-teal {{ stroke: {TEAL}; }}
    .loop {{
      stroke: {MUTED};
      stroke-width: 4.5;
      stroke-dasharray: 9 12;
      opacity: 0.55;
    }}
    .step {{
      fill: #FFFFFF;
      stroke: {LINE};
      stroke-width: 3;
    }}
    .step-text {{
      fill: {MUTED};
      font-size: 18px;
      font-weight: 750;
    }}
    .pin {{
      fill: {BLUE_SOFT};
      stroke: #C9D8E4;
      stroke-width: 2;
    }}
    .shadow {{
      filter: url(#shadow);
    }}
  </style>
  <defs>
    <marker id="arrow-blue" markerWidth="18" markerHeight="18" refX="16" refY="9" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M2 2 L16 9 L2 16 Z" fill="{BLUE}" />
    </marker>
    <marker id="arrow-teal" markerWidth="18" markerHeight="18" refX="16" refY="9" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M2 2 L16 9 L2 16 Z" fill="{TEAL}" />
    </marker>
    <marker id="arrow-muted" markerWidth="16" markerHeight="16" refX="14" refY="8" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M2 2 L14 8 L2 14 Z" fill="{MUTED}" opacity="0.55" />
    </marker>
    <filter id="shadow" x="-16%" y="-18%" width="132%" height="138%">
      <feDropShadow dx="0" dy="14" stdDeviation="15" flood-color="#102033" flood-opacity="0.10" />
    </filter>
  </defs>

  <rect class="canvas" x="0" y="0" width="{W}" height="{H}" />

  <g id="labeled-data" class="shadow">
    <rect class="panel" x="82" y="222" width="430" height="246" rx="36" />
    {text(126, 278, "Labeled data", "panel-label", "start")}
    {text(126, 308, "small trusted set", "panel-sub", "start")}
    {labeled}
  </g>

  <g id="unlabeled-data" class="shadow">
    <rect class="panel" x="82" y="540" width="430" height="328" rx="36" />
    {text(126, 596, "Unlabeled data", "panel-label", "start")}
    {text(126, 626, "larger pool", "panel-sub", "start")}
    {unlabeled}
  </g>

  <g id="initial-model">
    {model_chip(742, 374, 292, 286, BLUE, "#103A60", "Model", "trained")}
  </g>

  <g id="pseudo-labels" class="shadow">
    <rect x="1180" y="390" width="330" height="304" rx="42" fill="{TEAL_SOFT}" stroke="#BCD7D2" stroke-width="5" />
    {text(1345, 462, "Pseudo-labels", "pseudo-title")}
    {text(1345, 492, "accepted predictions", "pseudo-sub")}
    {pseudo}
  </g>

  <g id="improved-model">
    {model_chip(1620, 374, 246, 286, TEAL, "#084A44", "Model", "improved")}
  </g>

  <g id="arrows">
    <path class="arrow arrow-blue" d="M512 338 C594 338 654 380 734 456" marker-end="url(#arrow-blue)" />
    <path class="arrow arrow-blue" d="M512 702 C606 700 660 624 734 562" marker-end="url(#arrow-blue)" />
    <path class="arrow arrow-teal" d="M1034 514 C1086 514 1124 514 1172 514" marker-end="url(#arrow-teal)" />
    <path class="arrow arrow-teal" d="M1510 514 C1550 514 1578 514 1612 514" marker-end="url(#arrow-teal)" />
    <path class="arrow loop" d="M1744 686 C1620 842 1008 848 904 688" marker-end="url(#arrow-muted)" />
  </g>

  <g id="step-labels">
    <rect class="step" x="590" y="356" width="114" height="38" rx="19" />
    {text(647, 381, "train", "step-text")}
    <rect class="step" x="1078" y="466" width="116" height="38" rx="19" />
    {text(1136, 491, "predict", "step-text")}
    <rect class="step" x="1534" y="466" width="116" height="38" rx="19" />
    {text(1592, 491, "retrain", "step-text")}
  </g>
</svg>
"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg(), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
