"""ASCII hero for the profile README.

A small 8-second scene, rendered like a video and then converted to ASCII frame by
frame: an agent (an original blob with eyes) notices the crab from the old banner,
the crab snaps its claws, the agent bounces, then sits there thinking.

Each frame is rasterised from signed shapes, supersampled 3x3 per character cell,
shaded with a fixed light, and mapped to a density ramp. The frames are written to
an animated SVG (CSS keyframes, no JavaScript), one file per GitHub theme.

    python tools/ascii_hero.py            -> profile/hero-dark.svg, profile/hero-light.svg
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

COLS, ROWS = 112, 22          # character grid
SS = 3                        # supersampling per cell axis
FPS, DURATION = 12, 8.0
FRAMES = int(FPS * DURATION)
RAMP = " .,:;-=+*#%@"

# Layout in "units": one unit = one character width, a row is two units tall,
# so circles stay circles with a ~0.5 glyph aspect ratio.
FONT_PX = 12
CELL_W = FONT_PX * 0.6
CELL_H = CELL_W * 2
PAD_X = 16
WIDTH = 840
ART_H = ROWS * CELL_H
HEIGHT = int(ART_H + 64)

# Material ids (painter's order: later wins).
EMPTY, AGENT, CRAB, CLAW, SPOT, CRAB_EYE = range(6)
CLASS = {AGENT: "a", CRAB: "c", CLAW: "k", SPOT: "s", CRAB_EYE: "e"}

THEMES = {
    "dark": {"a": "#e6edf3", "c": "#f97316", "k": "#fdba74", "s": "#ef4444", "e": "#e6edf3", "t": "#e6edf3", "m": "#8b949e", "p": "#2f81f7"},
    "light": {"a": "#1f2328", "c": "#ea580c", "k": "#c2410c", "s": "#b91c1c", "e": "#1f2328", "t": "#1f2328", "m": "#656d76", "p": "#0969da"},
}

LIGHT = np.array([-0.5, -0.65, 0.58])
LIGHT /= np.linalg.norm(LIGHT)

# Sample positions (centre of each sub-cell), in units.
_xs = (np.arange(COLS * SS) + 0.5) / SS
_ys = (np.arange(ROWS * SS) + 0.5) / SS * 2
X, Y = np.meshgrid(_xs, _ys)


# ─── timing helpers ──────────────────────────────────────────────────────────

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def ease(v: float) -> float:
    v = clamp01(v)
    return v * v * (3 - 2 * v)


def window(t: float, a: float, b: float, fade: float = 0.25) -> float:
    """1 inside [a, b], easing in/out over `fade` seconds."""
    return ease((t - a) / fade) * ease((b - t) / fade)


def pulse(t: float, at: float, length: float) -> float:
    """0 -> 1 -> 0 bump centred on `at`."""
    d = (t - at) / (length / 2)
    return max(0.0, 1 - d * d)


# ─── shapes ──────────────────────────────────────────────────────────────────

def ellipse(cx, cy, rx, ry, floor=0.22):
    nx, ny = (X - cx) / rx, (Y - cy) / ry
    inside = nx * nx + ny * ny <= 1
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    shade = floor + (1 - floor) * np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    return inside, shade


def capsule(x1, y1, x2, y2, r):
    px, py = X - x1, Y - y1
    dx, dy = x2 - x1, y2 - y1
    h = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - dx * h, py - dy * h) <= r


# ─── scene ───────────────────────────────────────────────────────────────────

def crab_x(t: float) -> float:
    off, stop = 128.0, 74.0
    if t < 0.6:
        return off
    if t < 2.6:
        return off + (stop - off) * ease((t - 0.6) / 2.0)
    if t < 6.3:
        return stop
    return stop + (off - stop) * ease((t - 6.3) / 1.5)


def render(t: float):
    mat = np.zeros(X.shape, dtype=np.int8)
    shade = np.zeros(X.shape)

    def paint(mask, material, value):
        mat[mask] = material
        shade[mask] = value[mask] if isinstance(value, np.ndarray) else value

    # Agent ------------------------------------------------------------------
    breath = math.sin(2 * math.pi * t / 2.0)
    hop = pulse(t, 4.75, 0.6)                       # happy bounce after the snap
    land = pulse(t, 5.1, 0.25)                      # squash on landing
    rx = 12 * (1 + 0.025 * breath + 0.08 * land)
    ry = 12 * (1 - 0.025 * breath - 0.08 * land + 0.05 * hop)
    acx, acy = 34.0, 23.0 - 0.6 * breath - 7 * hop + 1.0 * land

    body, body_shade = ellipse(acx, acy, rx, ry, floor=0.5)
    paint(body, AGENT, body_shade)

    watching = window(t, 1.3, 6.4, 0.35)
    thinking = window(t, 6.5, 7.9, 0.3)
    look_x = 3.0 * watching - 1.5 * thinking
    look_y = -2.6 * thinking
    blink = max(pulse(t, 1.0, 0.2), pulse(t, 3.4, 0.2), pulse(t, 6.2, 0.2))
    happy = window(t, 4.5, 5.6, 0.12)
    eye_ry = 4.0 * (1 - 0.85 * blink) * (1 - 0.7 * happy)
    eye_rx = 2.1 * (1 + 0.25 * happy)
    for side in (-1, 1):
        ex = acx + side * 4.8 + look_x
        ey = acy - 2.2 + look_y - 0.8 * happy
        eye, _ = ellipse(ex, ey, eye_rx, max(eye_ry, 0.35))
        paint(eye & body, EMPTY, 0.0)

    # thinking dots orbiting the agent
    if thinking > 0.01:
        for i in range(3):
            a = 2 * math.pi * (t * 0.9 + i / 3)
            dx, dy = (rx + 4.5) * math.cos(a), (ry + 4.5) * math.sin(a)
            dot, dot_shade = ellipse(acx + dx, acy + dy, 1.4 * thinking, 1.4 * thinking)
            paint(dot, AGENT, dot_shade)

    # Crab -------------------------------------------------------------------
    cx = crab_x(t)
    walking = 0.6 < t < 2.6 or t > 6.3
    step = 2 * math.pi * 3.2 * t
    cy = 32.0 + (0.5 * math.sin(2 * step) if walking else 0.0)
    snap = window(t, 3.0, 4.5, 0.1)
    gap = math.radians(10 + 32 * snap * abs(math.sin(2 * math.pi * 2.5 * (t - 3.0))))

    for side in (-1, 1):
        # legs: three per side, feet swinging while walking
        for i in range(3):
            phase = step + i * 2.1 + (math.pi if side > 0 else 0)
            swing = 1.3 * math.sin(phase) if walking else 0.0
            lift = 0.9 * max(0.0, math.cos(phase)) if walking else 0.0
            hx, hy = cx + side * (4.5 + i * 1.6), cy + 2.5
            kx, ky = cx + side * (9.5 + i * 1.8) + swing * 0.5, cy + 4.5 - lift
            fx, fy = cx + side * (11.5 + i * 2.0) + swing, cy + 9.0 - lift
            paint(capsule(hx, hy, kx, ky, 0.6) | capsule(kx, ky, fx, fy, 0.5), CLAW, 0.62)

        # arm + pincer
        ax, ay = cx + side * 6.5, cy - 3.0
        px, py = cx + side * 11.0, cy - 12.0
        paint(capsule(ax, ay, px, py + 2, 0.9), CLAW, 0.7)
        pincer, pincer_shade = ellipse(px, py, 3.6, 3.6)
        ang = np.arctan2(Y - py, (X - px) * side)        # mirrored so both open outward
        opening = np.abs(ang - math.radians(-70)) < gap  # wedge pointing up and out
        paint(pincer & ~opening, CLAW, pincer_shade)

        # eye stalks + eyes
        paint(capsule(cx + side * 2.2, cy - 4.0, cx + side * 3.0, cy - 7.4, 0.45), CLAW, 0.6)
        eye, eye_shade = ellipse(cx + side * 3.0, cy - 8.3, 1.25, 1.25)
        paint(eye, CRAB_EYE, eye_shade)

    shell, shell_shade = ellipse(cx, cy, 9.0, 5.6)
    paint(shell, CRAB, shell_shade)
    for sx, sy, r in ((-3.2, 0.4, 1.5), (2.6, -1.4, 1.3), (0.6, 2.4, 1.2)):
        spot, _ = ellipse(cx + sx, cy + sy, r, r)
        paint(spot & shell, SPOT, shell_shade)

    return mat, shade


def to_cells(mat, shade, ink_is_dark: bool = False):
    """Collapse SSxSS samples into one character + material per cell.

    On a light background dense glyphs read as shadow, so the light is flipped.
    """
    if ink_is_dark:
        shade = np.where(mat != EMPTY, np.clip(1.42 - shade, 0, 1), 0.0)
    m = mat.reshape(ROWS, SS, COLS, SS).transpose(0, 2, 1, 3).reshape(ROWS, COLS, SS * SS)
    s = shade.reshape(ROWS, SS, COLS, SS).transpose(0, 2, 1, 3).reshape(ROWS, COLS, SS * SS)
    value = s.mean(axis=2)
    counts = np.stack([(m == k).sum(axis=2) for k in range(1, 6)], axis=2)
    material = counts.argmax(axis=2) + 1
    material[counts.max(axis=2) == 0] = EMPTY
    idx = np.clip(np.rint(value * (len(RAMP) - 1)), 0, len(RAMP) - 1).astype(int)
    idx[value < 0.06] = 0
    material[idx == 0] = EMPTY
    return idx, material


# ─── SVG ─────────────────────────────────────────────────────────────────────

def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def frame_svg(idx, material) -> str:
    rows = []
    for r in range(ROWS):
        line_mat, line_idx = material[r], idx[r]
        last = max((c for c in range(COLS) if line_idx[c]), default=-1)
        if last < 0:
            continue
        runs, cur, buf = [], None, []
        for c in range(last + 1):
            k = int(line_mat[c]) if line_idx[c] else EMPTY
            ch = RAMP[line_idx[c]]
            if k != cur and buf:
                runs.append((cur, "".join(buf)))
                buf = []
            cur = k
            buf.append(ch)
        runs.append((cur, "".join(buf)))
        spans = "".join(esc(txt) if k == EMPTY else f'<tspan class="{CLASS[k]}">{esc(txt)}</tspan>' for k, txt in runs)
        rows.append(f'<text y="{(r + 0.8) * CELL_H:.1f}">{spans}</text>')
    return "".join(rows)


def build(theme: dict, frames: list[str]) -> str:
    step = 100 / FRAMES
    tagline = "hi, i'm promipi - backend developer · .NET · Azure · AI"
    groups = "\n".join(
        f'<g class="f{" still" if i == int(4.0 * FPS) else ""}" style="animation-delay:{i / FPS:.4f}s">{f}</g>'
        for i, f in enumerate(frames)
    )
    type_chars = len(tagline) + 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="ASCII animation: a little agent meets a crab. {esc(tagline)}" xml:space="preserve">
<title>{esc(tagline)}</title>
<style>
text {{ font-family: ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, 'Liberation Mono', Menlo, monospace; font-size: {FONT_PX}px; white-space: pre; fill: {theme['a']}; }}
.a {{ fill: {theme['a']}; }} .c {{ fill: {theme['c']}; }} .k {{ fill: {theme['k']}; }} .s {{ fill: {theme['s']}; }} .e {{ fill: {theme['e']}; }}
.f {{ visibility: hidden; animation: show {DURATION}s step-end infinite; }}
@keyframes show {{ 0% {{ visibility: visible; }} {step:.4f}% {{ visibility: hidden; }} 100% {{ visibility: hidden; }} }}
.tag {{ font-size: 14px; fill: {theme['t']}; }} .tag .p {{ fill: {theme['p']}; }} .tag .m {{ fill: {theme['m']}; }}
.type {{ animation: type 2.4s steps({type_chars}) 0.4s both; }}
@keyframes type {{ from {{ width: 0; }} to {{ width: {WIDTH}px; }} }}
.cursor {{ fill: {theme['p']}; animation: blink 1s step-end infinite; }}
@keyframes blink {{ 50% {{ opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .f {{ animation: none; }} .f.still {{ visibility: visible; }} .type {{ animation: none; width: {WIDTH}px; }} .cursor {{ animation: none; }} }}
</style>
<g transform="translate({PAD_X} 0)">
{groups}
</g>
<clipPath id="typing"><rect class="type" x="0" y="{ART_H + 14:.0f}" width="{WIDTH}" height="30"/></clipPath>
<g clip-path="url(#typing)"><text class="tag" x="{PAD_X}" y="{ART_H + 36:.0f}"><tspan class="p">&gt;</tspan> hi, i'm <tspan class="p">promipi</tspan> <tspan class="m">- backend developer · .NET · Azure · AI</tspan></text></g>
<rect class="cursor" x="{PAD_X + (len(tagline) + 2) * 14 * 0.6 + 2:.0f}" y="{ART_H + 24:.0f}" width="8" height="16"/>
</svg>
"""


def main() -> None:
    scenes = [render(i / FPS) for i in range(FRAMES)]
    out = Path(__file__).resolve().parent.parent / "profile"
    out.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        frames = [frame_svg(*to_cells(mat, shade, ink_is_dark=name == "light")) for mat, shade in scenes]
        path = out / f"hero-{name}.svg"
        path.write_text(build(theme, frames), encoding="utf-8")
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB, {FRAMES} frames")


if __name__ == "__main__":
    main()
