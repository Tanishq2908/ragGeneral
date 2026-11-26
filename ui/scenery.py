"""Decorative background for the Streamlit app: a translucent tree whose branches sway and leaves flutter.

Pure SVG + CSS animations (Streamlit doesn't run page JavaScript), generated once with a fixed seed.
"""
import math
import random

LEAF = "M0 0C7-6 8-17 0-26C-8-17-7-6 0 0Z"
FILLS = ("var(--leaf-1)", "var(--leaf-2)", "var(--leaf-3)")

# origin x, origin y, limb path, stroke width, sway angle (deg), sway duration (s), leaf clusters (x, y, count, spread)
LIMBS = [
    (296, 540, "M296 540C250 500 200 470 140 430C110 410 92 380 86 350", 16, 2.2, 7.5, [(86, 340, 22, 60), (150, 420, 14, 46), (200, 470, 10, 36)]),
    (304, 500, "M304 500C350 470 400 440 450 400C480 375 500 350 512 320", 15, 2.6, 8.4, [(512, 310, 22, 62), (440, 400, 14, 44)]),
    (300, 440, "M300 440C280 380 250 320 220 270C205 245 195 220 190 190", 12, 2.0, 6.8, [(190, 180, 22, 62), (235, 295, 12, 40)]),
    (306, 430, "M306 430C330 370 360 310 390 260C405 235 415 205 418 175", 12, 2.4, 7.2, [(418, 165, 22, 62), (370, 300, 12, 40)]),
    (304, 415, "M304 415C302 340 300 260 302 180C303 140 306 110 310 80", 10, 1.6, 9.2, [(310, 80, 26, 70), (300, 200, 14, 50)]),
    (292, 600, "M292 600C250 585 210 575 170 580", 8, 3.2, 5.8, [(160, 575, 14, 40)]),
    (316, 620, "M316 620C360 600 400 595 440 600", 8, 3.4, 6.2, [(452, 595, 14, 40)]),
]

CSS = """
<style>
[data-testid="stElementContainer"]:has(#grove-scenery) { position: absolute; }
#grove-scenery { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
#grove-tree { position: absolute; right: -4vw; bottom: -2vh; height: min(98vh, 900px); width: auto; opacity: .2; }
#grove-tree .trunk { fill: var(--bark); }
#grove-tree .limb { fill: none; stroke: var(--bark); stroke-linecap: round; }
#grove-tree .leaf { transform-box: fill-box; transform-origin: 50% 100%; animation: grove-flutter var(--d) ease-in-out var(--delay) infinite alternate; }
#grove-tree .sway { transform-box: view-box; animation: grove-sway var(--d) ease-in-out var(--delay) infinite alternate; }
#grove-canopy { transform-box: view-box; transform-origin: 300px 800px; animation: grove-breeze 11s ease-in-out infinite alternate; }
@keyframes grove-breeze  { from { transform: rotate(-.6deg); } to { transform: rotate(.9deg); } }
@keyframes grove-sway    { from { transform: rotate(calc(var(--a) * -1)); } to { transform: rotate(var(--a)); } }
@keyframes grove-flutter { from { transform: rotate(calc(var(--r) - 9deg)); } to { transform: rotate(calc(var(--r) + 9deg)) scale(.94); } }
.grove-falling { position: absolute; top: -40px; width: 16px; height: 22px; opacity: 0; animation: grove-fall var(--d) linear var(--delay) infinite; }
.grove-falling svg { width: 100%; height: 100%; fill: var(--leaf-2); animation: grove-spin 3.2s ease-in-out infinite alternate; }
@keyframes grove-fall {
  0% { transform: translate(0, 0); opacity: 0; } 8% { opacity: .35; }
  50% { transform: translate(var(--x1), 55vh); } 90% { opacity: .3; }
  100% { transform: translate(var(--x2), 108vh); opacity: 0; }
}
@keyframes grove-spin { from { transform: rotate(-40deg); } to { transform: rotate(50deg) scaleX(.6); } }
@media (max-width: 760px) { #grove-tree { right: -30vw; height: 80vh; opacity: .14; } }
@media (prefers-reduced-motion: reduce) {
  #grove-tree *, #grove-canopy, .grove-falling, .grove-falling svg { animation: none !important; }
  .grove-falling { display: none; }
}
</style>
"""


def scenery_html(seed=7):
    """Returns (css, markup). Render css with st.html and markup with st.markdown(unsafe_allow_html=True):
    st.html strips SVG, st.markdown keeps it."""
    rng = random.Random(seed)
    parts = ['<div id="grove-scenery" aria-hidden="true">',
             '<svg id="grove-tree" viewBox="0 0 600 800" preserveAspectRatio="xMidYMax meet">',
             '<path class="trunk" d="M272 800C280 700 286 610 290 520C293 470 296 440 300 410L312 412C314 450 316 490 318 540C322 620 330 710 340 800Z"/>',
             '<g id="grove-canopy">']
    for ox, oy, d, width, angle, dur, clusters in LIMBS:
        parts.append(f'<g class="sway" style="transform-origin:{ox}px {oy}px;--a:{angle}deg;--d:{dur}s;--delay:{-rng.random() * dur:.2f}s">'
                     f'<path class="limb" d="{d}" stroke-width="{width}"/>')
        for cx, cy, count, spread in clusters:
            for _ in range(count):
                a, dist = rng.random() * math.tau, math.sqrt(rng.random()) * spread
                x, y = cx + math.cos(a) * dist, cy + math.sin(a) * dist * .75
                parts.append(
                    f'<g transform="translate({x:.1f} {y:.1f}) scale({.7 + rng.random() * .7:.2f})">'
                    f'<path class="leaf" d="{LEAF}" fill="{rng.choice(FILLS)}" fill-opacity="{.65 + rng.random() * .35:.2f}" '
                    f'style="--r:{(rng.random() - .5) * 140:.0f}deg;--d:{2.2 + rng.random() * 2.6:.2f}s;--delay:{-rng.random() * 4:.2f}s"/></g>')
        parts.append("</g>")
    parts.append("</g></svg>")
    for _ in range(6):
        parts.append(f'<div class="grove-falling" style="left:{10 + rng.random() * 85:.0f}%;--d:{14 + rng.random() * 12:.1f}s;'
                     f'--delay:{-rng.random() * 20:.1f}s;--x1:{(rng.random() - .5) * 160:.0f}px;--x2:{(rng.random() - .5) * 220:.0f}px">'
                     f'<svg viewBox="-10 -28 20 30"><path d="{LEAF}"/></svg></div>')
    parts.append("</div>")
    return CSS, "".join(parts)
