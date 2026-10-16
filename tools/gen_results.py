#!/usr/bin/env python3
"""Generate the Results figure (five benchmark column panels) and splice it into index.html."""
import html as H
import re
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the gh-pages checkout
SCRATCH = os.path.join(WT, "build")                                # previews land here (git-ignored)
os.makedirs(SCRATCH, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------- palette (emphasis form)
TERRA, GRPO_GREY, PRIOR_GREY = "#C8663A", "#8E7B70", "#CDBFB4"

# ----------------------------------------------------------------- data
# One panel per benchmark: strongest prior RL method, GRPO in our setup, CoPE.
# (name, value) triples; the full tables stay in the table view.
MAIN = [
    ("ALFWorld", "success rate (%)", 100, [0, 50, 100], [("ECHO", 94.5), ("GRPO", 90.5), ("CoPE", 95.5)]),
    ("WebShop", "success rate (%)", 100, [0, 50, 100], [("ECHO", 71.5), ("GRPO", 64.5), ("CoPE", 77.0)]),
    ("ScienceWorld", "success rate (%)", 100, [0, 50, 100], [("ECHO", 67.5), ("GRPO", 64.5), ("CoPE", 72.5)]),
    ("τ²-bench", "overall Avg@4 (%)", 50, [0, 25, 50], [("InfoPO", 17.7), ("GRPO", 27.5), ("CoPE", 28.7)]),
    ("AppWorld", "Normal split (%)", 60, [0, 30, 60], [("GiGPO", 25.7), ("GRPO", 41.7), ("CoPE", 55.4)]),
]


# ----------------------------------------------------------------- benchmark panel

def panel(title, unit, ymax, yticks, bars):
    W, HH = 200, 236
    x0, x1, y0, y1 = 34, 192, 52, 186
    n = len(bars)
    pitch = (x1 - x0) / n
    w = 30

    def py(v):
        return y1 - v / ymax * (y1 - y0)

    out = [f'<svg viewBox="0 0 {W} {HH}" role="img" aria-label="{H.escape(title)}, {H.escape(unit)}: ' +
           H.escape(", ".join(f"{m} {v:.1f}" for m, v in bars)) + '.">']
    out.append(f'<text class="ttl" x="{x0}" y="20">{title}</text>')
    out.append(f'<text class="small mut" x="{x0}" y="36">{unit}</text>')
    for t in yticks:
        out.append(f'<line class="grid" x1="{x0}" y1="{py(t):.1f}" x2="{x1}" y2="{py(t):.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    for i, (name, v) in enumerate(bars):
        cx = x0 + pitch * (i + 0.5)
        top = py(v)
        r = min(4, (y1 - top) / 2)
        d = (f"M{cx - w / 2:.1f},{y1} V{top + r:.1f} a{r},{r} 0 0 1 {r},-{r} "
             f"H{cx + w / 2 - r:.1f} a{r},{r} 0 0 1 {r},{r} V{y1} z")
        if name == "CoPE":
            color, cls = TERRA, "colg is-ours"
        elif name == "GRPO":
            color, cls = GRPO_GREY, "colg is-ref"
        else:
            color, cls = PRIOR_GREY, "colg"
        out.append(f'<g class="{cls}" style="--i:{i}"><title>{H.escape(name)} · {H.escape(title)}: {v:.1f}</title>'
                   f'<path class="col" d="{d}" fill="{color}"/>'
                   f'<text class="val" x="{cx:.1f}" y="{top - 7:.1f}" text-anchor="middle">{v:.1f}</text>'
                   f'<text class="xlab" x="{cx:.1f}" y="{y1 + 18}" text-anchor="middle">{name}</text></g>')
    out.append("</svg>")
    return "\n".join(out)


def main_figure(caption, table_html):
    panels = "\n".join(f'    <div class="chart-panel">\n{panel(*p)}\n    </div>' for p in MAIN)
    return f'''<figure class="chart chart-cols chart-main">
  <figcaption class="chart-cap">{caption}</figcaption>
  <div class="chart-row chart-row-5">
{panels}
  </div>
  <details class="chart-table"><summary>Table view (all methods)</summary>{table_html}</details>
</figure>'''


# ----------------------------------------------------------------- assembly

def main():
    path = f"{WT}/index.html"
    s = open(path, encoding="utf-8").read()

    m = re.search(r'<template id="results-tables">(.*?)</template>', s, re.S)
    if m:
        tables = m.group(1)
    else:
        start = s.index('    <div class="scroll">')
        end = re.search(r'    <p style="margin-top:24px">', s).start()
        tables = s[start:end]
    parts = re.findall(r'(<div class="scroll">.*?</table>\s*</div>)', tables, re.S)
    assert len(parts) == 3, len(parts)
    t_sim, t_tau, t_app = parts
    tables = "\n".join(parts)

    main_chart = main_figure(
        "<b>Main results.</b> CoPE against GRPO trained in the same setup, and the strongest prior RL method on each "
        "benchmark (ECHO on the simulated tasks; InfoPO on &tau;<sup>2</sup>-bench and GiGPO on AppWorld, as published). "
        "ALFWorld counts success over all subtasks; AppWorld is the Normal split with Qwen2.5-14B-Instruct, every other "
        "benchmark uses Qwen2.5-7B-Instruct. Prompting baselines, the remaining RL methods and the AppWorld Challenge "
        "split are in the table view.",
        t_sim + t_tau + t_app)

    block = f'''    <div class="results">
{main_chart}
    </div>
    <template id="results-tables">{tables}</template>'''

    existing = re.compile(r'    <div class="results">.*?</template>', re.S)
    if existing.search(s):
        s = existing.sub(lambda _: block, s, count=1)
    else:
        start = s.index('    <div class="scroll">')
        end_marker = re.search(r'    <p style="margin-top:24px">.*?</p>\n', s, re.S)
        assert end_marker and end_marker.start() > start
        s = s[:start] + block + "\n" + s[end_marker.end():]

    if "charts.css" not in s:
        s = s.replace('<link rel="stylesheet" href="static/css/anim.css" />',
                      '<link rel="stylesheet" href="static/css/anim.css" />\n<link rel="stylesheet" href="static/css/charts.css" />')
    open(path, "w", encoding="utf-8").write(s)
    print("results charts written")


if __name__ == "__main__":
    main()
