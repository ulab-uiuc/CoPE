#!/usr/bin/env python3
"""Generate the Results charts (SVG column panels + SVG lines) and splice them into index.html."""
import html as H
import re
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the gh-pages checkout
SCRATCH = os.path.join(WT, "build")                                # previews land here (git-ignored)
os.makedirs(SCRATCH, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------- palette
# Adjacent-pair order validated with the dataviz validator against the page
# surface (#FEF6F0); the light blue sits below 3:1 and is relieved by the
# value label on every column.
AMBER, VIOLET, GREEN, LIGHTBLUE, BLUE, TERRA = "#B8801E", "#8A5FC2", "#3F8F5A", "#6FA3E8", "#2F6FC4", "#C8663A"

# ----------------------------------------------------------------- data (RL methods only; prompting rows stay in the table view)

SIM_METHODS = [  # (name, colour), left to right
    ("ECHO", AMBER), ("RFT", VIOLET), ("RWML + GRPO", GREEN), ("GRPO + EMPG", LIGHTBLUE), ("GRPO", BLUE), ("CoPE", TERRA),
]
SIM_PANELS = [  # (benchmark, {method: value})
    ("ALFWorld", {"ECHO": 94.5, "RFT": 77.0, "RWML + GRPO": 92.0, "GRPO + EMPG": 89.5, "GRPO": 90.5, "CoPE": 95.5}),
    ("WebShop", {"ECHO": 71.5, "RFT": 16.0, "RWML + GRPO": 64.0, "GRPO + EMPG": 67.5, "GRPO": 64.5, "CoPE": 77.0}),
    ("ScienceWorld", {"ECHO": 67.5, "RFT": 4.5, "RWML + GRPO": 65.0, "GRPO + EMPG": 46.5, "GRPO": 64.5, "CoPE": 72.5}),
]

TAU_METHODS = [("Search-R1", AMBER), ("UserRL", VIOLET), ("RAGEN", GREEN), ("InfoPO", LIGHTBLUE), ("GRPO (ours)", BLUE), ("CoPE", TERRA)]
TAU_PANELS = [("Overall", {"Search-R1": 13.5, "UserRL": 12.7, "RAGEN": 16.7, "InfoPO": 17.7, "GRPO (ours)": 27.5, "CoPE": 28.7})]

APP_METHODS = [("PPO†", AMBER), ("RLOO†", VIOLET), ("GiGPO†", GREEN), ("GRPO†", LIGHTBLUE), ("GRPO (ours)", BLUE), ("CoPE", TERRA)]
APP_PANELS = [
    ("Normal", {"PPO†": 19.1, "RLOO†": 24.8, "GiGPO†": 25.7, "GRPO†": 24.8, "GRPO (ours)": 41.7, "CoPE": 55.4}),
    ("Challenge", {"PPO†": None, "RLOO†": None, "GiGPO†": None, "GRPO†": None, "GRPO (ours)": 18.7, "CoPE": 22.5}),
]

ALF_CATS = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]
ALF_LINES = [
    ("CoPE", TERRA, [100.0, 100.0, 97.3, 96.0, 89.3, 91.1]),
    ("GRPO", BLUE, [100.0, 73.7, 94.6, 84.0, 89.3, 88.9]),
    ("ECHO", AMBER, [97.8, 89.5, 91.9, 96.0, 96.4, 93.3]),
    ("RWML + GRPO", GREEN, [100.0, 89.5, 89.2, 92.0, 96.4, 84.4]),
]
TAU_CATS = ["Telecom", "Retail", "Airline"]
TAU_LINES = [
    ("CoPE", TERRA, [40.6, 29.4, 16.2]),
    ("GRPO (ours)", BLUE, [38.8, 17.5, 26.2]),
    ("InfoPO", AMBER, [18.1, 18.8, 16.3]),
    ("RAGEN", GREEN, [17.5, 17.5, 15.0]),
]


# ----------------------------------------------------------------- column panel

def columns(title, methods, values, ymax, yticks):
    W, HH = 300, 236
    x0, x1, y0, y1 = 36, 290, 40, 196
    n = len(methods)
    pitch = (x1 - x0) / n
    w = min(28, pitch * 0.62)

    def py(v):
        return y1 - v / ymax * (y1 - y0)

    out = [f'<svg viewBox="0 0 {W} {HH}" role="img" aria-label="{H.escape(title)}: ' +
           H.escape(", ".join(f"{m} {values[m]:.1f}" for m, _ in methods if values[m] is not None)) + '.">']
    out.append(f'<text class="ttl" x="{x0}" y="22">{title}</text>')
    for t in yticks:
        out.append(f'<line class="grid" x1="{x0}" y1="{py(t):.1f}" x2="{x1}" y2="{py(t):.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 7}" y="{py(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    for i, (name, color) in enumerate(methods):
        v = values[name]
        cx = x0 + pitch * (i + 0.5)
        if v is None:
            out.append(f'<text class="val na" x="{cx:.1f}" y="{y1 - 8}" text-anchor="middle">—</text>')
            continue
        top = py(v)
        r = min(4, (y1 - top) / 2)
        d = (f"M{cx - w / 2:.1f},{y1} V{top + r:.1f} a{r},{r} 0 0 1 {r},-{r} "
             f"H{cx + w / 2 - r:.1f} a{r},{r} 0 0 1 {r},{r} V{y1} z")
        ours = ' is-ours' if name == "CoPE" else ''
        out.append(f'<g class="colg{ours}" style="--i:{i}"><title>{H.escape(name)} · {title}: {v:.1f}</title>'
                   f'<path class="col" d="{d}" fill="{color}"/>'
                   f'<text class="val" x="{cx:.1f}" y="{top - 7:.1f}" text-anchor="middle">{v:.1f}</text></g>')
    out.append("</svg>")
    return "\n".join(out)


def legend(methods, kind="sq"):
    return f'<div class="legend legend-{kind}">' + "".join(
        f'<span><i style="--c:{c}"></i>{n}</span>' for n, c in methods) + "</div>"


def col_figure(cls, caption, methods, panels, ymax, yticks, table_html):
    panel_html = "\n".join(f'    <div class="chart-panel">\n{columns(t, methods, v, ymax, yticks)}\n    </div>' for t, v in panels)
    return f'''<figure class="chart chart-cols {cls}">
  <figcaption class="chart-cap">{caption}</figcaption>
  {legend(methods)}
  <div class="chart-row chart-row-{len(panels)}">
{panel_html}
  </div>
  <details class="chart-table"><summary>Table view (all methods)</summary>{table_html}</details>
</figure>'''


# ----------------------------------------------------------------- lines

def lines(title, cats, series, ymin, ymax, yticks, unit="success rate (%)"):
    W, HH = 470, 300
    x0, x1, y0, y1 = 70, 380, 40, 240
    xs = [x0 + i * (x1 - x0) / (len(cats) - 1) for i in range(len(cats))]

    def py(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    out = [f'<svg viewBox="0 0 {W} {HH}" role="img" aria-label="{H.escape(title)}: {", ".join(n for n, _, _ in series)}.">']
    out.append(f'<text class="ttl" x="{x0}" y="24">{title}</text>')
    for t in yticks:
        out.append(f'<line class="grid" x1="{x0}" y1="{py(t):.1f}" x2="{x1}" y2="{py(t):.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{py(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    for c, x in zip(cats, xs):
        out.append(f'<text class="tick" x="{x:.1f}" y="{y1 + 20}" text-anchor="middle">{c}</text>')
    out.append(f'<text class="small mut" x="{x1}" y="24" text-anchor="end">{unit}</text>')

    ends = sorted([(py(vals[-1]), i) for i, (_, _, vals) in enumerate(series)])
    placed = []
    for y, i in ends:
        if placed and y - placed[-1][0] < 14:
            y = placed[-1][0] + 14
        placed.append((y, i))
    shift = max(0, placed[-1][0] - (y1 + 4))
    placed = [(y - shift, i) for y, i in placed]
    label_y = {i: y for y, i in placed}

    for si, (name, color, vals) in enumerate(series):
        pts = [(x, py(v)) for x, v in zip(xs, vals)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<g class="series" style="--i:{si}">')
        out.append(f'<path class="line" d="{d}" stroke="{color}" pathLength="1"/>')
        for (x, y), c, v in zip(pts, cats, vals):
            out.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"><title>{H.escape(name)} · {c}: {v:.1f}</title></circle>')
        ly = label_y[si]
        ex, ey = pts[-1]
        if abs(ly - ey) > 1:
            out.append(f'<line class="leader" x1="{ex + 6:.1f}" y1="{ey:.1f}" x2="{x1 + 14:.1f}" y2="{ly:.1f}" stroke="{color}"/>')
        out.append(f'<text class="end" x="{x1 + 18}" y="{ly + 4:.1f}">{name}</text>')
        out.append("</g>")
    out.append("</svg>")
    return legend([(n, c) for n, c, _ in series], "ln") + "\n" + "\n".join(out)


def line_table(cats, series, label):
    head = "".join(f"<th>{c}</th>" for c in cats)
    rows = "".join(f"<tr><td>{n}</td>" + "".join(f"<td>{v:.1f}</td>" for v in vals) + "</tr>" for n, _, vals in series)
    return f'<div class="scroll"><table><caption><b>{label}</b></caption><thead><tr><th>Method</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


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

    sim_chart = col_figure("chart-sim",
                           "<b>Simulated agent tasks.</b> Success rate (%) of the RL methods on ALFWorld (all subtasks), WebShop and ScienceWorld, all with Qwen2.5-7B-Instruct. Prompting baselines are in the table view.",
                           SIM_METHODS, SIM_PANELS, 100, [0, 25, 50, 75, 100], t_sim)
    tau_chart = col_figure("chart-tau",
                           "<b>&tau;<sup>2</sup>-bench.</b> Overall Avg@4 success rate (%), Qwen2.5-7B-Instruct.",
                           TAU_METHODS, TAU_PANELS, 50, [0, 25, 50], t_tau)
    app_chart = col_figure("chart-app",
                           "<b>AppWorld.</b> Success rate (%), Qwen2.5-14B-Instruct. <sup>&dagger;</sup> as reported in G2PO, which did not report the Challenge split.",
                           APP_METHODS, APP_PANELS, 60, [0, 20, 40, 60], t_app)

    alf_lines = lines("ALFWorld subtasks", ALF_CATS, ALF_LINES, 60, 100, [60, 70, 80, 90, 100])
    tau_lines = lines("τ²-bench domains", TAU_CATS, TAU_LINES, 0, 45, [0, 15, 30, 45])
    lines_fig = f'''<figure class="chart chart-lines">
  <figcaption class="chart-cap"><b>Subtasks and domains.</b> Per-subtask success rate (%) for the RL methods with auxiliary targets on ALFWorld, and per-domain success rate on &tau;<sup>2</sup>-bench.</figcaption>
  <div class="chart-row chart-row-2">
    <div class="chart-panel">
{alf_lines}
    </div>
    <div class="chart-panel">
{tau_lines}
    </div>
  </div>
  <details class="chart-table"><summary>Table view</summary>{line_table(ALF_CATS, ALF_LINES, "ALFWorld subtasks, success rate (%)")}{line_table(TAU_CATS, TAU_LINES, "&tau;&sup2;-bench domains, success rate (%)")}</details>
</figure>'''

    note = '''    <p class="results-note">
      CoPE improves over GRPO on every benchmark in aggregate, but not on every slice: on
      ALFWorld the Cool subtask favours ECHO and RWML, and on &tau;<sup>2</sup>-bench the
      overall lift comes from a ~12-point gain on Retail against a drop on Airline. The
      paper analyses both.
    </p>'''

    block = f'''    <div class="results">
{sim_chart}

    <h3>Realistic applications</h3>

    <div class="chart-row chart-row-apps">
{tau_chart}
{app_chart}
    </div>
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
