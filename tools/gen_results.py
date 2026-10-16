#!/usr/bin/env python3
"""Generate the Results charts (HTML/CSS bars + SVG lines) and splice them into index.html."""
import html as H
import re
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the gh-pages checkout
SCRATCH = os.path.join(WT, "build")                                # previews land here (git-ignored)
os.makedirs(SCRATCH, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------- data

SIM_COLS = ["ALFWorld", "WebShop", "ScienceWorld"]
SIM = [
    ("Agentic prompting", [
        ("Gemini-2.5-flash-lite", [46.0, 6.5, 15.0]),
        ("GPT-5-mini", [52.0, 7.0, 43.5]),
        ("GPT-5.5", [93.5, 56.5, 79.0]),
        ("Qwen2.5-7B", [34.0, 2.5, 2.5]),
        ("Qwen2.5-7B + Reflexion", [54.0, 4.0, 6.0]),
    ]),
    ("RL methods w/o auxiliary loss", [
        ("RFT", [77.0, 16.0, 4.5]),
        ("GRPO", [90.5, 64.5, 64.5]),
        ("GRPO + EMPG", [89.5, 67.5, 46.5]),
    ]),
    ("RL methods w/ auxiliary loss", [
        ("RWML + GRPO", [92.0, 64.0, 65.0]),
        ("ECHO", [94.5, 71.5, 67.5]),
        ("CoPE", [95.5, 77.0, 72.5]),
    ]),
]

TAU_COLS = ["Overall"]
TAU = [
    ("Agentic prompting", [("Qwen2.5", [11.7]), ("ReAct", [12.3])]),
    ("Published RL results", [("RAGEN", [16.7]), ("Search-R1", [13.5]), ("UserRL", [12.7]), ("InfoPO", [17.7])]),
    ("Our controlled runs", [("GRPO (ours)", [27.5]), ("CoPE", [28.7])]),
]

APP_COLS = ["Normal", "Challenge"]
APP = [
    ("Agentic prompting", [("Qwen2.5", [1.2, 0.7]), ("ReAct", [10.7, 2.6])]),
    ("Published RL results", [("PPO†", [19.1, None]), ("RLOO†", [24.8, None]), ("GiGPO†", [25.7, None]), ("GRPO†", [24.8, None])]),
    ("Our controlled runs", [("GRPO (ours)", [41.7, 18.7]), ("CoPE", [55.4, 22.5])]),
]

ALF_CATS = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]
ALF_LINES = [
    ("CoPE", "#C8663A", [100.0, 100.0, 97.3, 96.0, 89.3, 91.1]),
    ("GRPO", "#2F6FC4", [100.0, 73.7, 94.6, 84.0, 89.3, 88.9]),
    ("ECHO", "#B8801E", [97.8, 89.5, 91.9, 96.0, 96.4, 93.3]),
    ("RWML + GRPO", "#3F8F5A", [100.0, 89.5, 89.2, 92.0, 96.4, 84.4]),
]
TAU_CATS = ["Telecom", "Retail", "Airline"]
TAU_LINES = [
    ("CoPE", "#C8663A", [40.6, 29.4, 16.2]),
    ("GRPO (ours)", "#2F6FC4", [38.8, 17.5, 26.2]),
    ("InfoPO", "#B8801E", [18.1, 18.8, 16.3]),
    ("RAGEN", "#3F8F5A", [17.5, 17.5, 15.0]),
]

REF = {"GRPO", "GRPO (ours)"}


def fmt(v):
    return "—" if v is None else f"{v:.1f}"


# ----------------------------------------------------------------- bars

def bars(cols, groups, vmax, ticks, caption, table_html, extra_style=""):
    n = len(cols)
    out = [f'<figure class="chart chart-bars" style="--cols:{n};--max:{vmax};--ticks:{ticks}{extra_style}">']
    out.append(f'  <figcaption class="chart-cap">{caption}</figcaption>')
    out.append('  <div class="bars">')
    out.append('    <div class="bars-head"><span></span>' + "".join(f"<span>{c}</span>" for c in cols) + "</div>")
    r = 0
    for gname, rows in groups:
        out.append(f'    <div class="bars-group">{gname}</div>')
        for name, vals in rows:
            cls = " is-ours" if name == "CoPE" else (" is-ref" if name in REF else "")
            cells = []
            for c, v in zip(cols, vals):
                if v is None:
                    cells.append(f'<span class="bar is-na" data-col="{c}"><i class="track"></i><b>{fmt(v)}</b></span>')
                else:
                    cells.append(f'<span class="bar" data-col="{c}" title="{H.escape(name)} · {c}: {fmt(v)}">'
                                 f'<i class="track"></i><i class="fill" style="--v:{v}"></i><b>{fmt(v)}</b></span>')
            out.append(f'    <div class="bars-row{cls}" style="--r:{r}"><span class="bars-label">{name}</span>' + "".join(cells) + "</div>")
            r += 1
    axis = "".join(f'<em style="left:{100 * k / ticks:.3f}%">{round(vmax * k / ticks):g}</em>' for k in range(ticks + 1))
    out.append('    <div class="bars-axis"><span></span>' + "".join(f'<span class="axis">{axis}</span>' for _ in cols) + "</div>")
    out.append("  </div>")
    out.append(f'  <details class="chart-table"><summary>Table view</summary>{table_html}</details>')
    out.append("</figure>")
    return "\n".join(out)


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

    # end labels with collision resolution (min 14px apart), leaders when moved
    ends = sorted([(py(vals[-1]), i) for i, (_, _, vals) in enumerate(series)])
    placed = []
    for y, i in ends:
        if placed and y - placed[-1][0] < 14:
            y = placed[-1][0] + 14
        placed.append((y, i))
    # keep inside plot
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
    legend = '<div class="legend">' + "".join(
        f'<span><i style="--c:{c}"></i>{n}</span>' for n, c, _ in series) + "</div>"
    return legend + "\n" + "\n".join(out)


def line_table(cats, series, label):
    head = "".join(f"<th>{c}</th>" for c in cats)
    rows = "".join(f"<tr><td>{n}</td>" + "".join(f"<td>{v:.1f}</td>" for v in vals) + "</tr>" for n, _, vals in series)
    return f'<div class="scroll"><table><caption><b>{label}</b></caption><thead><tr><th>Method</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


# ----------------------------------------------------------------- assembly

def main():
    path = f"{WT}/index.html"
    s = open(path, encoding="utf-8").read()

    # Pull the original tables out of the page (first run) or out of the stored block (re-runs)
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

    sim_chart = bars(SIM_COLS, SIM, 100, 4,
                     "<b>Simulated agent tasks.</b> Success rate (%) on ALFWorld (all subtasks), WebShop and ScienceWorld. All RL methods use Qwen2.5-7B-Instruct.",
                     t_sim)
    tau_chart = bars(TAU_COLS, TAU, 50, 2,
                     "<b>&tau;<sup>2</sup>-bench.</b> Overall Avg@4 success rate (%) with Qwen2.5-7B-Instruct.",
                     t_tau, extra_style=";--label:120px")
    app_chart = bars(APP_COLS, APP, 60, 3,
                     "<b>AppWorld.</b> Success rate (%) with Qwen2.5-14B-Instruct. <sup>&dagger;</sup> as reported in G2PO.",
                     t_app, extra_style=";--label:120px")

    alf_lines = lines("ALFWorld subtasks", ALF_CATS, ALF_LINES, 60, 100, [60, 70, 80, 90, 100])
    tau_lines = lines("τ²-bench domains", TAU_CATS, TAU_LINES, 0, 45, [0, 15, 30, 45])
    lines_fig = f'''<figure class="chart chart-lines">
  <figcaption class="chart-cap"><b>Subtasks and domains.</b> Per-subtask success rate (%) for the RL methods with auxiliary targets on ALFWorld, and per-domain success rate on &tau;<sup>2</sup>-bench.</figcaption>
  <div class="chart-row">
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

    <h3>Where the gains come from</h3>

{lines_fig}

{note}
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
