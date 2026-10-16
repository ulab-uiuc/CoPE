#!/usr/bin/env python3
"""Generate the three animation figures and splice them into index.html.

Also writes a standalone preview page to the scratchpad for screenshots.
"""
import re
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the gh-pages checkout
SCRATCH = os.path.join(WT, "build")                                # previews land here (git-ignored)
os.makedirs(SCRATCH, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


TERRA = "#C8663A"
INK_SOFT = "#6B5447"
RULE = "#EFD9C9"
RED = "#BE4637"
OK = "#3F8F5A"
AMBER = "#E0A33A"
GREY = "#8E7B70"


def rich(*parts, size=9.5, dy=3.5):
    """Text with subscripts. A plain string is normal text; a 1-tuple is a subscript.

    The baseline shift is undone on the next normal segment (an empty tspan
    does not consume dy, so the shift must ride on real glyphs).
    """
    out = []
    shifted = False
    for p in parts:
        if isinstance(p, tuple):
            out.append(f'<tspan dy="{dy}" font-size="{size}">{p[0]}</tspan>')
            shifted = True
        elif p:
            if shifted:
                out.append(f'<tspan dy="-{dy}">{p}</tspan>')
                shifted = False
            else:
                out.append(p)
    return "".join(out)


def arrow_head(tip, direction, length=9, half=4):
    """Triangle path whose tip sits at `tip`, pointing along `direction`."""
    dx, dy = direction
    n = (dx * dx + dy * dy) ** 0.5
    dx, dy = dx / n, dy / n
    bx, by = tip[0] - length * dx, tip[1] - length * dy
    px, py = -dy * half, dx * half
    return (f"M{tip[0]:.1f},{tip[1]:.1f} L{bx + px:.1f},{by + py:.1f} "
            f"L{bx - px:.1f},{by - py:.1f} z")


def sub(base, s, size=9.5, dy=3.5, after=""):
    """base text, a subscript, and optional text back on the baseline."""
    return rich(base, (s,), after, size=size, dy=dy)


# ---------------------------------------------------------------- anim 1

def anim_targets():
    cy = 168
    slots = [("o", 0), ("a", 0), ("o", 1), ("a", 1), ("o", 2), ("a", 2),
             ("o", 3), ("a", 3), ("o", 4), ("a", 4), ("o", 5), ("a", 5)]
    xs = [34 + 41 * i for i in range(13)]
    y_x = xs[12]

    out = []
    out.append('<svg viewBox="0 0 560 300" role="img" aria-label="Which step-wise target to predict: '
               'value targets share one scalar across steps, observation targets are long, noisy and one step ahead, '
               'multi-step action targets are short, clean and step-specific.">')
    # titles
    out.append('<text class="ttl sc sc-0 fin" x="20" y="30">A trajectory, rewarded only at the end</text>')
    out.append(f'<text class="ttl sc sc-1" x="20" y="30">Value targets  {rich("V(h", ("t",), ")", size=11)}</text>')
    out.append(f'<text class="ttl sc sc-2" x="20" y="30">Observation targets  {rich("ô", ("t+1",), size=11)}</text>')
    out.append(f'<text class="ttl sc sc-3" x="20" y="30">Multi-step action targets  {rich("a", ("t:t+3",), "  (CoPE)", size=11)}</text>')

    # links
    for i in range(12):
        x1, x2 = xs[i] + 14, xs[i + 1] - 14
        dash = ' stroke-dasharray="3 3"' if i == 11 else ''
        out.append(f'<line class="link" style="--i:{i}" x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}"{dash}/>')

    # nodes
    for i, (kind, idx) in enumerate(slots):
        x = xs[i]
        label = sub(kind, str(idx))
        if kind == "o":
            out.append(f'<g class="node o-node pp" style="--i:{i}"><circle cx="{x}" cy="{cy}" r="13"/>'
                       f'<text x="{x}" y="{cy + 4}" text-anchor="middle">{label}</text></g>')
        else:
            k = idx - 2
            hit = f' a-hit' if k >= 0 else ''
            kstyle = f';--k:{k}' if k >= 0 else ''
            out.append(f'<g class="node a-node pp{hit}" style="--i:{i}{kstyle}">'
                       f'<rect x="{x - 13}" y="{cy - 13}" width="26" height="26" rx="6"/>'
                       f'<text x="{x}" y="{cy + 4}" text-anchor="middle">{label}</text></g>')

    # reward
    out.append(f'<g class="node y-node pp" style="--i:12"><circle cx="{y_x}" cy="{cy}" r="14" fill="{AMBER}"/>'
               f'<text x="{y_x}" y="{cy + 5}" text-anchor="middle" fill="#fff" font-weight="700" font-size="15">Y</text>'
               f'<text class="small mut" x="{y_x}" y="{cy + 31}" text-anchor="middle">Y = 1</text></g>')

    # history bracket (o0 .. o2)
    out.append(f'<g class="hist sc sc-1 sc-2 sc-3 fin pp"><path d="M21,186 v6 h190 v-6" fill="none" stroke="{INK_SOFT}" stroke-width="1.2"/>'
               f'<text class="small mut" x="116" y="207" text-anchor="middle">{rich("history h", ("t",), "  (t = 2)", size=9.5, dy=2.5)}</text></g>')

    # scene 0: question
    out.append(f'<g class="q sc pp"><circle cx="198" cy="128" r="12" fill="#fff" stroke="{TERRA}" stroke-width="1.5" stroke-dasharray="3 2"/>'
               f'<text x="198" y="133" text-anchor="middle" fill="{TERRA}" font-weight="700" font-size="15">?</text></g>')

    # scene 1: broadcast the scalar to every step
    out.append(f'<path class="bcast drawn sc sc-1" d="M{y_x},154 V126 H34" pathLength="1" fill="none" stroke="{AMBER}" stroke-width="1.5"/>')
    o_xs = [xs[i] for i, (kind, _) in enumerate(slots) if kind == "o"]
    for j, x in enumerate(reversed(o_xs)):
        out.append(f'<g class="vchip sc pp" style="--j:{j}"><rect x="{x - 11}" y="117" width="22" height="17" rx="4" fill="{AMBER}"/>'
                   f'<text x="{x}" y="130" text-anchor="middle" fill="#fff" font-weight="700" font-size="12">1</text></g>')
    out.append(f'<g class="verdict verdict-1 sc pp"><path d="M28,259 l12,12 M40,259 l-12,12" stroke="{RED}" stroke-width="2.5" stroke-linecap="round"/>'
               '<text x="50" y="270">shared: every step gets the same scalar</text></g>')

    # scene 2: next-observation target
    out.append(f'<path class="obs-arrow drawn sc sc-2" d="M203,157 C203,128 206,90 220,90" pathLength="1" fill="none" stroke="{INK_SOFT}" stroke-width="1.6"/>')
    out.append(f'<path class="obs-head sc sc-2" d="{arrow_head((220, 90), (1, 0))}" fill="{INK_SOFT}"/>')
    log = [f'<g class="log sc sc-2 pp"><rect x="226" y="50" width="186" height="82" rx="6" fill="#fff" stroke="{RULE}"/>',
           f'<text class="small mut" x="234" y="65">{rich("ô", ("3",), " · terminal log", size=9.5, dy=2.5)}</text>']
    widths = [120, 150, 92, 138, 110, 130]
    for i, w in enumerate(widths):
        y = 72 + 8 * i
        fill = OK if i == 4 else "#D9D2CB"
        log.append(f'<rect class="log-line" style="--i:{i}" x="234" y="{y}" width="{w}" height="4" rx="2" fill="{fill}"/>')
    log.append(f'<text class="log-line" style="--i:4" x="350" y="108" font-size="11" fill="{OK}" font-weight="600">relevant</text>')
    log.append(f'<text class="log-tag sc" x="404" y="127" text-anchor="end" font-size="12" font-weight="700" fill="{RED}">long · noisy</text>')
    log.append('</g>')
    out.append("".join(log))
    out.append(f'<g class="verdict verdict-2 sc pp"><path d="M28,259 l12,12 M40,259 l-12,12" stroke="{RED}" stroke-width="2.5" stroke-linecap="round"/>'
               '<text x="50" y="270">one step ahead, and mostly noise</text></g>')

    # scene 3: multi-step action targets
    a_targets = [(239, 98), (321, 62), (403, 26), (485, -10)]
    for k, (x, cyq) in enumerate(a_targets):
        cxq = (198 + x) / 2
        out.append(f'<path class="arc drawn sc sc-3" style="--k:{k}" d="M198,154 Q{cxq},{cyq} {x},154" pathLength="1" '
                   f'fill="none" stroke="{TERRA}" stroke-width="2"/>')
        out.append(f'<path class="arc-head sc sc-3" style="--k:{k}" d="{arrow_head((x, 154), (x - cxq, 154 - cyq))}" fill="{TERRA}"/>')
    out.append(f'<g class="verdict verdict-3 sc pp"><path d="M27,265 l5,5 l10,-11" fill="none" stroke="{OK}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
               '<text x="50" y="270">short, clean, step-specific, K steps ahead</text></g>')
    out.append('</svg>')

    caps = [
        ('anim-cap anim-cap-static', '<b>One reward, many steps.</b> A rollout is a sequence of observations and actions, but the only supervision is a scalar <i>Y</i> at the very end. What should each history <i>h<sub>t</sub></i> learn to predict?'),
        ('anim-cap', '<b>Value targets.</b> Estimate how likely success is from <i>h<sub>t</sub></i>. The label is the same <i>Y</i> for every step of the trajectory, so no step gets anything step-specific.'),
        ('anim-cap', '<b>Observation targets.</b> Predict the next observation. Web pages and terminal logs are long and mostly noise, and the target reaches only one step ahead.'),
        ('anim-cap', '<b>Multi-step action targets (CoPE).</b> Forecast the next <i>K</i> actions from <i>h<sub>t</sub></i>. Each target is short and clean, differs by step, and reaches <i>K</i> steps ahead.'),
    ]
    return figure("anim-targets", "2300,2300,2500,2900", "\n".join(out), caps)


# ---------------------------------------------------------------- anim 2

COPE_PATH = "M48.0,225.5 L54.0,219.8 L60.1,218.0 L66.1,211.6 L72.2,203.4 L78.2,201.5 L84.2,199.3 L90.3,190.7 L96.3,185.5 L102.4,186.2 L108.4,181.3 L114.4,172.6 L120.5,171.7 L126.5,172.0 L132.6,164.0 L138.6,158.3 L144.6,160.5 L150.7,157.7 L156.7,148.9 L162.8,147.7 L168.8,149.9 L174.8,143.6 L180.9,137.2 L186.9,139.5 L193.0,138.8 L199.0,131.0 L205.0,128.9 L211.1,131.7 L217.1,127.4 L223.2,121.4 L229.2,122.8 L235.2,123.1 L241.3,117.2 L247.3,114.8 L253.4,116.8 L259.4,114.2 L265.4,109.5 L271.5,109.9 L277.5,110.1 L283.6,106.2 L289.6,104.4 L295.6,105.1 L301.7,103.0 L307.7,100.3 L313.8,100.5 L319.8,99.5 L325.8,96.8 L331.9,96.6 L337.9,96.4 L344.0,93.5 L350.0,92.6"
GRPO_PATH = "M48.0,224.3 L54.0,225.5 L60.1,220.5 L66.1,212.1 L72.2,211.6 L78.2,211.9 L84.2,204.1 L90.3,198.7 L96.3,200.8 L102.4,197.9 L108.4,189.7 L114.4,188.5 L120.5,190.2 L126.5,184.1 L132.6,178.3 L138.6,180.0 L144.6,179.0 L150.7,171.9 L156.7,169.8 L162.8,171.7 L168.8,167.7 L174.8,162.4 L180.9,163.0 L186.9,162.7 L193.0,157.6 L199.0,155.3 L205.0,156.2 L211.1,153.6 L217.1,149.8 L223.2,149.6 L229.2,148.8 L235.2,145.5 L241.3,144.1 L247.3,143.7 L253.4,141.3 L259.4,139.5 L265.4,139.4 L271.5,137.2 L277.5,134.9 L283.6,135.3 L289.6,134.2 L295.6,130.7 L301.7,130.6 L307.7,131.6 L313.8,128.2 L319.8,125.5 L325.8,127.8 L331.9,127.1 L337.9,122.2 L344.0,122.4 L350.0,125.3"


def anim_eff():
    out = []
    out.append('<svg viewBox="0 0 370 300" role="img" aria-label="Task score against training steps on AppWorld: '
               'CoPE reaches a score of 0.5 at about step 50, GRPO at about step 80, 1.6 times fewer steps.">')
    out.append('<defs><clipPath id="eff-clip"><rect class="reveal" x="48" y="30" width="0" height="230"/></clipPath></defs>')
    out.append('<text class="ttl" x="48" y="26">AppWorld</text>')
    out.append(f'<line x1="176" y1="21" x2="190" y2="21" stroke="{TERRA}" stroke-width="2.5"/><text class="small" x="194" y="25">CoPE (ours)</text>')
    out.append(f'<line x1="290" y1="21" x2="304" y2="21" stroke="{GREY}" stroke-width="2"/><text class="small" x="308" y="25">GRPO</text>')
    for y in (180, 110, 40):
        out.append(f'<line class="grid" x1="48" y1="{y}" x2="350" y2="{y}"/>')
    out.append('<line class="axis" x1="48" y1="250" x2="350" y2="250"/><line class="axis" x1="48" y1="250" x2="48" y2="40"/>')
    for s, x in ((0, 48), (25, 123.5), (50, 199), (75, 274.5), (100, 350)):
        out.append(f'<text class="tick" x="{x}" y="266" text-anchor="middle">{s}</text>')
    for v, y in (("0.0", 250), ("0.3", 180), ("0.6", 110), ("0.9", 40)):
        out.append(f'<text class="tick" x="42" y="{y + 4}" text-anchor="end">{v}</text>')
    out.append('<text class="small mut" x="199" y="286" text-anchor="middle">training steps</text>')
    out.append('<text class="small mut" x="14" y="145" text-anchor="middle" transform="rotate(-90 14 145)">task score</text>')
    out.append('<line class="target" x1="48" y1="133.3" x2="350" y2="133.3"/><text class="small mut" x="52" y="129">score 0.5</text>')
    out.append(f'<g clip-path="url(#eff-clip)"><path class="curve-g" d="{GRPO_PATH}"/><path class="curve-c" d="{COPE_PATH}"/></g>')
    out.append(f'<g class="mk mk-c sc sc-1 fin"><line class="drop" x1="199" y1="131" x2="199" y2="250"/><circle cx="199" cy="131" r="4" fill="{TERRA}"/></g>')
    out.append(f'<g class="mk mk-g sc sc-1 fin"><line class="drop" x1="289.6" y1="134.2" x2="289.6" y2="250"/><circle cx="289.6" cy="134.2" r="4" fill="{GREY}"/></g>')
    out.append('<g class="brk sc sc-1 fin pp"><path d="M199,230 v6 h90.6 v-6"/>'
               '<text x="244.3" y="225" text-anchor="middle"><tspan font-size="15" font-weight="700">1.6×</tspan><tspan font-size="11.5"> fewer steps</tspan></text></g>')
    out.append('</svg>')
    caps = [
        ('anim-cap', '<b>Sample efficiency on AppWorld.</b> Task score during training for CoPE and GRPO, same backbone and same rollout budget per step.'),
        ('anim-cap anim-cap-static', '<b>1.6&times; fewer steps.</b> CoPE reaches a 0.5 task score at about step 50; GRPO needs about 80. On ScienceWorld the gap grows to 2.0&times;.'),
    ]
    return figure("anim-eff", "3000,2600", "\n".join(out), caps)


# ---------------------------------------------------------------- anim 3

def anim_loss():
    ox, oy, c = 72, 52, 27
    n = 7
    out = []
    out.append('<svg viewBox="0 0 560 270" role="img" aria-label="Loss layout on a history-by-action grid: '
               'GRPO supervises the diagonal, CoPE adds an action forecasting loss that extends the diagonal into a band of width K.">')
    board = [f'<g class="board"><rect x="{ox}" y="{oy}" width="{n * c}" height="{n * c}" fill="#fff"/>']
    for i in range(n + 1):
        board.append(f'<line class="gl" x1="{ox + i * c}" y1="{oy}" x2="{ox + i * c}" y2="{oy + n * c}"/>')
        board.append(f'<line class="gl" x1="{ox}" y1="{oy + i * c}" x2="{ox + n * c}" y2="{oy + i * c}"/>')
    board.append('</g>')
    out.append("".join(board))
    for j in range(n):
        out.append(f'<text class="lbl" style="--i:{j}" x="{ox + j * c + c / 2}" y="44" text-anchor="middle">{sub("a", str(j))}</text>')
    for i in range(n):
        out.append(f'<text class="lbl" style="--i:{7 + i}" x="{ox - 8}" y="{oy + i * c + c / 2 + 4}" text-anchor="end">{sub("h", str(i))}</text>')
    out.append(f'<text class="lbl small" style="--i:14" x="{ox + n * c / 2}" y="24" text-anchor="middle">{sub("action a", "k", 9, 2.5)}</text>')
    out.append(f'<text class="lbl small" style="--i:14" x="26" y="{oy + n * c / 2}" text-anchor="middle" transform="rotate(-90 26 {oy + n * c / 2})">{sub("history h", "t", 9, 2.5)}</text>')
    for r in range(n):
        for k in range(4):
            col = r + k
            if col >= n:
                continue
            scenes = "sc-1 sc-2 sc-3" if k == 0 else "sc-2 sc-3"
            out.append(f'<g class="cell k{k} sc {scenes} fin pp" style="--i:{r}">'
                       f'<rect x="{ox + col * c + 1}" y="{oy + r * c + 1}" width="{c - 2}" height="{c - 2}" rx="4"/>'
                       f'<text x="{ox + col * c + c / 2}" y="{oy + r * c + c / 2 + 4}" text-anchor="middle">{k}</text></g>')
    # right panel
    out.append('<text class="formula sc sc-1 sc-2 sc-3 fin" x="300" y="96">&#8466; = &#8466;<tspan class="sub" dy="4">GRPO</tspan>'
               '<tspan class="fterm" dy="-4"> + &beta; &#8466;</tspan><tspan class="fterm sub" dy="4">AF</tspan></text>')
    out.append(f'<g class="tag-grpo sc sc-1 sc-2 sc-3 fin pp"><rect x="300" y="114" width="11" height="11" rx="2" fill="{TERRA}"/>'
               '<text class="small" x="318" y="124">k = 0 · the current action (GRPO)</text>'
               '<text class="small mut" x="300" y="142">policy gradient on all rollouts</text></g>')
    out.append('<g class="tag-af sc sc-2 sc-3 fin pp">'
               '<rect x="300" y="160" width="11" height="11" rx="2" fill="#4F7CAC"/><rect x="313" y="160" width="11" height="11" rx="2" fill="#86A7CC"/><rect x="326" y="160" width="11" height="11" rx="2" fill="#BBD0E6"/>'
               '<text class="small" x="344" y="170">k = 1 … K · forecast actions (AF)</text>'
               '<text class="small mut" x="300" y="188">one SFT loss, successful rollouts only</text></g>')
    out.append(f'<rect class="col-hl sc sc-3" x="{ox + 4 * c}" y="{oy + 1 * c}" width="{c}" height="{4 * c}" rx="5"/>')
    out.append('<g class="col-note sc sc-3 pp">'
               f'<text x="300" y="226" font-weight="600" fill="#B8801E">{rich("a", ("4",), " is forecast from h", ("1",), ", h", ("2",), ", h", ("3",), ", h", ("4",))}</text>'
               '<text class="small mut" x="300" y="244">a temporal ensemble of K + 1 histories</text></g>')
    out.append('</svg>')
    caps = [
        ('anim-cap', '<b>Where does the loss land?</b> Rows are histories <i>h<sub>t</sub></i>, columns are actions. A cell is supervised when the model is asked to produce that action from that history.'),
        ('anim-cap', '<b>GRPO: the diagonal.</b> Each history is scored only on its own next action, <i>k</i> = 0, with the policy-gradient loss on all rollouts.'),
        ('anim-cap anim-cap-static', '<b>CoPE: the band.</b> One extra SFT loss fits the next <i>K</i> actions on successful rollouts, so the diagonal extends to <i>k</i> = 1 &hellip; <i>K</i>. Nothing else changes.'),
        ('anim-cap', '<b>Read by column.</b> Each action is now forecast from <i>K</i> + 1 histories. The shared backbone absorbs this temporal ensemble, so forecasting improves acting.'),
    ]
    return figure("anim-loss", "1100,1700,2400,2600", "\n".join(out), caps)


# ---------------------------------------------------------------- figure wrapper

def figure(cls, durations, svg, caps):
    cap_html = "\n".join(f'        <span class="{c}" data-for="{i}">{t}</span>' for i, (c, t) in enumerate(caps))
    return f'''    <figure class="anim {cls}" data-durations="{durations}">
{svg}
      <div class="anim-bar">
        <div class="anim-dots" role="group" aria-label="Scenes"></div>
        <button class="anim-replay" type="button"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>Replay</button>
      </div>
      <figcaption class="anim-caps">
{cap_html}
      </figcaption>
    </figure>'''


def main():
    targets, eff, loss = anim_targets(), anim_eff(), anim_loss()
    row = f'    <div class="anim-row">\n{targets}\n{eff}\n    </div>'

    path = f"{WT}/index.html"
    html = open(path, encoding="utf-8").read()

    # head: stylesheet + script
    if "anim.css" not in html:
        html = html.replace('<link rel="stylesheet" href="static/css/style.css" />',
                            '<link rel="stylesheet" href="static/css/style.css" />\n'
                            '<link rel="stylesheet" href="static/css/anim.css" />\n'
                            '<script src="static/js/anim.js" defer></script>')

    # motivation figure -> animation row (re-entrant: replaces an existing row)
    existing_row = re.compile(r'    <div class="anim-row">.*?\n    </div>', re.S)
    motiv = re.compile(r'    <figure>\n      <img src="static/images/fig-motivation.png".*?</figure>', re.S)
    if existing_row.search(html):
        html = existing_row.sub(lambda m: row, html, count=1)
    else:
        assert motiv.search(html), "motivation figure not found"
        html = motiv.sub(lambda m: row, html, count=1)

    # method: loss animation before the overview figure
    existing_loss = re.compile(r'    <figure class="anim anim-loss".*?</figure>', re.S)
    marker = '    <figure>\n      <img src="static/images/fig-overview.png"'
    if existing_loss.search(html):
        html = existing_loss.sub(lambda m: loss, html, count=1)
    else:
        assert marker in html, "overview figure not found"
        html = html.replace(marker, loss + "\n\n" + marker, 1)

    open(path, "w", encoding="utf-8").write(html)

    # standalone preview for screenshots
    preview = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="file://{WT}/static/css/style.css" />
<link rel="stylesheet" href="file://{WT}/static/css/anim.css" />
<script src="file://{WT}/static/js/anim.js" defer></script>
</head><body><main><section><div class="wrap">
{row}
<div style="height:24px"></div>
{loss}
</div></section></main></body></html>'''
    open(f"{SCRATCH}/preview.html", "w", encoding="utf-8").write(preview)
    print("ok: index.html updated, preview written")


if __name__ == "__main__":
    main()
