#!/usr/bin/env python3
"""Generate the loss-computation animation (anim 4), splice it into Method, drop the static overview."""
import re
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the gh-pages checkout
SCRATCH = os.path.join(WT, "build")                                # previews land here (git-ignored)
os.makedirs(SCRATCH, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_anims import WT, SCRATCH, TERRA, OK, GREY, rich, sub, arrow_head, figure  # noqa: E402

AF1 = "#4F7CAC"


def anim_flow():
    slots = [("o", 0), ("a", 0), ("o", 1), ("a", 1), ("o", 2), ("a", 2),
             ("o", 3), ("a", 3), ("o", 4), ("a", 4), ("o", 5), ("a", 5)]
    xs = [118 + 38 * i for i in range(12)]
    cy_top, cy_bot = 64, 104
    out = []
    out.append('<svg viewBox="0 0 560 300" role="img" aria-label="How the CoPE loss is computed: the GRPO loss uses the '
               'current action of every rollout, the action forecasting loss uses the next K actions of successful rollouts only, '
               'and both update one shared LLM.">')
    out.append('<text class="ttl sc sc-0 fin" x="20" y="26">Two rollouts, one shared backbone</text>')
    out.append(f'<text class="ttl sc sc-1" x="20" y="26">{rich("ℒ", ("GRPO",), ": the current action, all rollouts", size=10.5)}</text>')
    out.append(f'<text class="ttl sc sc-2" x="20" y="26">{rich("ℒ", ("AF",), ": forecast the next K actions, successful rollouts only", size=10.5)}</text>')
    out.append(f'<text class="ttl sc sc-3" x="20" y="26">{rich("ℒ = ℒ", ("GRPO",), " + λ ℒ", ("AF",), size=10.5)}</text>')

    # shared history: both rollouts have seen o0 a0 o1 a1 o2 when the losses apply
    hx0, hx1 = xs[0] - 16, xs[4] + 16
    out.append(f'<rect class="hist-box sc sc-0 sc-1 sc-2 sc-3 fin pp" x="{hx0}" y="{cy_top - 17}" width="{hx1 - hx0}" height="{cy_bot - cy_top + 34}" rx="9"/>')
    out.append(f'<text class="hist-lbl sc sc-0 sc-1 sc-2 sc-3 fin small mut pp" x="{(hx0 + hx1) / 2}" y="44" text-anchor="middle">{rich("history h", ("2",), "  (seen so far)", size=8.5, dy=2.5)}</text>')

    for row, cy in (("top", cy_top), ("bot", cy_bot)):
        fail = " fl-fail" if row == "top" else ""
        for i in range(11):
            out.append(f'<line class="link{fail}" style="--i:{i}" x1="{xs[i] + 12}" y1="{cy}" x2="{xs[i + 1] - 12}" y2="{cy}"/>')
        for i, (kind, idx) in enumerate(slots):
            x = xs[i]
            label = sub(kind, str(idx), 8.5, 3)
            extra = fail
            if kind == "a" and idx == 2:
                extra += " fl-a2"
            elif kind == "a" and idx >= 3:
                extra += " fl-af" if row == "bot" else " fl-dim"
            kstyle = f";--k:{idx - 3}" if (kind == "a" and idx >= 3) else ""
            if kind == "o":
                out.append(f'<g class="node o-node pp{extra}" style="--i:{i}{kstyle}"><circle cx="{x}" cy="{cy}" r="11"/>'
                           f'<text x="{x}" y="{cy + 4}" text-anchor="middle">{label}</text></g>')
            else:
                out.append(f'<g class="node a-node pp{extra}" style="--i:{i}{kstyle}">'
                           f'<rect x="{x - 11}" y="{cy - 11}" width="22" height="22" rx="5"/>'
                           f'<text x="{x}" y="{cy + 4}" text-anchor="middle">{label}</text></g>')
    out.append(f'<g class="badge pp"><path d="M26,{cy_top - 5} l10,10 M36,{cy_top - 5} l-10,10" stroke="{GREY}" stroke-width="2.2" stroke-linecap="round"/>'
               f'<text class="small mut" x="46" y="{cy_top + 4}">Y(τ) = 0</text></g>')
    out.append(f'<g class="badge pp"><path d="M25,{cy_bot} l4,4 l9,-10" fill="none" stroke="{OK}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
               f'<text class="small" x="46" y="{cy_bot + 4}" fill="{OK}" font-weight="600">Y(τ) = 1</text></g>')

    xa2, xa3, xa4, xa5 = xs[5], xs[7], xs[9], xs[11]
    out.append(f'<text class="k0 sc sc-1 sc-2 sc-3 fin" x="{xa2}" y="44" text-anchor="middle" font-size="10.5" fill="{TERRA}" font-weight="700">k = 0</text>')
    for k, x in enumerate((xa3, xa4, xa5), start=1):
        out.append(f'<text class="k123 sc sc-2 sc-3 fin" style="--k:{k - 1}" x="{x}" y="44" text-anchor="middle" font-size="10.5" fill="{AF1}" font-weight="700">k = {k}</text>')
    out.append(f'<text class="fl-note sc sc-2 sc-3 fin small mut" x="{xa5 + 11}" y="88" text-anchor="end">{rich("a", ("3:5",), " of a failed rollout: not used", size=8, dy=2.5)}</text>')

    by, bh = 150, 26
    out.append(f'<path class="ar ar-grpo drawn sc sc-1 sc-2 sc-3 fin" d="M{xa2},{cy_top + 11} V{cy_bot - 11} M{xa2},{cy_bot + 11} V{by - 2}" pathLength="1"/>')
    out.append(f'<path class="hd hd-grpo sc sc-1 sc-2 sc-3 fin" d="{arrow_head((xa2, by - 1), (0, 1), 8, 3.5)}" fill="{TERRA}"/>')
    for k, x in enumerate((xa3, xa4, xa5)):
        out.append(f'<path class="ar ar-af drawn sc sc-2 sc-3 fin" style="--k:{k}" d="M{x},{cy_bot + 11} V{by - 2}" pathLength="1"/>')
        out.append(f'<path class="hd hd-af sc sc-2 sc-3 fin" style="--k:{k}" d="{arrow_head((x, by - 1), (0, 1), 8, 3.5)}" fill="{AF1}"/>')
    out.append(f'<g class="box box-pi sc sc-1 sc-2 sc-3 fin pp"><rect x="{xa2 - 20}" y="{by}" width="40" height="{bh}" rx="6"/>'
               f'<text x="{xa2}" y="{by + 17}" text-anchor="middle">{sub("π", "θ", 9, 3)}</text></g>')
    for k, x in enumerate((xa3, xa4, xa5)):
        out.append(f'<g class="box box-p sc sc-2 sc-3 fin pp" style="--k:{k}"><rect x="{x - 17}" y="{by}" width="34" height="{bh}" rx="6"/>'
                   f'<text x="{x}" y="{by + 17}" text-anchor="middle">{sub("p", "θ", 9, 3)}</text></g>')
    out.append(f'<text class="small mut sc sc-1 sc-2 sc-3 fin" x="{xa2 - 26}" y="{by + 17}" text-anchor="end">{sub("act a", "2", 8, 2.5)}</text>')

    ly, lh = 206, 40
    xg, xaf = 270, 412
    out.append(f'<path class="ar ar-pi drawn sc sc-1 sc-2 sc-3 fin" d="M{xa2},{by + bh} L{xg + 14},{ly - 2}" pathLength="1"/>')
    out.append(f'<path class="hd hd-pi sc sc-1 sc-2 sc-3 fin" d="{arrow_head((xg + 14, ly - 1), (xg + 14 - xa2, ly - by - bh), 8, 3.5)}" fill="{TERRA}"/>')
    for k, x in enumerate((xa3, xa4, xa5)):
        tx = xaf - 10 + 10 * k
        out.append(f'<path class="ar ar-p drawn sc sc-2 sc-3 fin" style="--k:{k}" d="M{x},{by + bh} L{tx},{ly - 2}" pathLength="1"/>')
        out.append(f'<path class="hd hd-p sc sc-2 sc-3 fin" style="--k:{k}" d="{arrow_head((tx, ly - 1), (tx - x, ly - by - bh), 8, 3.5)}" fill="{AF1}"/>')
    out.append(f'<g class="lossbox box-lg sc sc-1 sc-2 sc-3 fin pp"><rect x="{xg - 52}" y="{ly}" width="104" height="{lh}" rx="7"/>'
               f'<text class="l1" x="{xg}" y="{ly + 17}" text-anchor="middle">{sub("ℒ", "GRPO", 9.5, 3)}</text>'
               f'<text class="l2" x="{xg}" y="{ly + 32}" text-anchor="middle">all rollouts</text></g>')
    out.append(f'<g class="lossbox box-la sc sc-2 sc-3 fin pp"><rect x="{xaf - 58}" y="{ly}" width="116" height="{lh}" rx="7"/>'
               f'<text class="l1" x="{xaf}" y="{ly + 17}" text-anchor="middle">{sub("ℒ", "AF", 9.5, 3)}</text>'
               f'<text class="l2" x="{xaf}" y="{ly + 32}" text-anchor="middle">only Y(τ) = 1</text></g>')
    out.append(f'<text class="plus sc sc-3 fin pp" x="{(xg + 52 + xaf - 58) / 2}" y="{ly + 25}" text-anchor="middle" font-size="15" font-weight="700">+ λ</text>')

    out.append(f'<path class="ar ar-sum drawn sc sc-3 fin" d="M{xg},{ly + lh} V260 M{xaf},{ly + lh} V260" pathLength="1"/>')
    out.append(f'<g class="llm pp"><rect class="llm-bg" x="118" y="262" width="429" height="26" rx="7"/>'
               f'<rect class="llm-hl sc sc-3 fin" x="118" y="262" width="429" height="26" rx="7"/>'
               f'<text x="332.5" y="279" text-anchor="middle">Shared LLM backbone θ   ·   {rich("π", ("θ",), " acts,  ", size=8.5, dy=2.5)}{rich("p", ("θ",), " forecasts", size=8.5, dy=2.5)}</text></g>')
    out.append('</svg>')

    caps = [
        ('anim-cap', '<b>Two rollouts, one backbone.</b> Each rollout is scored once at the end: <i>Y</i>(&tau;) = 0 for failure, 1 for success. The same LLM produces every action.'),
        ('anim-cap', '<b>&#8466;<sub>GRPO</sub> on all rollouts.</b> The policy &pi;<sub>&theta;</sub> is updated on the current action <i>a<sub>t</sub></i> of every rollout, with the group-relative advantage from <i>Y</i>.'),
        ('anim-cap', '<b>&#8466;<sub>AF</sub> on successful rollouts only.</b> From the same history, the backbone also forecasts <i>a</i><sub>t+1</sub>, <i>a</i><sub>t+2</sub>, <i>a</i><sub>t+3</sub> and is fit to them by supervised learning. Failed rollouts contribute nothing here.'),
        ('anim-cap anim-cap-static', '<b>&#8466; = &#8466;<sub>GRPO</sub> + &lambda; &#8466;<sub>AF</sub>.</b> Both losses update the same parameters, so acting and forecasting share every weight of the backbone.'),
    ]
    return figure("anim-flow", "1400,2400,2800,2400", "\n".join(out), caps)


def main():
    path = f"{WT}/index.html"
    html = open(path, encoding="utf-8").read()
    flow = anim_flow()
    html = re.sub(r'    <figure class="anim anim-flow".*?</figure>\n\n', "", html, count=1, flags=re.S)
    assert '    <figure class="anim anim-loss"' in html
    html = html.replace('    <figure class="anim anim-loss"', flow + "\n\n" + '    <figure class="anim anim-loss"', 1)
    html = re.sub(r'    <figure>\n      <img src="static/images/fig-overview.png".*?</figure>\n\n', "", html, count=1, flags=re.S)
    open(path, "w", encoding="utf-8").write(html)

    preview = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="file://{WT}/static/css/style.css" />
<link rel="stylesheet" href="file://{WT}/static/css/anim.css" />
<script src="file://{WT}/static/js/anim.js" defer></script>
</head><body><main><section><div class="wrap">
<div style="max-width:640px;margin:0 auto">{flow}</div>
</div></section></main></body></html>'''
    open(f"{SCRATCH}/preview_flow.html", "w", encoding="utf-8").write(preview)
    print("flow animation written; overview figure removed")


if __name__ == "__main__":
    main()
