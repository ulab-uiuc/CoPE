#!/usr/bin/env python
"""Render tau2 simulations as a browsable HTML trajectory viewer.

The numbers in TAU2_GRPO.md say *what* changed between checkpoints; they cannot show
*how*. Telecom's collapse in particular is only legible by reading a transcript: the
policy starts emitting bare `<tool_call>` tags with no JSON body and loops to the step
cap. This renders the conversations so those failures can be inspected directly, with
the degenerate turns flagged.

    python scripts/tau2_viz_trajectories.py \
        base=tau2_align_base2_airline grpo=tau2_align_grpo25_airline \
        --out runlogs/trajectories.html
"""

import argparse
import html
import json
import os
import re
import sys
from collections import Counter

SIM_DIR = "${TAU2_BENCH_DIR}/data/simulations"
# Degeneration signature measured across telecom: the hermes tool-call delimiter
# leaking into message content, with no structured tool_calls field alongside it.
DEGEN = re.compile(r"<tool_call>")


def load(tag):
    for p in (f"{SIM_DIR}/{tag}.json.json", f"{SIM_DIR}/{tag}.json/results.json",
              f"{SIM_DIR}/{tag}.json"):
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    return None


def classify(sim):
    r = (sim.get("reward_info") or {}).get("reward", 0.0) or 0.0
    if r >= 0.99:
        return "solved"
    if sim.get("termination_reason") == "max_steps":
        return "truncated"
    return "failed"


def is_degenerate(msg):
    return msg.get("role") == "assistant" and DEGEN.search(msg.get("content") or "") \
        and not msg.get("tool_calls")


def render_msg(m, idx):
    role = m.get("role", "?")
    content = (m.get("content") or "").strip()
    tcs = m.get("tool_calls") or []
    degen = is_degenerate(m)

    parts = []
    if content:
        parts.append(f'<div class="body">{html.escape(content)}</div>')
    for tc in tcs:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else "?")
        args = fn.get("arguments") or ""
        if isinstance(args, str) and len(args) > 400:
            args = args[:400] + " …"
        parts.append(
            f'<div class="tool"><span class="tname">{html.escape(str(name))}</span>'
            f'<span class="targs">{html.escape(str(args))}</span></div>')
    if not parts:
        parts.append('<div class="body empty">(empty)</div>')

    flag = '<span class="flag">degenerate</span>' if degen else ""
    return (f'<div class="msg {role}{" degen" if degen else ""}">'
            f'<div class="meta"><span class="role">{role}</span>'
            f'<span class="idx">#{idx}</span>{flag}</div>'
            f'{"".join(parts)}</div>')


def render_sim(sim, gid):
    outcome = classify(sim)
    msgs = sim.get("messages") or []
    ndegen = sum(1 for m in msgs if is_degenerate(m))
    ri = sim.get("reward_info") or {}
    bd = ri.get("reward_breakdown") or {}
    bd_txt = "  ".join(f"{k}={v}" for k, v in bd.items()) if bd else "—"

    head = (
        f'<div class="sumline">'
        f'<span class="pill {outcome}">{outcome}</span>'
        f'<span class="tid">{html.escape(str(sim.get("task_id", "?")))}</span>'
        f'<span class="stat">{len(msgs)} msgs</span>'
        f'<span class="stat">{html.escape(str(sim.get("termination_reason", "?")))}</span>'
        + (f'<span class="stat warn">{ndegen} degenerate</span>' if ndegen else "")
        + f'<span class="stat">reward {ri.get("reward", 0)}</span>'
        f'</div>'
        f'<div class="bd">{html.escape(bd_txt)}</div>')

    body = "".join(render_msg(m, i) for i, m in enumerate(msgs))
    return (f'<details class="sim" data-outcome="{outcome}" data-group="{gid}" '
            f'data-degen="{"y" if ndegen else "n"}">'
            f'<summary>{head}</summary><div class="thread">{body}</div></details>')


CSS = """
:root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f2f2ef;--line:#e0e0db;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#78776f;
--good:#0ca30c;--warning:#fab219;--critical:#d03b3b;--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242423;--line:#3a3a37;
--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8e8d84;
--s1:#3987e5;--s2:#d95926}}
:root[data-theme=dark]{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242423;
--line:#3a3a37;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8e8d84;
--s1:#3987e5;--s2:#d95926}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-1);color:var(--text-primary);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--surface-1);
border-bottom:1px solid var(--line);padding:14px 20px}
h1{margin:0 0 4px;font-size:16px;font-weight:600}
.sub{color:var(--text-secondary);font-size:12.5px}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
select,button{font:inherit;font-size:13px;padding:5px 9px;border:1px solid var(--line);
border-radius:6px;background:var(--surface-1);color:var(--text-primary);cursor:pointer}
.tiles{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px}
.tile{min-width:104px}
.tile .n{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.15}
.tile .l{font-size:11.5px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em}
main{padding:16px 20px 60px;max-width:1060px}
.grouphdr{font-size:12px;color:var(--text-muted);text-transform:uppercase;
letter-spacing:.05em;margin:22px 0 8px}
.sim{border:1px solid var(--line);border-radius:8px;margin-bottom:7px;
background:var(--surface-1);overflow:hidden}
.sim[hidden]{display:none}
summary{cursor:pointer;padding:9px 12px;list-style:none}
summary::-webkit-details-marker{display:none}
.sumline{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.pill{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;color:#fff}
.pill.solved{background:var(--good)}.pill.failed{background:var(--critical)}
.pill.truncated{background:var(--warning);color:#0b0b0b}
.tid{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--text-secondary);
max-width:430px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stat{font-size:11.5px;color:var(--text-muted);font-variant-numeric:tabular-nums}
.stat.warn{color:var(--critical);font-weight:600}
.bd{font-size:11px;color:var(--text-muted);font-family:ui-monospace,monospace;
padding:0 12px 8px;margin-top:-2px}
.thread{border-top:1px solid var(--line);padding:10px 12px;background:var(--surface-2)}
.msg{margin:0 0 8px;padding:8px 10px;border-radius:7px;background:var(--surface-1);
border-left:3px solid var(--line)}
.msg.assistant{border-left-color:var(--s1)}
.msg.user{border-left-color:var(--s2)}
.msg.tool{border-left-color:var(--text-muted)}
.msg.degen{background:color-mix(in srgb,var(--critical) 9%,var(--surface-1));
border-left-color:var(--critical)}
.meta{display:flex;gap:8px;align-items:center;margin-bottom:3px}
.role{font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
color:var(--text-secondary)}
.idx{font-size:10.5px;color:var(--text-muted);font-variant-numeric:tabular-nums}
.flag{font-size:10px;font-weight:600;color:var(--critical);
border:1px solid var(--critical);border-radius:3px;padding:0 5px}
.body{white-space:pre-wrap;word-break:break-word;font-size:13px}
.body.empty{color:var(--text-muted);font-style:italic}
.tool{margin-top:5px;font-family:ui-monospace,monospace;font-size:11.5px;
background:var(--surface-2);border-radius:5px;padding:5px 7px}
.tname{color:var(--s1);font-weight:600;margin-right:7px}
.targs{color:var(--text-secondary);word-break:break-all}
.none{color:var(--text-muted);padding:30px 0;font-style:italic}
"""

JS = """
const sims=[...document.querySelectorAll('.sim')];
function apply(){
  const g=document.getElementById('fg').value,o=document.getElementById('fo').value,
        d=document.getElementById('fd').value;
  sims.forEach(s=>{
    s.hidden=!((g==='all'||s.dataset.group===g)&&(o==='all'||s.dataset.outcome===o)
               &&(d==='all'||s.dataset.degen===d));
  });
  document.querySelectorAll('.grouphdr').forEach(h=>{
    let n=0,e=h.nextElementSibling;
    while(e&&e.classList.contains('sim')){if(!e.hidden)n++;e=e.nextElementSibling}
    h.hidden=n===0;
  });
  const vis=sims.filter(s=>!s.hidden).length;
  document.getElementById('nvis').textContent=vis;
}
['fg','fo','fd'].forEach(i=>document.getElementById(i).addEventListener('change',apply));
document.getElementById('expand').addEventListener('click',()=>{
  const any=sims.some(s=>!s.hidden&&!s.open);
  sims.filter(s=>!s.hidden).forEach(s=>s.open=any);
});
document.getElementById('theme').addEventListener('click',()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next=cur==='dark'?'light':(cur==='light'?'dark':
    (matchMedia('(prefers-color-scheme: dark)').matches?'light':'dark'));
  document.documentElement.setAttribute('data-theme',next);
});
apply();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="label=simulation_tag")
    ap.add_argument("--out", default="runlogs/trajectories.html")
    ap.add_argument("--limit", type=int, default=40, help="sims per group")
    a = ap.parse_args()

    blocks, tiles, total = [], [], Counter()
    opts = []
    for spec in a.groups:
        label, _, tag = spec.partition("=")
        tag = tag or label
        blob = load(tag)
        if blob is None:
            print(f"skip {label}: no data for {tag}", file=sys.stderr)
            continue
        sims = blob.get("simulations", [])
        c = Counter(classify(s) for s in sims)
        n_deg = sum(1 for s in sims if any(is_degenerate(m) for m in s.get("messages") or []))
        total.update(c)
        rate = 100 * c["solved"] / max(1, len(sims))
        tiles.append((label, rate, len(sims), n_deg))
        opts.append(label)
        # Solved first, then truncated, then failed -- the interesting reads are the
        # extremes, and burying them under 100 ordinary failures hides them.
        order = {"solved": 0, "truncated": 1, "failed": 2}
        sims = sorted(sims, key=lambda s: order[classify(s)])[:a.limit]
        blocks.append(f'<div class="grouphdr">{html.escape(label)} · {tag}</div>'
                      + "".join(render_sim(s, label) for s in sims))

    if not blocks:
        print("no data loaded", file=sys.stderr)
        return 1

    tile_html = "".join(
        f'<div class="tile"><div class="n">{r:.1f}%</div>'
        f'<div class="l">{html.escape(l)} solved</div></div>'
        f'<div class="tile"><div class="n">{d}</div>'
        f'<div class="l">{html.escape(l)} degenerate</div></div>'
        for l, r, n, d in tiles)

    gopts = "".join(f'<option value="{html.escape(o)}">{html.escape(o)}</option>' for o in opts)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tau2 trajectories</title><style>{CSS}</style></head><body>
<header>
<h1>tau²-bench trajectories</h1>
<div class="sub">Avg@4 test split · paper-era tau2 (c5b2d22) · gpt-4o-mini customer.
Turns flagged <b>degenerate</b> emit a bare <code>&lt;tool_call&gt;</code> with no
structured call — the failure mode behind telecom's collapse.</div>
<div class="tiles">{tile_html}</div>
<div class="controls">
<select id="fg"><option value="all">all runs</option>{gopts}</select>
<select id="fo"><option value="all">all outcomes</option><option value="solved">solved</option>
<option value="truncated">truncated</option><option value="failed">failed</option></select>
<select id="fd"><option value="all">any</option><option value="y">has degeneration</option>
<option value="n">clean</option></select>
<button id="expand">expand / collapse</button>
<button id="theme">theme</button>
<span class="stat"><b id="nvis">0</b> shown</span>
</div></header>
<main>{''.join(blocks)}</main>
<script>{JS}</script></body></html>"""

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(doc)
    print(f"wrote {a.out}  ({sum(total.values())} sims across {len(tiles)} runs)")
    for l, r, n, d in tiles:
        print(f"  {l:12s} solved {r:5.1f}%  n={n:3d}  degenerate {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
