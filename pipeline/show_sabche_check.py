#!/usr/bin/env python3
"""
show_sabche_check.py — eyeball the cleaned Sabche spans.

HTML page with ~1,500-char passages from 8 books (4 old, 4 new, one of the
old ones re-aligned), Sabche spans highlighted in alternating colours so two
adjacent headings stay distinguishable. For the re-aligned book the same
passage is shown a second time with the raw (pre-shift) YAML offsets.

Read-only: reads the sidecar, split and verdicts; writes only the HTML.

Usage
-----
    python scratch/sabche/scripts/show_sabche_check.py
    open scratch/sabche/sabche_check.html
"""

from __future__ import annotations

import argparse
import csv
import html
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SAB = ROOT / "data/processed/sabche"
WINDOW = 1500
COLOURS = ["#f6c945", "#7cc4f5"]


def read_split(path: Path) -> dict[str, str]:
    lines = [l for l in path.open(encoding="utf-8") if not l.startswith("#")]
    return {r["pecha_id"]: r["split"] for r in csv.DictReader(lines)}


def render(text, spans, lo, hi, dropped=()):
    """spans: sorted (s, e); dropped: (s, e) drawn as red dashed outline."""
    marks = [(s, e, "ok", n) for n, (s, e) in enumerate(spans)]
    marks += [(s, e, "drop", 0) for s, e in dropped]
    marks = sorted((max(s, lo), min(e, hi), k, n) for s, e, k, n in marks if s < hi and e > lo)
    out, cur = [], lo
    for s, e, kind, n in marks:
        if s < cur:  # never overlap in the sidecar; guard anyway
            s = cur
        if e <= s:
            continue
        out.append(f'<span class="ctx">{html.escape(text[cur:s])}</span>')
        if kind == "ok":
            out.append(f'<span class="sb" style="background:{COLOURS[n % 2]}" '
                       f'title="span {n + 1}: {s}-{e}">{html.escape(text[s:e])}</span>')
        else:
            out.append(f'<span class="drop" title="dropped (re-aligned, unverified): {s}-{e}">'
                       f'{html.escape(text[s:e])}</span>')
        cur = e
    out.append(f'<span class="ctx">{html.escape(text[cur:hi])}</span>')
    return "".join(out).replace("\n", "<br>")


def pick_window(text, spans, rng):
    """Window of WINDOW chars around a span in the middle 20-80% of the
    book's text (not its front-matter outline), snapped to a line start.
    Prefers passages that mix headings with body text: 2-12 headings, and
    headings covering at most half the window."""
    L = len(text)
    mid = [sp for sp in spans if 0.2 * L <= sp[0] <= 0.8 * L] or spans
    cands = []
    for s, _ in rng.sample(mid, min(20, len(mid))):
        lo = max(0, s - 250)
        nl = text.rfind("\n", 0, lo)
        lo = nl + 1 if nl != -1 and lo - nl < 200 else lo
        hi = min(L, lo + WINDOW)
        inside = [(max(a, lo), min(b, hi)) for a, b in spans if a < hi and b > lo]
        cover = sum(b - a for a, b in inside) / max(hi - lo, 1)
        good = 2 <= len(inside) <= 12 and cover <= 0.5
        cands.append((good, len(inside), lo, hi))
    good = [c for c in cands if c[0]]
    best = max(good or cands, key=lambda c: c[1])
    return best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--realigned", default="P000063",
                    help="old-batch re-aligned book to include")
    ap.add_argument("--out", default=str(ROOT / "scratch/sabche/sabche_check.html"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    audit = {r["pecha_id"]: r for r in
             csv.DictReader((ROOT / "data/processed/tsawa/tsawa_audit.csv").open(encoding="utf-8"))}
    split = read_split(SAB / "sabche_split_v2_frozen.csv")
    verdict = {r["pecha_id"]: r for r in
               csv.DictReader((SAB / "sabche_book_verdicts.csv").open(encoding="utf-8"))}
    clean, raw, dropped, acts = (defaultdict(list) for _ in range(4))
    for r in csv.DictReader((SAB / "sabche_spans_clean.csv").open(encoding="utf-8")):
        p = r["pecha_id"]
        if p not in split:
            continue
        raw[p].append((int(r["raw_start"]), int(r["raw_end"])))
        if r["dropped"] == "True":
            dropped[p].append((int(r["start"]), int(r["end"])))
        else:
            clean[p].append((int(r["start"]), int(r["end"])))
            acts[p].append(r["action"])

    def eligible(batch, verdicts):
        return sorted(p for p in split
                      if (p[0] == "P") == (batch == "old")
                      and verdict[p]["verdict"] in verdicts and len(clean[p]) >= 20)

    if verdict.get(args.realigned, {}).get("verdict") != "realigned" or args.realigned not in split:
        raise SystemExit(f"{args.realigned} is not a kept re-aligned book")
    old = [args.realigned] + rng.sample([p for p in eligible("old", {"keep"})], 3)
    new = rng.sample(eligible("new", {"keep"}), 4)

    def card(pid):
        text = Path(audit[pid]["base_path"]).read_text(encoding="utf-8")
        sp = sorted(clean[pid])
        lo, hi = pick_window(text, sp, rng)
        v = verdict[pid]
        a = acts[pid]
        n_in = sum(1 for s, e in sp if s < hi and e > lo)
        shifted = sum("shift" in x for x in a)
        snapped = sum(("start-" in x or "end+" in x) for x in a)
        meta = (f"<b>{pid}</b> · {'old' if pid[0] == 'P' else 'new'} batch · "
                f"<b>{split[pid]}</b> split · verdict: <b>{v['verdict']}</b> · "
                f"{len(sp)} spans in book ({shifted} shifted, {snapped} snapped, "
                f"{len(dropped[pid])} dropped) · heading-like "
                f"{float(v['heading_rate_raw']):.0%} raw → {float(v['heading_rate_final']):.0%} · "
                f"chars {lo:,}–{hi:,}, {n_in} span{'s' if n_in != 1 else ''} shown")
        body = f"<div class='txt'>{render(text, sp, lo, hi, dropped[pid])}</div>"
        if v["verdict"] == "realigned":
            rsp = sorted(raw[pid])
            body = (f"<div class='lab'>Cleaned (shifted + snapped) — what the model trains on</div>{body}"
                    f"<div class='lab raw'>Same passage, raw Sabche.yml offsets (before re-alignment)</div>"
                    f"<div class='txt'>{render(text, rsp, lo, hi)}</div>")
        return f"<div class='ex'><div class='pid'>{meta}</div>{body}</div>"

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sabche span check</title>
<style>
:root{{--bg:#fff;--fg:#222;--muted:#777;--ctx:#999;--box:#ddd;--stats:#f4f4f4}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16181c;--fg:#e6e6e6;--muted:#9aa0a6;
  --ctx:#8a8f96;--box:#33363c;--stats:#202328}}}}
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 16px;
      color:var(--fg);background:var(--bg)}}
h1{{font-size:1.4rem}} h2{{margin-top:2rem;font-size:1.15rem}}
.stats{{background:var(--stats);padding:.8rem 1rem;border-radius:6px;line-height:1.6}}
.ex{{border:1px solid var(--box);border-radius:6px;padding:.8rem;margin:1rem 0}}
.pid{{font-size:.8rem;color:var(--muted);margin-bottom:.4rem;line-height:1.5}}
.lab{{font-size:.8rem;font-weight:600;margin:.6rem 0 .2rem}} .lab.raw{{color:#c0392b}}
.txt{{font-family:"Noto Serif Tibetan","Jomolhari","Microsoft Himalaya",serif;
      font-size:1.35rem;line-height:2.4;overflow-wrap:anywhere}}
.ctx{{color:var(--ctx)}}
.sb{{color:#111;border-radius:3px;padding:2px 0}}
.drop{{outline:2px dashed #e74c3c;border-radius:3px;color:var(--fg)}}
</style></head><body>
<h1>Sabche span check — cleaned spans, split v2</h1>
<div class="stats">
Source: <code>data/processed/sabche/sabche_spans_clean.csv</code> (dropped=False), books from
<code>sabche_split_v2_frozen.csv</code>. Grey is body text. Sabche spans alternate
yellow and blue, so a colour change with no grey between is a boundary between two
separate headings. A <span class="drop">red dashed</span> box is a span dropped from the
dataset (re-aligned but unverified). Spans stop <i>before</i> the shad by convention.
Hover a span for its offsets.<br>
Seed {args.seed}: 4 old-batch books (incl. re-aligned {args.realigned}) and 4 new-batch
books, each with ≥ 20 spans; ~{WINDOW:,} characters from the middle of each book.
</div>
<h2>Old batch</h2>
{''.join(card(p) for p in old)}
<h2>New batch</h2>
{''.join(card(p) for p in new)}
</body></html>"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page, encoding="utf-8")
    print("old:", old, "\nnew:", new)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
