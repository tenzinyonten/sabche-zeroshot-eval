#!/usr/bin/env python3
"""
zeroshot_run.py - zero-shot SABCHE (outline heading) span extraction with an LLM,
scored against the Sabche v2 gold spans.

Pipeline: 16,000-character windows overlapping by 2,000 and cut at a shad ->
one LLM call per window -> the model returns anchors (label/head/tail/frame),
not offsets -> the locator maps anchors back to character offsets -> spans from
overlapping windows are deduplicated -> per-book IoU@0.5 greedy one-to-one
scoring (character offsets, inclusive).

    # Gemini (needs GEMINI_API_KEY)
    python scripts/zeroshot_run.py --texts-dir data/raw_opf --books P000067 \
        --provider gemini --model gemini-3.1-flash-lite --out runs/gemini

    # Claude (needs ANTHROPIC_API_KEY)
    python scripts/zeroshot_run.py --texts-dir data/raw_opf --books P000067 \
        --provider anthropic --model claude-sonnet-5 --out runs/claude

Raw replies are cached one file per window, so a re-run costs nothing for windows
that already have one. --stop-after N pauses after N books that needed API calls;
--locate-only re-scores the cache without any API call.

API keys are read from the environment only and are never written to disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gemini_chunks import make_chunks  # noqa: E402
from gemini_locate import (  # noqa: E402
    canonical_label,
    locate_parsed_quote,
    parse_quote_dump,
)

GOLD = ROOT / "data" / "sabche_gold_v2.csv"
MANIFEST = ROOT / "data" / "books_manifest.csv"
PROMPT = ROOT / "prompts" / "gemini_sabche_anchors_v1.md"
MAX_SPAN = 2000  # head-to-tail ceiling for the locator


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_prompt(path: Path) -> str:
    """The prompt is what sits between the two --- rulers in the prompt file."""
    parts = path.read_text(encoding="utf-8").split("\n---\n")
    if len(parts) < 3:
        raise SystemExit(f"{path}: expected the prompt between two --- rulers")
    return parts[1].strip() + "\n"


def load_book_text(texts_dir: Path, pecha_id: str) -> str:
    """base/v001.txt of an OpenPecha .opf checkout (plain or nested layout)."""
    for rel in (f"{pecha_id}.opf/{pecha_id}.opf/base/v001.txt",
                f"{pecha_id}.opf/base/v001.txt"):
        p = texts_dir / rel
        if p.is_file():
            return p.read_text(encoding="utf-8")
    raise SystemExit(f"{pecha_id}: base/v001.txt not found under {texts_dir} "
                     "(see scripts/fetch_texts.py)")


def load_gold(ids: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Gold spans as INCLUSIVE (start, end); the CSV stores end-exclusive."""
    gold: dict[str, list[tuple[int, int]]] = {b: [] for b in ids}
    for r in csv.DictReader(open(GOLD, encoding="utf-8")):
        if r["pecha_id"] in gold:
            gold[r["pecha_id"]].append((int(r["start"]), int(r["end_exclusive"]) - 1))
    return {k: sorted(v) for k, v in gold.items()}


def split_of() -> dict[str, str]:
    return {r["pecha_id"]: r["split"] for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def make_config(types, temperature: float):
    """thinking_level is the new SDK spelling; older ones take a budget."""
    kw = dict(temperature=temperature, response_mime_type="application/json")
    try:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"), **kw)
    except Exception:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024), **kw)


def call_gemini(client, types, model: str, prompt: str, chunk: str,
                temperature: float, tries: int = 5):
    cfg = make_config(types, temperature)
    for attempt in range(tries):
        try:
            r = client.models.generate_content(
                model=model, contents=prompt + chunk, config=cfg)
            usage = getattr(r, "usage_metadata", None)
            return r.text or "", {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "finish": str(getattr(r.candidates[0], "finish_reason", "")) if r.candidates else "",
            }
        except Exception as exc:  # noqa: BLE001 — the SDK raises many shapes
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                if "PerDay" in msg or "per day" in msg.lower():
                    raise SystemExit(f"daily quota reached, stopping: {msg[:200]}")
                wait = 30 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return "", {"finish": "failed"}


# Same schema the quotation benchmark pins for Claude: frame optional, the rest
# required, nothing else allowed.
SPAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "frame": {"type": "string"},
                    "head": {"type": "string"},
                    "tail": {"type": "string"},
                },
                "required": ["label", "head", "tail"],
            },
        }
    },
    "required": ["spans"],
}


def call_claude(client, model: str, prompt: str, chunk: str, tries: int = 5):
    """Sonnet 5 rejects temperature/top_p and takes adaptive thinking."""
    import anthropic
    # The prompt is identical for every window, so cache it as a prefix: the
    # window text goes in a second block after the breakpoint. Same tokens are
    # sent either way - this only changes billing ($0.20/MTok read vs $2.00).
    content = [
        {"type": "text", "text": prompt,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": chunk},
    ]
    for attempt in range(tries):
        try:
            with client.messages.stream(
                model=model, max_tokens=64000,
                thinking={"type": "adaptive"},
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": SPAN_SCHEMA}},
                messages=[{"role": "user", "content": content}],
            ) as stream:                      # 64K max_tokens requires streaming
                r = stream.get_final_message()
            text = "".join(b.text for b in r.content if b.type == "text")
            return text, {"prompt_tokens": r.usage.input_tokens,
                          "output_tokens": r.usage.output_tokens,
                          "cache_write": getattr(r.usage, "cache_creation_input_tokens", 0),
                          "cache_read": getattr(r.usage, "cache_read_input_tokens", 0),
                          "finish": str(r.stop_reason)}
        except anthropic.RateLimitError as exc:
            wait = int(getattr(exc.response, "headers", {}).get("retry-after", 30))
            print(f"    rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
        except anthropic.BadRequestError as exc:
            if "ttl" in str(exc).lower() and content[0]["cache_control"].get("ttl"):
                content[0]["cache_control"] = {"type": "ephemeral"}   # 5-min default
                continue
            raise
        except anthropic.APIStatusError as exc:
            if exc.status_code < 500 or attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return "", {"finish": "failed"}


# --------------------------------------------------------------------------
# locate + score
# --------------------------------------------------------------------------

def locate_window(wtext: str, raw: str, label: str) -> list[tuple[int, int]]:
    """Anchors -> inclusive offsets inside the window, in emission order.

    Forward cursor, with one retry from the window start before dropping a
    span — the resync the quotation run measured as F1 0.359 -> 0.418.
    """
    try:
        quotes = parse_quote_dump(raw)
    except Exception:
        return []
    out, seen, cursor = [], set(), 0
    for qt in quotes:
        if canonical_label(qt.label) != label:
            continue
        hit = locate_parsed_quote(wtext, qt, min_score=0.85,
                                  from_index=cursor, max_span=MAX_SPAN)
        if hit is None:
            hit = locate_parsed_quote(wtext, qt, min_score=0.85,
                                      from_index=0, max_span=MAX_SPAN)
        if hit is None or (hit.start, hit.end) in seen:
            continue
        seen.add((hit.start, hit.end))
        out.append((hit.start, hit.end))
        cursor = max(cursor, hit.end + 1)
    return out


def dedupe(spans: list[tuple[int, int]], thr: float = 0.5) -> list[tuple[int, int]]:
    """Windows overlap by 2,000 characters, so the same span can be found twice."""
    kept: list[tuple[int, int]] = []
    for s in sorted(spans):
        if any(iou(s, k) >= thr for k in kept):
            continue
        kept.append(s)
    return kept


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def score(gold: list[tuple[int, int]], pred: list[tuple[int, int]], thr: float = 0.5):
    cands = sorted(((iou(g, p), gi, pi)
                    for gi, g in enumerate(gold) for pi, p in enumerate(pred)
                    if iou(g, p) >= thr), reverse=True)
    ug, up = set(), set()
    for _v, gi, pi in cands:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
    tp = len(ug)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp,
            "fp": len(pred) - tp, "fn": len(gold) - tp,
            "n_gold": len(gold), "n_pred": len(pred)}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--texts-dir", default="data/raw_opf",
                    help="directory of OpenPecha .opf checkouts (scripts/fetch_texts.py)")
    ap.add_argument("--out", default="runs/zeroshot")
    ap.add_argument("--prompt", default=str(PROMPT))
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--window", type=int, default=16000)
    ap.add_argument("--overlap", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rpm", type=float, default=15.0)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="per book; 0 means all")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop after this many books that needed API calls, "
                         "printing the running score; re-run the same command "
                         "to continue, cached windows cost nothing")
    ap.add_argument("--locate-only", action="store_true",
                    help="score the cached replies, make no API calls")
    args = ap.parse_args()

    prompt = load_prompt(Path(args.prompt))
    texts = {b: load_book_text(Path(args.texts_dir), b) for b in args.books}
    gold = load_gold(args.books)
    splits = split_of()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    client = types = None
    if not args.locate_only:
        if args.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()          # resolves ANTHROPIC_API_KEY
        else:
            if not os.environ.get("GEMINI_API_KEY"):
                raise SystemExit("GEMINI_API_KEY is not set")
            from google import genai
            from google.genai import types as _types
            types = _types
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    gap = 60.0 / args.rpm
    summary = {}
    books_called = 0
    stopped_at = None
    for book in args.books:
        if stopped_at is not None:
            break
        text = texts[book]
        chunks = make_chunks(text, book, args.window, args.window - args.overlap)
        if args.max_windows:
            chunks = chunks[: args.max_windows]
        bdir = outdir / book
        bdir.mkdir(exist_ok=True)
        print(f"\n{book} ({splits.get(book, '?')}): {len(text):,} chars, "
              f"{len(chunks)} windows, {len(gold[book])} gold spans", flush=True)

        pred, n_called, usage_tot = [], 0, {"prompt": 0, "output": 0}
        for i, c in enumerate(chunks):
            cache = bdir / f"{i:04d}.json"
            if cache.exists():
                rec = json.loads(cache.read_text(encoding="utf-8"))
            elif args.locate_only:
                continue
            else:
                t0 = time.time()
                if args.provider == "anthropic":
                    raw, meta = call_claude(client, args.model, prompt, c["text"])
                else:
                    raw, meta = call_gemini(client, types, args.model, prompt,
                                            c["text"], args.temperature)
                rec = {"book": book, "window": i, "start": c["start"],
                       "end": c["end"], "raw": raw, **meta}
                cache.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                n_called += 1
                print(f"  window {i}: {meta.get('prompt_tokens')} in / "
                      f"{meta.get('output_tokens')} out", flush=True)
                time.sleep(max(0.0, gap - (time.time() - t0)))
            usage_tot["prompt"] += rec.get("prompt_tokens") or 0
            usage_tot["output"] += rec.get("output_tokens") or 0
            local = locate_window(c["text"], rec.get("raw", ""), "SABCHE")
            pred.extend((rec["start"] + s, rec["start"] + e) for s, e in local)

        pred = dedupe(pred)
        res = score(gold[book], pred)
        res.update(windows=len(chunks), api_calls=n_called, **usage_tot)
        summary[book] = res
        print(f"  IoU@0.5  F1={res['f1']:.3f}  P={res['precision']:.3f} "
              f"R={res['recall']:.3f}  pred={res['n_pred']} gold={res['n_gold']}")
        (outdir / f"{book}_spans.json").write_text(
            json.dumps({"book": book, "spans": pred}, ensure_ascii=False), encoding="utf-8")
        if n_called:
            books_called += 1
        if args.stop_after and books_called >= args.stop_after:
            stopped_at = book

    done = [b for b in args.books if b in summary]
    all_gold = [g for b in done for g in gold[b]]
    micro = {"tp": sum(s["tp"] for s in summary.values()),
             "n_pred": sum(s["n_pred"] for s in summary.values()),
             "n_gold": len(all_gold), "books": len(done)}
    p = micro["tp"] / micro["n_pred"] if micro["n_pred"] else 0.0
    r = micro["tp"] / micro["n_gold"] if micro["n_gold"] else 0.0
    micro["precision"], micro["recall"] = p, r
    micro["f1"] = 2 * p * r / (p + r) if p + r else 0.0
    summary["_micro"] = micro
    print(f"\n{len(done)} book(s) scored  IoU@0.5  F1={micro['f1']:.3f}  "
          f"P={p:.3f} R={r:.3f} ({micro['tp']}/{micro['n_gold']} gold found)")
    if stopped_at is not None:
        left = [b for b in args.books if b not in summary]
        print(f"\nstopped after {stopped_at} (--stop-after {args.stop_after}). "
              f"{len(left)} book(s) left: {' '.join(left) if left else 'none'}")
        print("re-run the same command to continue; cached windows cost nothing.")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written to {outdir}/summary.json")


if __name__ == "__main__":
    main()
