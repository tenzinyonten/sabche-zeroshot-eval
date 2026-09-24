#!/usr/bin/env python3
"""
claude_batch.py - run Claude Sonnet 5 on the sabche test windows through the
Message Batches API (half price, asynchronous), reusing the exact windows, prompt
and per-window cache format of scripts/zeroshot_run.py, so its scorer works
unchanged. Needs the book texts (scripts/fetch_texts.py) and ANTHROPIC_API_KEY.

Nothing is sent unless `create` is run with --yes. Without --yes it builds the
requests and reports what it would send.

    # 1. offline: how many windows, split into groups, estimated cost
    python scripts/claude_batch.py plan --groups 3

    # 2. dry run of one group (still offline), then submit it
    python scripts/claude_batch.py create --group g1 --books P000083 ...
    python scripts/claude_batch.py create --group g1 --books P000083 ... --yes

    # 3. wait, then download one file per window
    python scripts/claude_batch.py poll  --group g1
    python scripts/claude_batch.py fetch --group g1

    # 4. score (no API key needed)
    python scripts/zeroshot_run.py --books ... --out runs/claude/test --locate-only

The batch id is written to disk the moment the batch is created, a group cannot be
submitted twice, and windows that already have a saved reply are never re-sent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/zeroshot_run.py"
OUT = ROOT / "runs/claude/test"
STATE = ROOT / "runs/claude/batches.json"
TEXTS_DIR = ROOT / "data/raw_opf"
MODEL = "claude-sonnet-5"

# Estimated cost per window on Batches, central and bad case. Re-derived from the
# canary: ~19,900 uncached input tokens per full window (1.27 tokens per character),
# a 3,442-token cached prompt and ~4,300 output tokens.
EST_CENTRAL, EST_HIGH = 0.042, 0.053
CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def load_runner():
    spec = importlib.util.spec_from_file_location("runner", RUNNER)
    m = importlib.util.module_from_spec(spec)
    sys.argv = [str(RUNNER)]           # the runner parses argv only inside main()
    spec.loader.exec_module(m)
    return m


def load_books(m, ids: list[str]) -> dict[str, str]:
    return {b: m.load_book_text(TEXTS_DIR, b) for b in ids}


def test_books(m) -> list[str]:
    return sorted(b for b, s in m.split_of().items() if s == "test")


def windows_for(m, book: str, text: str):
    return m.make_chunks(text, book, 16000, 14000)


def pending(book: str, chunks) -> list[int]:
    return [i for i in range(len(chunks)) if not (OUT / book / f"{i:04d}.json").exists()]


def build_requests(m, books: list[str]):
    prompt = m.load_prompt(m.PROMPT)
    texts = load_books(m, books)
    reqs, per_book = [], {}
    for b in books:
        chunks = windows_for(m, b, texts[b])
        todo = pending(b, chunks)
        per_book[b] = (len(chunks), len(todo))
        for i in todo:
            cid = f"{b}__{i:04d}"
            assert CUSTOM_ID.match(cid), cid
            reqs.append({
                "custom_id": cid,
                "params": {
                    "model": MODEL,
                    "max_tokens": 64000,
                    "thinking": {"type": "adaptive"},
                    "output_config": {
                        "effort": "low",
                        "format": {"type": "json_schema", "schema": m.SPAN_SCHEMA},
                    },
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt,
                         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                        {"type": "text", "text": chunks[i]["text"]},
                    ]}],
                },
            })
    return reqs, per_book


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save_state(st: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def cmd_plan(args):
    m = load_runner()
    books = test_books(m)
    texts = load_books(m, books)
    counts = {b: len(pending(b, windows_for(m, b, texts[b]))) for b in books}
    total = sum(counts.values())
    groups = [[] for _ in range(args.groups)]
    load = [0] * args.groups
    for b in sorted(books, key=lambda x: -counts[x]):
        i = load.index(min(load))
        groups[i].append(b)
        load[i] += counts[b]
    print(f"{len(books)} test books, {total} windows still to run "
          f"(estimated ${total * EST_CENTRAL:.0f} central, ${total * EST_HIGH:.0f} bad case on Batches)\n")
    for k, (g, n) in enumerate(zip(groups, load), 1):
        print(f"g{k}: {n} windows, ~${n * EST_CENTRAL:.1f} to ${n * EST_HIGH:.1f}")
        print(f"    --group g{k} --books {' '.join(g)}")


def cmd_create(args):
    m = load_runner()
    st = load_state()
    if args.group in st:
        raise SystemExit(f"group {args.group} already submitted as {st[args.group]['id']}; "
                         "use poll/fetch (never submit the same group twice)")
    reqs, per_book = build_requests(m, args.books)
    size_mb = sum(len(json.dumps(r)) for r in reqs) / 1e6
    n = len(reqs)
    print(f"group {args.group}: {n} requests from {len(args.books)} books, {size_mb:.1f} MB")
    print(f"  estimated cost on Batches: ${n * EST_CENTRAL:.1f} (bad case ${n * EST_HIGH:.1f})")
    for b, (tot, todo) in per_book.items():
        print(f"  {b}: {todo}/{tot} windows to send")
    if not n:
        print("nothing to send")
        return
    if not args.yes:
        print("\ndry run: nothing sent. Add --yes to submit.")
        return
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=reqs)
    st[args.group] = {"id": batch.id, "books": args.books, "n_requests": n,
                      "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_state(st)                      # the id is saved before anything else can fail
    print(f"submitted {batch.id} ({n} requests). State saved to {STATE}")


def cmd_poll(args):
    import anthropic
    st = load_state()
    b = anthropic.Anthropic().messages.batches.retrieve(st[args.group]["id"])
    c = b.request_counts
    print(f"{args.group} {b.id}: {b.processing_status}  "
          f"processing={c.processing} succeeded={c.succeeded} errored={c.errored} "
          f"canceled={c.canceled} expired={c.expired}")


def cmd_fetch(args):
    import anthropic
    m = load_runner()
    st = load_state()
    info = st[args.group]
    client = anthropic.Anthropic()
    texts = load_books(m, info["books"])
    chunks = {b: windows_for(m, b, texts[b]) for b in info["books"]}
    saved = bad = 0
    tok = {"in": 0, "out": 0, "read": 0, "write": 0}
    for res in client.messages.batches.results(info["id"]):
        book, idx = res.custom_id.split("__")
        idx = int(idx)
        if res.result.type != "succeeded":
            print(f"  {res.custom_id}: {res.result.type}")
            bad += 1
            continue
        msg = res.result.message
        u = msg.usage
        rec = {
            "book": book, "window": idx,
            "start": chunks[book][idx]["start"], "end": chunks[book][idx]["end"],
            "raw": "".join(bl.text for bl in msg.content if bl.type == "text"),
            "prompt_tokens": u.input_tokens + (u.cache_read_input_tokens or 0)
                             + (u.cache_creation_input_tokens or 0),
            "output_tokens": u.output_tokens,
            "cache_write": u.cache_creation_input_tokens or 0,
            "cache_read": u.cache_read_input_tokens or 0,
            "finish": str(msg.stop_reason),
        }
        p = OUT / book / f"{idx:04d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        saved += 1
        tok["in"] += u.input_tokens
        tok["out"] += u.output_tokens
        tok["read"] += u.cache_read_input_tokens or 0
        tok["write"] += u.cache_creation_input_tokens or 0
    cost = (tok["in"] * 2 + tok["read"] * 0.2 + tok["write"] * 4 + tok["out"] * 10) / 1e6 / 2
    print(f"saved {saved} windows, {bad} not succeeded. "
          f"tokens: in {tok['in']:,} out {tok['out']:,} cache read {tok['read']:,} write {tok['write']:,}")
    print(f"actual batch cost: about ${cost:.2f} (Batches is billed at half price)")
    if bad:
        print("re-run `plan`/`create` for the missing windows; saved ones are skipped")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--groups", type=int, default=3)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("create")
    p.add_argument("--group", required=True)
    p.add_argument("--books", nargs="+", required=True)
    p.add_argument("--yes", action="store_true", help="actually submit (spends money)")
    p.set_defaults(fn=cmd_create)
    for name, fn in (("poll", cmd_poll), ("fetch", cmd_fetch)):
        p = sub.add_parser(name)
        p.add_argument("--group", required=True)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
