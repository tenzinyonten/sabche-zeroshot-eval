"""Copied verbatim from the quotation benchmark repo
(Tibetan-quotation-detection, src/layer_detection/gemini_locate.py) on 2026-09-22.
Kept unchanged so tsawa and quotation share one locator.

Recover book offsets from Gemini span *text* (never from model offsets).

Phase B3 of the zero-shot pipeline: ``str.find`` first, then a cheap fuzzy
fallback. Gold spans in this repo are inclusive on both ends.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

ELLIPSIS_RE = re.compile(r"(?:\.{3}|…|⋯)")
QUOTE_BLOCK_RE = re.compile(
    r"\[Quote\s+(\d+)\]\s*(.*?)(?=\[Quote\s+\d+\]|\Z)",
    re.DOTALL | re.IGNORECASE,
)
FIELD_RE = re.compile(
    r"(Frame|Starts at|Ends at|Text|Closer)\s*:\s*",
    re.IGNORECASE,
)
SKIP_WS = frozenset(" \t\n\r\u00a0\u3000")
PUNCT_MAP = str.maketrans({"༑": "།"})
TSHEG = "་"
SHAD = frozenset("།༎༏༐༑")
PROBE_LEN = 24
FUZZY_MAX_NEEDLE = 2000
DEFAULT_MIN_SCORE = 0.85
LABEL_ALIASES = {"QUOTE": "QUOTATION", "QUOTATION": "QUOTATION"}


@dataclass(frozen=True)
class LocatedSpan:
    start: int
    end: int
    score: float
    method: str
    n_hits: int = 1

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ParsedQuote:
    quote_id: str
    label: str
    text: str
    frame: str = ""
    start_text: str = ""
    end_text: str = ""
    closer: str = ""


def locate_span(
    haystack: str,
    needle: str,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    from_index: int = 0,
) -> LocatedSpan | None:
    """Find ``needle`` in ``haystack``. Offsets are inclusive.

    Order: exact substring, whitespace-normalized exact, ellipsis anchors,
    then probe-window fuzzy (needles longer than ``FUZZY_MAX_NEEDLE`` skip fuzzy).
    """
    needle = (needle or "").strip()
    if not needle or from_index >= len(haystack):
        return None

    search = haystack[from_index:]
    exact = _find_all(search, needle)
    if exact:
        start = from_index + exact[0]
        return LocatedSpan(start, start + len(needle) - 1, 1.0, "exact", len(exact))

    fragments = _ellipsis_parts(needle)
    if len(fragments) >= 2:
        anchored = locate_from_anchors(
            haystack, fragments[0], fragments[-1], min_score=min_score, from_index=from_index
        )
        if anchored is not None:
            return anchored

    normalized = _normalized_find(haystack, needle, from_index)
    if normalized is not None:
        return normalized

    if len(needle) > FUZZY_MAX_NEEDLE:
        return None
    return _fuzzy_find(haystack, needle, min_score, from_index)


def locate_from_anchors(
    haystack: str,
    start_text: str,
    end_text: str,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    from_index: int = 0,
    max_span: int | None = None,
) -> LocatedSpan | None:
    """Span from the first start-fragment hit to the first end-fragment after it.

    ``max_span`` caps how far past the start anchor the end anchor may sit. A
    tail the model mistyped is otherwise found thousands of characters later,
    swallowing everything in between; capping turns that into a clean miss.
    """
    start_text = _strip_ellipsis((start_text or "").strip())
    end_text = _strip_ellipsis((end_text or "").strip())
    if not start_text or not end_text:
        return None

    start_hit = locate_span(haystack, start_text, min_score=min_score, from_index=from_index)
    if start_hit is None:
        return None
    limit = len(haystack)
    if max_span is not None and max_span > 0:
        limit = min(limit, start_hit.start + max_span)
    end_hit = locate_span(
        haystack[:limit], end_text, min_score=min_score, from_index=start_hit.start
    )
    if end_hit is None or end_hit.end < start_hit.start:
        return None
    clean = {"exact", "normalized"}
    method = "anchors" if start_hit.method in clean and end_hit.method in clean else "anchors_fuzzy"
    score = min(start_hit.score, end_hit.score)
    return LocatedSpan(start_hit.start, end_hit.end, score, method, start_hit.n_hits)


def recover_offsets(
    chunk_text: str,
    span_text: str,
    chunk_start: int,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
) -> tuple[int, int] | None:
    """Map a model-emitted string onto inclusive book offsets."""
    hit = locate_span(chunk_text, span_text, min_score=min_score)
    if hit is None:
        return None
    return chunk_start + hit.start, chunk_start + hit.end


def parse_quote_dump(raw: str) -> list[ParsedQuote]:
    """Parse the human Gemini dump or a JSON ``{\"spans\":[...]}`` payload.

    JSON items may carry the full ``text`` or the anchor pair ``head``/``tail``
    from ``docs/prompts/gemini_quote_only_anchors.md``. An empty ``tail`` there
    means the span is 40 characters or shorter and ``head`` holds all of it.
    """
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("[{"):
        return _parse_json_spans(stripped)

    quotes: list[ParsedQuote] = []
    for match in QUOTE_BLOCK_RE.finditer(raw):
        quote_id = match.group(1)
        fields = _parse_fields(match.group(2))
        text = fields.get("text", "").strip()
        start_text = fields.get("starts at", "").strip()
        end_text = fields.get("ends at", "").strip()
        quotes.append(
            ParsedQuote(
                quote_id=quote_id,
                label="QUOTATION",
                text=text,
                frame=fields.get("frame", "").strip(),
                start_text=start_text,
                end_text=end_text,
                closer=fields.get("closer", "").strip(),
            )
        )
    return quotes


def locate_parsed_quote(
    haystack: str,
    quote: ParsedQuote,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    from_index: int = 0,
    max_span: int | None = None,
) -> LocatedSpan | None:
    """Prefer full ``text`` when complete; otherwise start/end anchors.

    ``from_index`` lets a caller walk a book in emission order, which is what
    keeps a 20-character head anchor from matching an earlier lookalike.
    ``max_span`` caps the head-to-tail distance for anchor input.
    """
    text = quote.text.strip()
    if text and not ELLIPSIS_RE.search(text):
        hit = locate_span(haystack, text, min_score=min_score, from_index=from_index)
        if hit is not None:
            return hit
    if quote.start_text and quote.end_text:
        return locate_from_anchors(
            haystack,
            quote.start_text,
            quote.end_text,
            min_score=min_score,
            from_index=from_index,
            max_span=max_span,
        )
    if text:
        return locate_span(haystack, text, min_score=min_score, from_index=from_index)
    if quote.start_text:
        return locate_span(
            haystack, quote.start_text, min_score=min_score, from_index=from_index
        )
    return None


def canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(str(label).upper(), str(label).upper())


def _parse_json_spans(raw: str) -> list[ParsedQuote]:
    payload = json.loads(raw)
    items = payload["spans"] if isinstance(payload, dict) else payload
    quotes: list[ParsedQuote] = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            quotes.append(ParsedQuote(quote_id=str(i), label="QUOTATION", text=item))
            continue
        head = item.get("head", "")
        tail = item.get("tail", "")
        text = item.get("text", "")
        if not text and head and not tail.strip():
            # anchor contract: a span of 40 chars or less comes back whole in `head`
            text = head
        quotes.append(
            ParsedQuote(
                quote_id=str(item.get("id", i)),
                label=canonical_label(item.get("label", "QUOTATION")),
                text=text,
                frame=item.get("frame", ""),
                start_text=item.get("start_text", item.get("starts_at", head)),
                end_text=item.get("end_text", item.get("ends_at", tail)),
                closer=item.get("closer", ""),
            )
        )
    return quotes


def _parse_fields(block: str) -> dict[str, str]:
    parts = FIELD_RE.split(block)
    fields: dict[str, str] = {}
    # split keeps delimiters: [preamble, name, value, name, value, ...]
    i = 1
    while i + 1 < len(parts):
        fields[parts[i].strip().lower()] = parts[i + 1]
        i += 2
    return fields


def _strip_ellipsis(text: str) -> str:
    return ELLIPSIS_RE.sub("", text).strip()


def _ellipsis_parts(text: str) -> list[str]:
    parts = [p.strip() for p in ELLIPSIS_RE.split(text) if p.strip()]
    return parts


def _find_all(haystack: str, needle: str) -> list[int]:
    hits: list[int] = []
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos < 0:
            return hits
        hits.append(pos)
        start = pos + max(1, len(needle))


def _next_visible(text: str, i: int) -> str:
    while i < len(text) and text[i] in SKIP_WS:
        i += 1
    return text[i] if i < len(text) else ""


def _normalize(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch in SKIP_WS:
            continue
        if ch == TSHEG and _next_visible(text, i + 1) in SHAD:
            # orthography drops the tsheg before a shad; transcriptions disagree
            continue
        chars.append(ch.translate(PUNCT_MAP))
        index_map.append(i)
    return "".join(chars), index_map


def _normalized_find(haystack: str, needle: str, from_index: int) -> LocatedSpan | None:
    hay_norm, hay_map = _normalize(haystack[from_index:])
    needle_norm, _ = _normalize(needle)
    if not needle_norm or not hay_norm:
        return None
    hits = _find_all(hay_norm, needle_norm)
    if not hits:
        return None
    local = hits[0]
    orig_start = from_index + hay_map[local]
    orig_end = from_index + hay_map[local + len(needle_norm) - 1]
    return LocatedSpan(orig_start, orig_end, 1.0, "normalized", len(hits))


def _fuzzy_find(
    haystack: str,
    needle: str,
    min_score: float,
    from_index: int,
) -> LocatedSpan | None:
    search = haystack[from_index:]
    needle_norm, _ = _normalize(needle)
    search_norm, search_map = _normalize(search)
    if len(needle_norm) < 8 or not search_norm:
        return None

    probe = needle_norm[: min(PROBE_LEN, len(needle_norm))]
    candidates: set[int] = set()
    start = 0
    while True:
        pos = search_norm.find(probe, start)
        if pos < 0:
            break
        candidates.add(pos)
        start = pos + 1
        if len(candidates) > 40:
            break

    if not candidates:
        # shorter probe if the opening chars drifted
        probe = needle_norm[len(needle_norm) // 4 : len(needle_norm) // 4 + min(16, len(needle_norm) // 2)]
        if len(probe) >= 8:
            start = 0
            while True:
                pos = search_norm.find(probe, start)
                if pos < 0:
                    break
                # estimated needle start
                est = max(0, pos - len(needle_norm) // 4)
                candidates.add(est)
                start = pos + 1
                if len(candidates) > 40:
                    break

    best: LocatedSpan | None = None
    window = len(needle_norm)
    matcher = SequenceMatcher()
    matcher.set_seq2(needle_norm)
    for pos in candidates:
        lo = max(0, pos - 8)
        hi = min(len(search_norm), pos + window + 8)
        chunk = search_norm[lo:hi]
        matcher.set_seq1(chunk)
        blocks = matcher.get_matching_blocks()
        # Best contiguous-ish cover of the needle inside chunk.
        score = matcher.ratio()
        if score < min_score:
            continue
        matching = [b for b in blocks if b.size]
        if not matching:
            continue
        span_lo = min(b.a for b in matching)
        span_hi = max(b.a + b.size for b in matching)
        orig_lo = from_index + search_map[lo + span_lo]
        orig_hi = from_index + search_map[min(len(search_map) - 1, lo + span_hi - 1)]
        hit = LocatedSpan(orig_lo, orig_hi, score, "fuzzy", 1)
        if best is None or hit.score > best.score:
            best = hit
    return best


def gold_tuples(spans: Sequence[dict], label: str = "QUOTATION") -> list[tuple[str, int, int]]:
    wanted = canonical_label(label)
    out = []
    for span in spans:
        if canonical_label(span["label"]) != wanted:
            continue
        out.append((wanted, int(span["start"]), int(span["end"])))
    return out
