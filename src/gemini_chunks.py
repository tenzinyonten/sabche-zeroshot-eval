"""Copied verbatim from the quotation benchmark repo
(Tibetan-quotation-detection, src/layer_detection/gemini_chunks.py) on 2026-09-22.
Kept unchanged so tsawa and quotation share one locator.

Tibetan text chunking and gold span validation for zero-shot Gemini.

Handles character-level snapping at shads, building half-open chunks,
and verifying inclusion invariants for gold spans.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Sequence, Mapping, Iterable

SHAD = "།"


def snap(text: str, target: int, radius: int = 100) -> int:
    """Move a cut to just after the nearest shad."""
    lo = max(0, target - radius)
    hi = min(len(text), target + radius)
    best = None
    for i in range(lo, hi):
        if text[i] == SHAD:
            if best is None or abs(i + 1 - target) < abs(best - target):
                best = i + 1  # cut AFTER the shad, keep it with the current chunk
    return best if best is not None else target


def make_chunks(
    text: str,
    book_id: str,
    W: int,
    S: int,
) -> list[dict[str, Any]]:
    """Build overlapping chunks using shad-based snapping.

    W: window size in characters
    S: stride/step in characters
    """
    chunks: list[dict[str, Any]] = []
    pos = 0
    i = 0
    while pos < len(text):
        if pos + W >= len(text):
            end = len(text)
        else:
            end = snap(text, pos + W)
            
        # Ensure we always make progress and cover everything
        if end <= pos:
            end = min(pos + W, len(text))
            
        chunks.append({
            "chunk_id": f"{book_id}_{i:05d}",
            "book_id": book_id,
            "start": pos,  # book offset, inclusive
            "end": end,    # book offset, exclusive
            "text": text[pos:end],
        })
        if end >= len(text):
            break
        nxt = snap(text, pos + S)
        pos = nxt if nxt > pos else pos + S  # never stall
        i += 1
    return chunks


def compute_span_lengths(
    spans: Iterable[Mapping[str, Any]],
    layers: Sequence[str],
) -> dict[str, list[int]]:
    """Extract inclusive character lengths of spans grouped by layer name.

    spans are dictionaries with 'start', 'end' (inclusive), and 'label' (uppercase).
    layers are the target layers to look at (e.g., ['QUOTATION', 'TSAWA', 'YIGCHUNG']).
    """
    lengths_by_layer: dict[str, list[int]] = {layer.upper(): [] for layer in layers}
    for span in spans:
        label = str(span["label"]).upper()
        if label in lengths_by_layer:
            # Inclusive character length = end - start + 1
            length = int(span["end"]) - int(span["start"]) + 1
            lengths_by_layer[label].append(length)
    return lengths_by_layer


def compute_percentiles(lengths: Sequence[int]) -> dict[str, int | float]:
    """Calculate p50, p95, p99, and max of a sequence of lengths."""
    if not lengths:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0}
    arr = np.asarray(lengths)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
    }


def verify_chunks_identity(text: str, chunks: list[dict[str, Any]]) -> bool:
    """Check that book[c['start']:c['end']] == c['text'] for every chunk."""
    for c in chunks:
        actual = text[c["start"] : c["end"]]
        if actual != c["text"]:
            return False
    return True


def verify_chunks_coverage(text_len: int, chunks: list[dict[str, Any]]) -> bool:
    """Check that chunks cover every character of the book.

    We union the half-open ranges [start, end) of all chunks.
    They must form a single contiguous interval from 0 to text_len.
    """
    if not chunks:
        return text_len == 0

    # Sort chunks by start offset
    sorted_chunks = sorted(chunks, key=lambda x: x["start"])

    # Ensure we start at 0
    if sorted_chunks[0]["start"] != 0:
        return False

    current_max = 0
    for c in sorted_chunks:
        # If there's a gap (current_max < start), then coverage is broken
        if c["start"] > current_max:
            return False
        current_max = max(current_max, c["end"])

    return current_max >= text_len


def verify_span_coverage_guarantee(
    spans: Iterable[Mapping[str, Any]],
    chunks: list[dict[str, Any]],
    W: int,
    S: int,
    layers: Sequence[str],
) -> tuple[bool, list[dict[str, Any]]]:
    """Verify that every gold span shorter than or equal to W - S appears whole in at least one chunk.

    Returns (passed, list_of_uncovered_spans).
    """
    target_layers = {layer.upper() for layer in layers}
    short_spans = []
    for g in spans:
        if str(g["label"]).upper() not in target_layers:
            continue
        g_len = int(g["end"]) - int(g["start"]) + 1
        if g_len <= W - S:
            short_spans.append(g)

    uncovered = []
    for g in short_spans:
        g_start = int(g["start"])
        g_end_incl = int(g["end"])
        # Span g is whole in chunk c iff c['start'] <= g_start and g_end_incl + 1 <= c['end']
        covered = False
        for c in chunks:
            if c["start"] <= g_start and (g_end_incl + 1) <= c["end"]:
                covered = True
                break
        if not covered:
            uncovered.append(g)

    return len(uncovered) == 0, uncovered


def verify_round_trip(
    spans: Iterable[Mapping[str, Any]],
    chunks: list[dict[str, Any]],
    text: str,
    layers: Sequence[str],
) -> bool:
    """Verify that we can map coordinates from a chunk's local find index back to the gold start.

    For every gold span that appears whole in some chunk, we check that slicing the chunk's text
    at the relative offset (gold_start - chunk_start) perfectly matches the gold span text.
    """
    target_layers = {layer.upper() for layer in layers}
    for g in spans:
        if str(g["label"]).upper() not in target_layers:
            continue
        g_start = int(g["start"])
        g_end_incl = int(g["end"])
        span_text = text[g_start : g_end_incl + 1]

        # Find a chunk that covers it whole
        enclosing_chunk = None
        for c in chunks:
            if c["start"] <= g_start and (g_end_incl + 1) <= c["end"]:
                enclosing_chunk = c
                break

        if enclosing_chunk is None:
            # Not covered whole by any chunk (must be longer than W - S), so skip
            continue

        chunk_text = enclosing_chunk["text"]
        chunk_start = enclosing_chunk["start"]
        expected_rel_offset = g_start - chunk_start

        # Check relative slicing matches span text exactly
        reconstructed_text = chunk_text[expected_rel_offset : expected_rel_offset + len(span_text)]
        if reconstructed_text != span_text:
            return False

    return True
