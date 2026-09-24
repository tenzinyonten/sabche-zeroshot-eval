#!/usr/bin/env python3
"""Clone OpenPecha .opf GitHub repos listed in an ID file.

OpenPecha / Pecha Format (.opf) background
------------------------------------------
Each pecha (text) is published as its own GitHub repository, typically under
the ``OpenPecha-Data`` org, named after a pecha ID such as ``I88CF073C``.
The working tree of that repo is (or contains) an ``.opf`` tree:

    <PECHA_ID>.opf/
        base/v001.txt          # full plain text (character offsets)
        layers/v001/*.yml      # annotation layers (Tsawa, Yigchung, …)

This script only *fetches*. It does not parse layers or assume a repo is
cleared for model training — data-rights questions are still open.

Clone target
------------
Repos are cloned into ``<raw-dir>/<PECHA_ID>.opf/``. That directory is the
*git* root. Many OpenPecha-Data remotes nest the actual Pecha Format tree
one level down as ``<PECHA_ID>.opf/<PECHA_ID>.opf/{base,layers}/``. Phase 2
must resolve that; this script does not flatten or rewrite the checkout.

A CSV manifest records which ID list (old vs new batch) each clone came
from and whether it succeeded.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = ["pecha_id", "source_batch", "clone_status", "org", "dest"]

# Tried in order when --org is not given. OpenPecha-Data is the primary host;
# Webuddhist-tech is a documented fallback for some older / relocated IDs.
DEFAULT_ORGS = ("OpenPecha-Data", "Webuddhist-tech")


def parse_id_file(path: Path) -> list[str]:
    """Return unique pecha IDs from a one-ID-per-line text file.

    Tolerates blank lines, surrounding whitespace, and ``#`` comments
    (full-line or trailing).
    """
    ids: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # IDs are usually a single token; ignore extra columns if someone
        # pastes a TSV by accident. Strip markdown fences (```) that often
        # hitch a ride when lists are copied out of a chat or gist.
        pecha_id = line.split()[0].strip("`")
        if not pecha_id or pecha_id == "```":
            continue
        if pecha_id not in seen:
            seen.add(pecha_id)
            ids.append(pecha_id)
    return ids


def infer_source_batch(ids_path: Path, override: str | None) -> str:
    """Tag the run as ``new`` / ``old`` from the filename, or use --batch."""
    if override:
        return override
    stem = ids_path.stem.lower()
    if "new" in stem:
        return "new"
    if "old" in stem:
        return "old"
    return stem


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MANIFEST_COLUMNS})


def upsert_manifest_row(rows: list[dict[str, str]], new_row: dict[str, str]) -> None:
    """Replace an existing (pecha_id, source_batch) row, or append."""
    key = (new_row["pecha_id"], new_row["source_batch"])
    for i, row in enumerate(rows):
        if (row.get("pecha_id"), row.get("source_batch")) == key:
            rows[i] = new_row
            return
    rows.append(new_row)


def which_clone_tool(prefer_gh: bool) -> str:
    """Return ``gh`` if usable, otherwise ``git``. Require at least git."""
    git = shutil.which("git")
    if not git:
        raise SystemExit("git is required on PATH to clone .opf repos.")
    if prefer_gh and shutil.which("gh"):
        # ``gh`` still shells out to git; prefer it when authenticated so
        # private-but-accessible repos work without embedding tokens in URLs.
        return "gh"
    return "git"


def run_clone(
    tool: str,
    org: str,
    pecha_id: str,
    dest: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if tool == "gh":
        cmd = ["gh", "repo", "clone", f"{org}/{pecha_id}", str(dest)]
    else:
        url = f"https://github.com/{org}/{pecha_id}.git"
        cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def classify_failure(returncode: int, stderr: str, stdout: str) -> str:
    """Map clone failure text to a stable manifest status (do not crash)."""
    blob = f"{stderr}\n{stdout}".lower()
    if returncode == 124 or "timed out" in blob:
        return "timeout"
    # gh + git both surface missing / private remotes this way
    private_or_missing = (
        "repository not found",
        "not found",
        "could not find",
        "404",
        "authentication failed",
        "permission denied",
        "access denied",
        "could not read from remote",
        "remote error: not found",
    )
    if any(tok in blob for tok in private_or_missing):
        return "inaccessible"
    return "error"


def dest_already_ok(dest: Path) -> bool:
    """True if a previous clone left a usable working tree."""
    if not dest.is_dir():
        return False
    # Git checkout, .opf at repo root, or the common nested layout
    # <dest>/<pecha_id>.opf/base/.
    if (dest / ".git").exists() or (dest / "base").is_dir():
        return True
    return any(p.is_dir() and (p / "base").is_dir() for p in dest.glob("*.opf"))


def fetch_one(
    pecha_id: str,
    orgs: tuple[str, ...],
    dest: Path,
    tool: str,
    timeout: int,
    skip_existing: bool,
) -> tuple[str, str]:
    """Clone pecha_id, trying orgs in order.

    Returns ``(status, org_used)``. Status values:
    already_present, cloned, inaccessible, timeout, error.
    """
    if skip_existing and dest_already_ok(dest):
        return "already_present", ""

    if dest.exists() and not dest_already_ok(dest):
        # Partial / failed previous attempt — remove so git clone can proceed.
        shutil.rmtree(dest)

    last_status = "inaccessible"
    last_org = orgs[0]
    for org in orgs:
        last_org = org
        try:
            proc = run_clone(tool, org, pecha_id, dest, timeout)
        except subprocess.TimeoutExpired:
            last_status = "timeout"
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            continue
        if proc.returncode == 0 and dest_already_ok(dest):
            return "cloned", org
        last_status = classify_failure(proc.returncode, proc.stderr, proc.stdout)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        # Only fall through to the next org on a missing/private remote.
        if last_status != "inaccessible":
            print(
                f"  ! {pecha_id} @ {org}: {last_status}\n"
                f"    {(proc.stderr or proc.stdout).strip()[:400]}",
                file=sys.stderr,
            )
            return last_status, org
        print(f"  · {pecha_id} not reachable at {org}/{pecha_id}, trying next org…")
    return last_status, last_org


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone OpenPecha .opf repos listed in a pecha-ID text file."
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        required=True,
        help="Text file with one pecha ID per line (e.g. data/ids/newdata.txt).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw_opf"),
        help="Directory to clone into (default: data/raw_opf).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw_opf/_manifest.csv"),
        help="CSV path for pecha_id, source_batch, clone_status (updated in place).",
    )
    parser.add_argument(
        "--batch",
        default=None,
        help="Source-batch tag written to the manifest. "
        "Default: inferred from the IDs filename (newdata → new, olddata → old).",
    )
    parser.add_argument(
        "--org",
        action="append",
        dest="orgs",
        help="GitHub org to try (repeatable). "
        f"Default: {' then '.join(DEFAULT_ORGS)}.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-repo clone timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Clone at most N IDs (useful for a smoke test).",
    )
    parser.add_argument(
        "--no-gh",
        action="store_true",
        help="Force plain git clone over HTTPS even if `gh` is installed.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-clone even when dest already exists (destructive for that folder).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse IDs and print planned clones without contacting GitHub.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ids_file = args.ids_file.expanduser().resolve()
    if not ids_file.is_file():
        raise SystemExit(f"IDs file not found: {ids_file}")

    pecha_ids = parse_id_file(ids_file)
    if args.limit is not None:
        pecha_ids = pecha_ids[: args.limit]
    if not pecha_ids:
        raise SystemExit(f"No pecha IDs found in {ids_file}")

    source_batch = infer_source_batch(ids_file, args.batch)
    orgs = tuple(args.orgs) if args.orgs else DEFAULT_ORGS
    raw_dir = args.raw_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    print(f"IDs file     : {ids_file} ({len(pecha_ids)} ids)")
    print(f"Source batch : {source_batch}")
    print(f"Orgs         : {', '.join(orgs)}")
    print(f"Raw dir      : {raw_dir}")
    print(f"Manifest     : {manifest_path}")

    if args.dry_run:
        for pecha_id in pecha_ids:
            print(f"  would clone {pecha_id} → {raw_dir / f'{pecha_id}.opf'}")
        return 0

    tool = which_clone_tool(prefer_gh=not args.no_gh)
    print(f"Clone tool   : {tool}")

    rows = load_manifest(manifest_path)
    counts: dict[str, int] = {}
    for i, pecha_id in enumerate(pecha_ids, start=1):
        dest = raw_dir / f"{pecha_id}.opf"
        print(f"[{i}/{len(pecha_ids)}] {pecha_id}")
        status, org_used = fetch_one(
            pecha_id=pecha_id,
            orgs=orgs,
            dest=dest,
            tool=tool,
            timeout=args.timeout,
            skip_existing=not args.no_skip_existing,
        )
        counts[status] = counts.get(status, 0) + 1
        print(f"  → {status}" + (f" ({org_used})" if org_used else ""))
        upsert_manifest_row(
            rows,
            {
                "pecha_id": pecha_id,
                "source_batch": source_batch,
                "clone_status": status,
                "org": org_used,
                "dest": str(dest),
            },
        )
        # Persist after each repo so a killed run still has a usable log.
        write_manifest(manifest_path, rows)

    print("\nDone.")
    for status, n in sorted(counts.items()):
        print(f"  {status:18s} {n}")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
