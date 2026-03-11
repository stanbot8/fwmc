#!/usr/bin/env python3
"""Verify SOURCES.yaml metadata against CrossRef records.

Reads literature/data/SOURCES.yaml and checks each entry with a DOI against
the CrossRef API. Reports mismatches in title, year, and first author last
name. Optionally patches the YAML to match CrossRef.

Usage:
    python3 literature/verify_notes.py
    python3 literature/verify_notes.py --fix-metadata --verbose
    python3 literature/verify_notes.py --cache /tmp/crossref_cache.json
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCES_PATH = Path(__file__).resolve().parent / "data" / "SOURCES.yaml"

CROSSREF_API = "https://api.crossref.org/works/"

USER_AGENT = (
    "FWMC/1.0 "
    "(https://github.com/stanbot8/fwmc; "
    "mailto:stanbot8@users.noreply.github.com)"
)

# Similarity threshold: titles scoring below this are flagged.
TITLE_SIMILARITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def strip_diacritics(text: str) -> str:
    """Remove diacritical marks, returning ASCII approximation."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace, drop punctuation."""
    text = strip_diacritics(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two title strings.

    Returns a float in [0, 1]. A value of 1.0 means every word matches.
    """
    words_a = set(normalize(a).split())
    words_b = set(normalize(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def first_author_last(authors_field: str) -> str:
    """Extract the last name of the first author from an author string.

    Handles formats such as:
        "Dorkenwald S, Matsliah A, ..."
        "Turner GC, Bazhenov M, Laurent G"
    """
    first = authors_field.split(",")[0].strip()
    # Drop trailing initials (single uppercase letters, possibly dotted).
    parts = first.split()
    # Walk backwards, removing initials.
    while len(parts) > 1 and re.match(r"^[A-Z]\.?$", parts[-1]):
        parts.pop()
    return strip_diacritics(" ".join(parts)).strip()


# ---------------------------------------------------------------------------
# CrossRef interaction
# ---------------------------------------------------------------------------
_crossref_cache: dict = {}


def fetch_crossref(doi: str, *, cache_path: str | None = None,
                   verbose: bool = False) -> dict | None:
    """Fetch metadata for a DOI from CrossRef, with optional JSON cache.

    Returns the parsed 'message' dict on success, or None on failure.
    """
    doi = doi.strip()

    # Check in-memory cache first.
    if doi in _crossref_cache:
        return _crossref_cache[doi]

    # Check on-disk cache.
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                disk_cache = json.load(fh)
            if doi in disk_cache:
                _crossref_cache[doi] = disk_cache[doi]
                return disk_cache[doi]
        except (json.JSONDecodeError, OSError):
            pass

    url = CROSSREF_API + urllib.request.quote(doi, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    if verbose:
        print(f"  [crossref] GET {url}")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = data.get("message", {})
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        if verbose:
            print(f"  [crossref] failed for {doi}: {exc}")
        return None

    _crossref_cache[doi] = message

    # Persist to disk cache.
    if cache_path:
        _save_disk_cache(cache_path)

    # Polite rate limiting: CrossRef asks for at most 50 req/s.
    time.sleep(0.1)
    return message


def _save_disk_cache(cache_path: str) -> None:
    """Write the in-memory cache to a JSON file."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    try:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(_crossref_cache, fh, indent=1)
    except OSError:
        pass


def extract_crossref_fields(message: dict) -> dict:
    """Pull title, year, and first author last name from a CrossRef message.

    Returns a dict with keys: title, year, first_author_last.
    Missing fields are set to None.
    """
    # Title: CrossRef returns a list of title strings.
    titles = message.get("title", [])
    cr_title = titles[0] if titles else None

    # Year: prefer published-print, then published-online, then issued.
    cr_year = None
    for date_key in ("published-print", "published-online", "issued"):
        parts = (message.get(date_key) or {}).get("date-parts", [[]])
        if parts and parts[0] and parts[0][0]:
            cr_year = int(parts[0][0])
            break

    # First author last name.
    authors = message.get("author", [])
    cr_author = None
    if authors:
        cr_author = authors[0].get("family")
        if cr_author:
            cr_author = strip_diacritics(cr_author)

    return {
        "title": cr_title,
        "year": cr_year,
        "first_author_last": cr_author,
    }


# ---------------------------------------------------------------------------
# Entry checking
# ---------------------------------------------------------------------------
class Mismatch:
    """A single metadata mismatch between local YAML and CrossRef."""

    def __init__(self, entry_id: str, field: str,
                 local_value, crossref_value, note: str = ""):
        self.entry_id = entry_id
        self.field = field
        self.local_value = local_value
        self.crossref_value = crossref_value
        self.note = note

    def __str__(self) -> str:
        parts = [
            f"[{self.entry_id}] {self.field}: "
            f"local={self.local_value!r}, crossref={self.crossref_value!r}"
        ]
        if self.note:
            parts.append(f"  ({self.note})")
        return "".join(parts)


def check_entry(entry: dict, *, cache_path: str | None = None,
                verbose: bool = False) -> list[Mismatch]:
    """Compare a single SOURCES.yaml entry against CrossRef.

    Returns a (possibly empty) list of Mismatch objects.
    """
    doi = entry.get("doi")
    entry_id = entry.get("id", "<unknown>")
    if not doi:
        return []

    message = fetch_crossref(doi, cache_path=cache_path, verbose=verbose)
    if message is None:
        if verbose:
            print(f"  [skip] no CrossRef response for {entry_id} ({doi})")
        return []

    cr = extract_crossref_fields(message)
    mismatches: list[Mismatch] = []

    # Title comparison.
    local_title = entry.get("title", "")
    if cr["title"] and local_title:
        sim = title_similarity(local_title, cr["title"])
        if sim < TITLE_SIMILARITY_THRESHOLD:
            mismatches.append(Mismatch(
                entry_id, "title", local_title, cr["title"],
                note=f"similarity={sim:.3f}",
            ))

    # Year comparison.
    local_year = entry.get("year")
    if cr["year"] is not None and local_year is not None:
        if int(local_year) != cr["year"]:
            mismatches.append(Mismatch(
                entry_id, "year", local_year, cr["year"],
            ))

    # First author last name.
    local_authors = entry.get("authors", "")
    local_last = first_author_last(local_authors)
    if cr["first_author_last"] and local_last:
        if normalize(local_last) != normalize(cr["first_author_last"]):
            mismatches.append(Mismatch(
                entry_id, "first_author_last", local_last,
                cr["first_author_last"],
            ))

    return mismatches


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
def load_all_entries(sources_path: str | Path | None = None) -> list[dict]:
    """Read the single SOURCES.yaml and return a flat list of source entries.

    Each entry is a dict with at least id, authors, year, title, and doi.
    """
    path = Path(sources_path) if sources_path else SOURCES_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return []

    entries: list[dict] = []
    for _dataset_key, dataset in data.items():
        if not isinstance(dataset, dict):
            continue
        for src in dataset.get("sources", []):
            if isinstance(src, dict):
                entries.append(src)
    return entries


# ---------------------------------------------------------------------------
# Metadata fixing
# ---------------------------------------------------------------------------
def apply_fixes(sources_path: Path, mismatches: list[Mismatch],
                verbose: bool = False) -> int:
    """Rewrite SOURCES.yaml, replacing mismatched fields with CrossRef values.

    Operates on the raw YAML text to preserve comments and formatting as
    much as possible. Returns the number of fields patched.
    """
    with open(sources_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    patched = 0
    for mm in mismatches:
        if mm.field == "title" and mm.crossref_value:
            old = f'title: "{mm.local_value}"'
            new = f'title: "{mm.crossref_value}"'
            if old in text:
                text = text.replace(old, new, 1)
                patched += 1
                if verbose:
                    print(f"  [fix] {mm.entry_id} title patched")

        elif mm.field == "year" and mm.crossref_value is not None:
            # Replace the year value on the same line as the entry id block.
            old_pat = re.compile(
                rf"(id:\s*{re.escape(mm.entry_id)}.*?year:\s*){re.escape(str(mm.local_value))}",
                re.DOTALL,
            )
            new_text, n = old_pat.subn(rf"\g<1>{mm.crossref_value}", text, count=1)
            if n:
                text = new_text
                patched += 1
                if verbose:
                    print(f"  [fix] {mm.entry_id} year patched")

    if patched:
        with open(sources_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    return patched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify SOURCES.yaml metadata against CrossRef records.",
    )
    parser.add_argument(
        "--fix-metadata", action="store_true",
        help="Overwrite SOURCES.yaml fields that disagree with CrossRef.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress and CrossRef request URLs.",
    )
    parser.add_argument(
        "--cache", default=None, metavar="PATH",
        help="Path to a JSON file for caching CrossRef responses.",
    )
    args = parser.parse_args()

    # Locate sources file.
    if not SOURCES_PATH.is_file():
        print(f"Error: {SOURCES_PATH} not found", file=sys.stderr)
        sys.exit(1)

    entries = load_all_entries()
    if not entries:
        print("No source entries found in SOURCES.yaml.")
        sys.exit(0)

    print(f"Loaded {len(entries)} source entries from {SOURCES_PATH}")

    all_mismatches: list[Mismatch] = []

    for entry in entries:
        doi = entry.get("doi")
        entry_id = entry.get("id", "<unknown>")
        if not doi:
            if args.verbose:
                print(f"  [skip] {entry_id}: no DOI")
            continue

        if args.verbose:
            print(f"  checking {entry_id} ...")

        mismatches = check_entry(
            entry, cache_path=args.cache, verbose=args.verbose,
        )
        all_mismatches.extend(mismatches)

    # Report results.
    if all_mismatches:
        print(f"\nFound {len(all_mismatches)} mismatch(es):\n")
        for mm in all_mismatches:
            print(f"  {mm}")

        if args.fix_metadata:
            print()
            n_fixed = apply_fixes(
                SOURCES_PATH, all_mismatches, verbose=args.verbose,
            )
            print(f"Patched {n_fixed} field(s) in {SOURCES_PATH}")
    else:
        print("\nAll entries match CrossRef records.")

    # Exit code: 1 if mismatches remain unfixed.
    if all_mismatches and not args.fix_metadata:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
