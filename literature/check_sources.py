#!/usr/bin/env python3
"""Validate literature/data/SOURCES.yaml and cross-reference DOIs against
inline citations in the FWMC C++ source tree.

Exit codes: 0 if all checks pass, 1 if any errors are found.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _SCRIPT_DIR / "data"
_SOURCES_YAML = _DATA_DIR / "SOURCES.yaml"
_SRC_DIR = _PROJECT_ROOT / "src"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"doi:(10\.\S+)")
_REQUIRED_SOURCE_FIELDS = {"id", "authors", "year", "title"}
_LOCATOR_FIELDS = {"doi", "pmc", "url"}


def _error(msg: str, *, errors: list[str]) -> None:
    """Record an error message."""
    errors.append(msg)
    print(f"  ERROR: {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"  WARN:  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_sources() -> dict[str, Any]:
    """Load and return the parsed SOURCES.yaml."""
    with open(_SOURCES_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Source entry validation
# ---------------------------------------------------------------------------

def validate_sources(data: dict[str, Any]) -> tuple[list[str], set[str]]:
    """Validate every source entry in *data*.

    Returns a tuple of (error list, set of all DOIs declared in the YAML).
    """
    errors: list[str] = []
    all_dois: set[str] = set()

    for dataset_key, dataset in data.items():
        print(f"[{dataset_key}]")

        # Dataset level checks.
        if not isinstance(dataset, dict):
            _error(
                f"dataset '{dataset_key}' is not a mapping",
                errors=errors,
            )
            continue

        if "description" not in dataset:
            _error(
                f"dataset '{dataset_key}' is missing a 'description' field",
                errors=errors,
            )

        sources = dataset.get("sources")
        if not sources:
            _error(
                f"dataset '{dataset_key}' has no 'sources' list",
                errors=errors,
            )
            continue

        if not isinstance(sources, list):
            _error(
                f"dataset '{dataset_key}': 'sources' should be a list",
                errors=errors,
            )
            continue

        for src in sources:
            src_id = src.get("id", "<unknown>")

            # Required fields.
            missing = _REQUIRED_SOURCE_FIELDS - set(src.keys())
            if missing:
                _error(
                    f"[{dataset_key}] source '{src_id}' missing fields: "
                    f"{', '.join(sorted(missing))}",
                    errors=errors,
                )

            # At least one locator.
            if not _LOCATOR_FIELDS & set(src.keys()):
                _error(
                    f"[{dataset_key}] source '{src_id}' has no locator "
                    f"(need at least one of: {', '.join(sorted(_LOCATOR_FIELDS))})",
                    errors=errors,
                )

            # Collect DOIs.
            doi = src.get("doi")
            if doi:
                all_dois.add(doi)

    return errors, all_dois


# ---------------------------------------------------------------------------
# Consensus CSV validation
# ---------------------------------------------------------------------------

def validate_consensus_refs(data: dict[str, Any]) -> list[str]:
    """Check that every 'consensus' CSV path exists and that source IDs
    referenced inside each CSV appear somewhere in the YAML."""
    errors: list[str] = []

    # Build a set of all known source IDs.
    known_ids: set[str] = set()
    for dataset in data.values():
        if not isinstance(dataset, dict):
            continue
        for src in dataset.get("sources", []):
            sid = src.get("id")
            if sid:
                known_ids.add(sid)

    for dataset_key, dataset in data.items():
        if not isinstance(dataset, dict):
            continue
        consensus = dataset.get("consensus")
        if not consensus:
            continue

        csv_path = _DATA_DIR / consensus
        if not csv_path.exists():
            _error(
                f"[{dataset_key}] consensus file not found: {csv_path}",
                errors=errors,
            )
            continue

        print(f"  checking consensus CSV: {consensus}")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(
                (line for line in fh if not line.startswith("#")),
            )
            if reader.fieldnames and "source" in reader.fieldnames:
                for row_num, row in enumerate(reader, start=2):
                    ref = row.get("source", "").strip()
                    if ref and ref not in known_ids:
                        _error(
                            f"[{dataset_key}] {consensus}:{row_num} "
                            f"references unknown source '{ref}'",
                            errors=errors,
                        )

    return errors


# ---------------------------------------------------------------------------
# DOI cross-referencing
# ---------------------------------------------------------------------------

def collect_source_dois() -> dict[str, list[str]]:
    """Scan .h and .cc files under src/ for inline DOI comments.

    Returns a mapping from DOI string to the list of file paths where it
    appears.
    """
    doi_map: dict[str, list[str]] = {}
    for pattern in ("**/*.h", "**/*.cc"):
        for path in _SRC_DIR.glob(pattern):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in _DOI_RE.finditer(text):
                doi = match.group(1).rstrip(".,;:)")
                doi_map.setdefault(doi, []).append(
                    str(path.relative_to(_PROJECT_ROOT))
                )
    return doi_map


def crossref_dois(
    yaml_dois: set[str],
    source_dois: dict[str, list[str]],
) -> list[str]:
    """Compare DOIs declared in the YAML against those cited in source code.

    Reports source DOIs that are not tracked in the YAML. DOIs in the YAML
    that are not cited in source are silently ignored (not every reference
    needs an inline citation).
    """
    errors: list[str] = []
    untracked = set(source_dois.keys()) - yaml_dois
    for doi in sorted(untracked):
        files = ", ".join(source_dois[doi])
        _error(
            f"DOI in source but not in SOURCES.yaml: {doi} (cited in {files})",
            errors=errors,
        )
    return errors


# ---------------------------------------------------------------------------
# CrossRef API verification
# ---------------------------------------------------------------------------

def verify_dois_via_crossref(dois: set[str]) -> list[str]:
    """Hit the CrossRef API to confirm each DOI resolves."""
    errors: list[str] = []
    print(f"\nVerifying {len(dois)} DOI(s) via CrossRef API ...")
    for doi in sorted(dois):
        url = f"https://api.crossref.org/works/{doi}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FWMC-check-sources/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    print(f"  OK   {doi}")
                else:
                    _error(
                        f"DOI returned HTTP {resp.status}: {doi}",
                        errors=errors,
                    )
        except urllib.error.HTTPError as exc:
            _error(
                f"DOI lookup failed (HTTP {exc.code}): {doi}",
                errors=errors,
            )
        except urllib.error.URLError as exc:
            _error(
                f"DOI lookup failed ({exc.reason}): {doi}",
                errors=errors,
            )
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate FWMC literature sources.",
    )
    parser.add_argument(
        "--verify-dois",
        action="store_true",
        help="Query CrossRef to verify every DOI resolves.",
    )
    args = parser.parse_args(argv)

    all_errors: list[str] = []

    # 1. Load YAML.
    print(f"Loading {_SOURCES_YAML.relative_to(_PROJECT_ROOT)} ...")
    data = load_sources()

    # 2. Validate source entries.
    print("\nValidating source entries ...")
    entry_errors, yaml_dois = validate_sources(data)
    all_errors.extend(entry_errors)

    # 3. Validate consensus CSV references.
    print("\nValidating consensus CSV references ...")
    all_errors.extend(validate_consensus_refs(data))

    # 4. Cross-reference DOIs against source code.
    print("\nCross-referencing DOIs against src/ ...")
    source_dois = collect_source_dois()
    print(f"  found {len(source_dois)} unique DOI(s) in source code")
    all_errors.extend(crossref_dois(yaml_dois, source_dois))

    # 5. Optional: verify DOIs via CrossRef.
    if args.verify_dois:
        all_errors.extend(verify_dois_via_crossref(yaml_dois))

    # Summary.
    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s) found.")
        return 1

    print("PASSED: all checks OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
