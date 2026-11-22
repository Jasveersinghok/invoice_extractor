#!/usr/bin/env python3
"""
Invoice extractor CLI.

Usage
-----
    python extract.py <file>               # auto-detect input mode
    python extract.py --mode text <file>   # force text mode
    python extract.py --mode vision <file> # force vision mode

Exit codes
----------
    0 : Success.  JSON is printed to stdout.
        Includes valid non-invoices and documents where validation failed
        but a result was parsed (needs_review is set in metadata).
    1 : Hard failure.  Nothing is printed to stdout.
        One error-JSON line is printed to stderr:
            {"error": "<type>", "message": "<text>"}
        Metadata is written with processing_status = "failed".

Outputs
-------
    stdout              : extraction JSON (on success)
    stderr              : error JSON (on hard failure only)
    outputs/metadata/   : per-document operational metadata (always written)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loader import LoadError, load_document
from llm import MODEL, PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK, call_llm
from prompts import PROMPT_VERSION, build_correction_messages, build_messages
from validate import validate_extraction
from review import add_heuristic_flags

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_DIR = Path("outputs") / "metadata"
PRICE_DATE: str = os.environ.get("PRICE_DATE", "2026-11-22")


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _blank_meta() -> dict:
    """Return a metadata dict initialised to failure defaults."""
    return {
        "processing_status": "failed",
        "needs_review": False,
        "review_reasons": [],
        "validation_errors": [],
        "input_mode": None,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "input_tokens": 0,
        "output_tokens": 0,
        "api_calls": 0,
        "api_retry_count": 0,
        "correction_attempted": False,
        "correction_outcome": "n/a",
        "cost_usd": 0.0,
        "seconds": 0.0,
        "price_in_per_mtok": PRICE_IN_PER_MTOK,
        "price_out_per_mtok": PRICE_OUT_PER_MTOK,
        "price_date": PRICE_DATE,
    }


def _write_metadata(stem: str, meta: dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    path = METADATA_DIR / f"{stem}.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _hard_fail(error_type: str, message: str, stem: str, meta: dict) -> None:
    """
    Record a hard failure, print error JSON to stderr, exit 1.
    Nothing is printed to stdout so the caller cannot mistake failure for a
    real non-invoice result.
    """
    meta["processing_status"] = "failed"
    meta["needs_review"] = True
    if "processing_failed" not in meta["review_reasons"]:
        meta["review_reasons"].append("processing_failed")
    _write_metadata(stem, meta)
    print(json.dumps({"error": error_type, "message": message}), file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------

def _add_usage(meta: dict, usage: dict, from_cache: bool) -> None:
    """
    Accumulate token and cost counters into meta.
    Cache hits: tokens recorded (original usage) but cost = 0.
    """
    if not from_cache:
        meta["api_calls"] += 1
    meta["input_tokens"] += usage["input_tokens"]
    meta["output_tokens"] += usage["output_tokens"]
    meta["api_retry_count"] += usage["api_retry_count"]
    # usage["cost_usd"] is already 0.0 for cache hits (set in llm.py)
    meta["cost_usd"] += usage["cost_usd"]


# ---------------------------------------------------------------------------
# Correction message builder
# ---------------------------------------------------------------------------

def _correction_description(val_result) -> str:
    """Build a human-readable description of what failed for the correction message."""
    parts = []
    if val_result.schema_errors:
        parts.append("Schema validation errors:\n  - " + "\n  - ".join(val_result.schema_errors))
    if val_result.totals_errors:
        parts.append("Totals arithmetic errors:\n  - " + "\n  - ".join(val_result.totals_errors))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract invoice data from a PDF or image file."
    )
    parser.add_argument("file", help="Path to a PDF, JPG, or PNG file.")
    parser.add_argument(
        "--mode",
        choices=["text", "vision"],
        default=None,
        help="Force input mode (default: auto-detect).",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    stem = file_path.stem
    t0 = time.time()
    meta = _blank_meta()

    # ------------------------------------------------------------------ #
    # Step 1 — Load document                                              #
    # ------------------------------------------------------------------ #
    try:
        doc = load_document(str(file_path), force_mode=args.mode)
    except LoadError as exc:
        _hard_fail("load_error", str(exc), stem, meta)

    file_bytes = file_path.read_bytes()
    meta["input_mode"] = doc["input_mode"]

    # ------------------------------------------------------------------ #
    # Step 2 — First LLM call                                             #
    # ------------------------------------------------------------------ #
    messages = build_messages(doc["content"], doc["input_mode"])

    try:
        first_result, first_usage, first_from_cache = call_llm(
            messages=messages,
            file_bytes=file_bytes,
            input_mode=doc["input_mode"],
            prompt_version=PROMPT_VERSION,
            attempt_type="first",
        )
    except RuntimeError as exc:
        _hard_fail("api_error", str(exc), stem, meta)

    _add_usage(meta, first_usage, first_from_cache)

    # ------------------------------------------------------------------ #
    # Step 3 — Validate first result                                      #
    # ------------------------------------------------------------------ #
    first_val = validate_extraction(first_result)

    # ------------------------------------------------------------------ #
    # Step 4 — Correction retry (at most once, only on schema/totals fail)#
    # ------------------------------------------------------------------ #
    final_result = first_result
    final_val = first_val

    if first_val.should_retry():
        meta["correction_attempted"] = True
        correction_desc = _correction_description(first_val)

        correction_msgs = build_correction_messages(
            original_messages=messages,
            first_response_json=json.dumps(first_result),
            failure_description=correction_desc,
        )

        try:
            second_result, second_usage, second_from_cache = call_llm(
                messages=correction_msgs,
                file_bytes=file_bytes,
                input_mode=doc["input_mode"],
                prompt_version=PROMPT_VERSION,
                attempt_type="correction",
                correction_msg=correction_desc,  # part of cache key
            )
            _add_usage(meta, second_usage, second_from_cache)

            second_val = validate_extraction(second_result)

            # Determine outcome by counting schema+totals errors only
            first_ec = first_val.retry_error_count()
            second_ec = second_val.retry_error_count()

            if second_ec < first_ec:
                meta["correction_outcome"] = "helped"
                final_result = second_result
                final_val = second_val
            elif second_ec == first_ec:
                meta["correction_outcome"] = "no_change"
                # Keep original on tie
            else:
                meta["correction_outcome"] = "worse"
                # Keep original

            # Store both extractions in metadata when correction was attempted
            meta["first_extraction"] = first_result
            meta["correction_extraction"] = second_result

        except RuntimeError as exc:
            # Correction API call itself failed — keep first result
            meta["correction_outcome"] = "n/a"
            meta["review_reasons"].append("correction_api_error")
            # first result stays as final_result / final_val (already set above)

    # ------------------------------------------------------------------ #
    # Step 5 — Set final review status                                    #
    # ------------------------------------------------------------------ #
    meta["validation_errors"] = final_val.errors
    if final_val.needs_review:
        meta["needs_review"] = True

    # Translate validation error codes to review_reasons (deduplicated)
    if final_val.schema_errors:
        _add_reason(meta, "schema_validation_failed")
    if final_val.totals_errors:
        _add_reason(meta, "totals_mismatch")
    for err in final_val.errors:
        if err.startswith("date_invalid:"):
            _add_reason(meta, "date_invalid")
        if err.startswith("date_null_on_invoice:"):
            _add_reason(meta, "date_null_on_invoice")

    # ------------------------------------------------------------------ #
    # Step 5b — Heuristic review flags (BONUS, Phase 5)                  #
    # These run after CORE validation and are kept as separate codes.    #
    # ------------------------------------------------------------------ #
    extracted_text = doc.get("content") if doc["input_mode"] == "text" else None
    meta["review_reasons"], h_flagged = add_heuristic_flags(
        result=final_result,
        review_reasons=meta["review_reasons"],
        extracted_text=extracted_text,
        input_mode=doc["input_mode"],
    )
    if h_flagged:
        meta["needs_review"] = True

    # ------------------------------------------------------------------ #
    # Step 6 — Write metadata and emit result                             #
    # ------------------------------------------------------------------ #
    meta["processing_status"] = "completed"
    meta["seconds"] = round(time.time() - t0, 2)
    _write_metadata(stem, meta)

    # Print the extraction JSON to stdout.
    # Even if validation failed we still print it; needs_review is in metadata.
    print(json.dumps(final_result, indent=2))


def _add_reason(meta: dict, reason: str) -> None:
    if reason not in meta["review_reasons"]:
        meta["review_reasons"].append(reason)


if __name__ == "__main__":
    main()
