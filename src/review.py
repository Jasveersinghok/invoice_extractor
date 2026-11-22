"""
Heuristic review flags (Phase 5 — BONUS signals).

These are separate from the CORE signals in validate.py.
They are added to metadata AFTER validate.py runs and are tagged with their
own reason codes so they can be reported and measured independently.

Heuristic signals (do not trigger correction retry):
  heuristic_missing_vendor  — vendor is null on a claimed invoice
  heuristic_missing_total   — total is null on a claimed invoice
  heuristic_missing_date    — invoice_date is null (overlaps with core date_null_on_invoice)
  heuristic_short_text      — extracted text is very short (< SHORT_TEXT_THRESHOLD chars)
  heuristic_vendor_not_in_text   — vendor string not found in extracted text (digital PDFs only)
  heuristic_inv_num_not_in_text  — invoice number not found in extracted text (digital PDFs only)
  heuristic_total_not_in_text    — total value string not found in extracted text (digital PDFs only)

None of these are LLM confidence scores.
"""

import re
from decimal import Decimal

SHORT_TEXT_THRESHOLD = 50  # chars; below this the extraction likely missed content


def add_heuristic_flags(
    result: dict,
    review_reasons: list[str],
    extracted_text: str | None = None,
    input_mode: str = "text",
) -> tuple[list[str], bool]:
    """
    Check heuristic review signals and return updated (review_reasons, needs_review).

    Args:
        result         : The (possibly corrected) extraction dict.
        review_reasons : Existing list of reason codes from validate.py (modified in place).
        extracted_text : Raw text extracted from the document (None for vision mode).
        input_mode     : "text" | "vision" — text-presence checks only run in text mode.

    Returns:
        (updated_review_reasons, any_heuristic_flagged)
    """
    flagged = False

    if not result.get("is_invoice"):
        return review_reasons, flagged  # non-invoices: no heuristic checks needed

    # --- Structural heuristics (always run) ---
    if result.get("vendor") is None:
        review_reasons = _add(review_reasons, "heuristic_missing_vendor")
        flagged = True

    if result.get("total") is None:
        review_reasons = _add(review_reasons, "heuristic_missing_total")
        flagged = True

    if result.get("invoice_date") is None:
        review_reasons = _add(review_reasons, "heuristic_missing_date")
        flagged = True

    # --- Text-length heuristic (text mode only) ---
    if input_mode == "text" and extracted_text is not None:
        if len(extracted_text.strip()) < SHORT_TEXT_THRESHOLD:
            review_reasons = _add(review_reasons, "heuristic_short_text")
            flagged = True

    # --- Value-not-in-text heuristics (text mode only) ---
    if input_mode == "text" and extracted_text is not None:
        text_lower = extracted_text.lower()

        vendor = result.get("vendor")
        if vendor and vendor.lower() not in text_lower:
            review_reasons = _add(review_reasons, "heuristic_vendor_not_in_text")
            flagged = True

        inv_num = result.get("invoice_number")
        if inv_num and inv_num.lower() not in text_lower:
            review_reasons = _add(review_reasons, "heuristic_inv_num_not_in_text")
            flagged = True

        total = result.get("total")
        if total is not None:
            # Look for the numeric value in the text (various formatting styles)
            total_str = _total_variants(total)
            if not any(v in extracted_text for v in total_str):
                review_reasons = _add(review_reasons, "heuristic_total_not_in_text")
                flagged = True

    return review_reasons, flagged


def _add(reasons: list[str], code: str) -> list[str]:
    if code not in reasons:
        reasons.append(code)
    return reasons


def _total_variants(total) -> list[str]:
    """
    Return several string representations of a numeric total to search in text.
    E.g. 1200.5 → ["1200.5", "1200.50", "1,200.5", "1,200.50"]
    """
    try:
        d = Decimal(str(total))
        base = str(d.normalize())
        # Also with 2 decimal places
        two_dp = f"{float(d):.2f}"
        # Also with comma thousands separator
        def comma(s):
            parts = s.split(".")
            integer = re.sub(r"(\d)(?=(\d{3})+$)", r"\1,", parts[0])
            return integer + ("." + parts[1] if len(parts) > 1 else "")

        variants = list({base, two_dp, comma(base), comma(two_dp)})
        return variants
    except Exception:
        return [str(total)]

