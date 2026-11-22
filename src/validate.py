"""
Validate an extracted invoice dict against schema.json and business rules.

Usage:
    result = validate_extraction(data)
    result.errors          → list of human-readable error strings
    result.schema_errors   → subset that are schema / JSON-structure errors
    result.totals_errors   → subset that are totals-arithmetic errors
    result.needs_review    → True if any hard-review signal is present
    result.should_retry()  → True when a correction retry should be attempted
    result.retry_error_count() → number of errors that retry might fix
"""

import calendar
import dataclasses
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

import jsonschema

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent.parent / "schema.json"
_SCHEMA: dict | None = None  # loaded lazily and cached

TOLERANCE: Decimal = Decimal(os.environ.get("TOTALS_TOLERANCE", "0.01"))


def _get_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ValidationResult:
    """
    Structured output of validate_extraction().

    errors        : all error strings (superset of schema_errors + totals_errors)
    schema_errors : jsonschema violations only
    totals_errors : arithmetic totals-mismatch errors only
    needs_review  : True when any hard review signal is present
    """
    errors: list[str]
    schema_errors: list[str]
    totals_errors: list[str]
    needs_review: bool

    def passed(self) -> bool:
        """True when no errors of any kind were found."""
        return len(self.errors) == 0

    def should_retry(self) -> bool:
        """True when a correction retry should be attempted (schema or totals failed)."""
        return bool(self.schema_errors or self.totals_errors)

    def retry_error_count(self) -> int:
        """Number of errors that a correction retry could plausibly fix."""
        return len(self.schema_errors) + len(self.totals_errors)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_extraction(result: dict) -> ValidationResult:
    """
    Validate an extracted dict against schema.json and business rules.

    Checks performed (in order):
      1. JSON Schema (schema.json) — catches missing fields, wrong types,
         non-invoice not fully null, bad date pattern, bad currency pattern.
      2. Real calendar date — rejects dates like 2026-02-30 that pass the
         regex pattern but do not exist on the calendar.
      3. Null invoice_date on a true invoice — flags for review (may be absent
         or ambiguous; either way a human should check).
      4. Totals arithmetic — sum(line amounts) + tax_total must equal total
         within TOLERANCE using Decimal arithmetic.
    """
    all_errors: list[str] = []
    schema_errors: list[str] = []
    totals_errors: list[str] = []
    needs_review: bool = False

    # ---- 1. JSON Schema validation ----------------------------------------
    validator = jsonschema.Draft202012Validator(_get_schema())
    for err in sorted(validator.iter_errors(result), key=lambda e: str(e.json_path)):
        msg = f"schema: {err.json_path}: {err.message}"
        all_errors.append(msg)
        schema_errors.append(msg)
        needs_review = True

    # If schema is invalid we cannot safely dereference fields — stop here.
    if schema_errors:
        return ValidationResult(all_errors, schema_errors, totals_errors, needs_review)

    # ---- 2. Calendar date check -------------------------------------------
    date_val = result.get("invoice_date")
    if date_val is not None:
        if not _is_real_date(date_val):
            msg = f"date_invalid: '{date_val}' passes the date pattern but is not a real calendar date"
            all_errors.append(msg)
            # Date errors are not in schema_errors or totals_errors
            # → they do NOT trigger a correction retry
            needs_review = True

    # ---- 3. Null date on a true invoice (ambiguous / missing) -------------
    if result.get("is_invoice") and date_val is None:
        msg = "date_null_on_invoice: invoice_date is null; may be absent or ambiguous — verify"
        all_errors.append(msg)
        # Core review signal per spec ("ambiguous date"); does NOT trigger retry
        needs_review = True

    # ---- 4. Totals arithmetic check ---------------------------------------
    if result.get("is_invoice") and result.get("line_items") is not None:
        t_errors = _check_totals(result)
        all_errors.extend(t_errors)
        totals_errors.extend(t_errors)
        if t_errors:
            needs_review = True

    return ValidationResult(all_errors, schema_errors, totals_errors, needs_review)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_real_date(date_str: str) -> bool:
    """
    Return True if date_str (which already matches YYYY-MM-DD) is a real date.

    calendar.monthrange() tells us the number of days in a given month,
    handling leap years correctly.
    """
    try:
        year, month, day = (int(x) for x in date_str.split("-"))
        if not (1 <= month <= 12 and year >= 1):
            return False
        _, days_in_month = calendar.monthrange(year, month)
        return 1 <= day <= days_in_month
    except Exception:
        return False


def _to_decimal(value) -> Decimal:
    """Convert a numeric value (int or float) to Decimal via str() to avoid float rounding."""
    return Decimal(str(value))


def _check_totals(result: dict) -> list[str]:
    """
    Verify: sum(line_item amounts) + tax_total == total  (within TOLERANCE).

    Returns a list of error strings (empty when totals balance).
    Negatives (discount rows) are allowed and expected.
    """
    try:
        line_sum: Decimal = sum(_to_decimal(item["amount"]) for item in result["line_items"])

        # tax_total: null treated as 0 for the check (spec: "absent tax = 0")
        tax_raw = result.get("tax_total")
        tax: Decimal = _to_decimal(tax_raw) if tax_raw is not None else Decimal("0")

        total_raw = result.get("total")
        if total_raw is None:
            # Can't check without a total — not an error we can verify
            return []
        total: Decimal = _to_decimal(total_raw)

        expected: Decimal = line_sum + tax
        diff: Decimal = abs(expected - total)

        if diff > TOLERANCE:
            return [
                f"totals_mismatch: "
                f"sum_of_lines({line_sum}) + tax({tax}) = {expected}, "
                f"but total = {total}  (diff={diff}, tolerance={TOLERANCE})"
            ]
        return []

    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        return [f"totals_check_error: {exc}"]

