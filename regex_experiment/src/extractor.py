"""
extractor.py — Pure-regex invoice field extractor for regex_experiment.
No LLMs, no API calls.  All patterns are general (not document-specific).

Key design insight: PyMuPDF text extraction often puts label and value on
*separate consecutive lines* (PDF column layout).  All extractors use a
"label then next-non-empty-line" strategy as well as inline patterns.
"""

import re
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lines(text: str) -> list:
    return text.splitlines()


def _parse_number(s: str) -> Optional[float]:
    """
    Parse a number string that may use Indian comma formatting
    (e.g. "1,00,000.50") or standard formatting ("1,234.56").
    Strips leading currency symbols and commas.
    Returns None if parsing fails.
    """
    if s is None:
        return None
    cleaned = re.sub(r"[£$₹€¥\s,]", "", s.strip())
    cleaned = cleaned.rstrip(".")
    try:
        v = float(cleaned)
        return v if v >= 0 else None
    except ValueError:
        return None


def _next_nonempty(lines: list, idx: int) -> Optional[str]:
    """Return the next non-empty line after index idx."""
    for j in range(idx + 1, min(idx + 5, len(lines))):
        s = lines[j].strip()
        if s:
            return s
    return None


def _first_number_in(s: str) -> Optional[float]:
    """First decimal number in a string."""
    for m in re.finditer(r"[£$₹€]?\s*[\d,]+\.?\d*", s):
        v = _parse_number(m.group())
        if v is not None and v > 0:
            return v
    return None


def _last_number_in(s: str) -> Optional[float]:
    """Last decimal number in a string."""
    nums = list(re.finditer(r"[£$₹€]?\s*[\d,]+\.?\d*", s))
    for m in reversed(nums):
        v = _parse_number(m.group())
        if v is not None and v > 0:
            return v
    return None


# ---------------------------------------------------------------------------
# is_invoice
# ---------------------------------------------------------------------------

_INVOICE_KW = re.compile(
    r"\b(tax\s+invoice|invoice|bill\s+of\s+sale|receipt|debit\s+note|credit\s+note|proforma)\b",
    re.IGNORECASE,
)
# Document-level headings that mean it is NOT an invoice despite mentioning the word
_NON_INVOICE_HEADINGS = re.compile(
    r"^(purchase\s+order|delivery\s+order|delivery\s+challan|quotation|quote|"
    r"purchase\s+requisition|work\s+order|packing\s+list)\b",
    re.IGNORECASE,
)

def _detect_invoice(text: str) -> bool:
    # First check: if the first non-empty line is a PO/non-invoice heading → False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _NON_INVOICE_HEADINGS.match(line):
            return False
        break  # Only check first non-empty line; then fall through

    # Look for the word "invoice" appearing as a document title/heading
    # (not just incidentally in body text like "attach a valid tax invoice")
    for line in text.splitlines()[:15]:
        if re.match(r"^\s*(tax\s+)?invoice\s*$", line, re.IGNORECASE):
            return True

    # Also check lines 0–10 for NON_INVOICE headings to catch POs
    for line in text.splitlines()[:10]:
        if _NON_INVOICE_HEADINGS.match(line.strip()):
            return False

    # Fallback: keyword search in full text — but only count leading occurrences
    m = _INVOICE_KW.search(text[:500])
    return bool(m)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

_SYMBOL_MAP = {"£": "GBP", "$": "USD", "₹": "INR", "€": "EUR", "¥": "JPY"}
_ISO_CODES   = re.compile(r"\b(GBP|USD|INR|EUR|JPY|CAD|AUD|SGD|AED|CHF)\b")

def _extract_currency(text: str) -> Optional[str]:
    # Symbol takes precedence
    for sym, code in _SYMBOL_MAP.items():
        if sym in text:
            return code
    # ISO code
    m = _ISO_CODES.search(text)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Invoice number
# ---------------------------------------------------------------------------

_INV_NUM_LABEL = re.compile(
    r"(?:invoice\s*(?:no\.?|number|#|num\.?)|inv\.?\s*(?:no\.?|#)|bill\s*(?:no\.?|number|#))\s*[:\-]?\s*",
    re.IGNORECASE,
)
_INV_NUM_VALUE = re.compile(r"[A-Z0-9][A-Z0-9/\-]{1,30}", re.IGNORECASE)

def _extract_invoice_number(text: str) -> Optional[str]:
    ls = _lines(text)
    for i, line in enumerate(ls):
        m = _INV_NUM_LABEL.search(line)
        if m:
            # Inline value
            rest = line[m.end():].strip()
            if rest:
                v = _INV_NUM_VALUE.match(rest)
                if v:
                    return v.group(0).strip()
            # Next line value
            nxt = _next_nonempty(ls, i)
            if nxt:
                v = _INV_NUM_VALUE.match(nxt.strip())
                if v:
                    return v.group(0).strip()
    return None


# ---------------------------------------------------------------------------
# Invoice date
# ---------------------------------------------------------------------------

_DATE_LABEL = re.compile(
    r"(?:invoice\s*date|date\s*of\s*invoice|issue\s*date|bill\s*date|(?<!\w)date(?!\w))\s*[:\-]?\s*",
    re.IGNORECASE,
)

# Date patterns ordered from most to least specific
_DATE_PATS = [
    # DD Month YYYY  or  Month DD, YYYY
    re.compile(
        r"\b(\d{1,2}[\s\-/](?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"[\s\-/,]\d{2,4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"[\s\-/,]+\d{1,2}[\s\-/,]+\d{2,4})\b",
        re.IGNORECASE,
    ),
    # DD/MM/YYYY or DD-MM-YYYY
    re.compile(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b"),
    # YYYY-MM-DD (ISO)
    re.compile(r"\b(\d{4}[/\-]\d{2}[/\-]\d{2})\b"),
]

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


def _normalize_date(raw: str) -> Optional[str]:
    """Convert a raw date string to ISO YYYY-MM-DD. Returns None on failure."""
    raw = raw.strip().rstrip(",")
    # ISO YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"(\d{4})[/\-](\d{2})[/\-](\d{2})$", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"(\d{2})[/\-](\d{2})[/\-](\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # DD Month YYYY
    m = re.match(r"(\d{1,2})[\s\-/,]+([A-Za-z]+)[\s\-/,]+(\d{2,4})$", raw)
    if m:
        day, mon_str, yr = m.group(1), m.group(2).lower(), m.group(3)
        mon = _MONTH_MAP.get(mon_str[:3])
        if mon:
            year = int(yr) if len(yr) == 4 else (2000 + int(yr))
            return f"{year}-{mon:02d}-{int(day):02d}"
    # Month DD, YYYY
    m = re.match(r"([A-Za-z]+)[\s\-/,]+(\d{1,2})[\s\-/,]+(\d{2,4})$", raw)
    if m:
        mon_str, day, yr = m.group(1).lower(), m.group(2), m.group(3)
        mon = _MONTH_MAP.get(mon_str[:3])
        if mon:
            year = int(yr) if len(yr) == 4 else (2000 + int(yr))
            return f"{year}-{mon:02d}-{int(day):02d}"
    return None


def _find_date_in(snippet: str) -> Optional[str]:
    for pat in _DATE_PATS:
        d = pat.search(snippet)
        if d:
            return _normalize_date(d.group(1).strip())
    return None


def _extract_date(text: str) -> Optional[str]:
    ls = _lines(text)
    # Try labelled dates (inline then next-line)
    for i, line in enumerate(ls):
        m = _DATE_LABEL.search(line)
        if m:
            rest = line[m.end():].strip()
            if rest:
                d = _find_date_in(rest)
                if d:
                    return d
            # Next line
            nxt = _next_nonempty(ls, i)
            if nxt:
                d = _find_date_in(nxt)
                if d:
                    return d
    # Fallback: scan text for first date pattern (skip "Due date")
    for pat in _DATE_PATS:
        for m in pat.finditer(text):
            # Avoid "due date" region
            start = max(0, m.start() - 20)
            context = text[start: m.start()].lower()
            if "due" in context:
                continue
            d = _normalize_date(m.group(1).strip())
            if d:
                return d
    return None


# ---------------------------------------------------------------------------
# Vendor
# ---------------------------------------------------------------------------

_VENDOR_LABEL = re.compile(
    r"^(from|bill\s+from|sold\s+by|seller|issued\s+by|supplier)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
# "Billed to" and similar — these mark the CLIENT block, not the vendor
_CLIENT_LABEL = re.compile(
    r"^(bill\s+to|billed\s+to|ship\s+to|sold\s+to|customer|client|buyer)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
_COMPANY_SUFFIX = re.compile(
    r"\b(ltd\.?|llc\.?|inc\\.?|pvt\.?|limited|corporation|corp\.?|co\.?|llp|plc)\b",
    re.IGNORECASE,
)
# Strip trailing invoice-number-like garbage (pipe + rest)
_VENDOR_GARBAGE = re.compile(r"\s*[\|]\s*.+$")
# Lines that look like addresses
_ADDRESS_LINE = re.compile(
    r"\b(?:floor|road|street|st\.|ave|avenue|lane|nagar|marg|colony|plot|sector|"
    r"building|tower|suite|block|phase|unit|no\.|pin\b|zip\b|gstin|ein|pan|vat\s+no)\b",
    re.IGNORECASE,
)


def _clean_vendor(raw: str) -> str:
    """Remove trailing pipe-separated invoice metadata."""
    return _VENDOR_GARBAGE.sub("", raw).strip()


def _extract_vendor(text: str) -> Optional[str]:
    ls = _lines(text)

    # Strategy 0: First line often encodes "CompanyName | InvoiceNo" as PDF bookmark
    if ls:
        first = _clean_vendor(ls[0].strip())
        if first and len(first.split()) >= 2 and not _ADDRESS_LINE.search(first):
            return first

    # Find where the client block starts so we can skip it
    client_start = len(ls)
    for i, line in enumerate(ls[:30]):
        if _CLIENT_LABEL.match(line.strip()):
            client_start = i
            break

    # Strategy 1: Explicit issuer label on its own line → next line is vendor
    for i, line in enumerate(ls[:client_start]):
        if _VENDOR_LABEL.match(line.strip()):
            nxt = _next_nonempty(ls, i)
            if nxt:
                return _clean_vendor(nxt)

    # Strategy 2: Company-suffix line before the client block
    for line in ls[1:min(client_start, 25)]:
        line = line.strip()
        if _COMPANY_SUFFIX.search(line) and len(line) > 4 and not _ADDRESS_LINE.search(line):
            return _clean_vendor(line)

    # Strategy 3: First non-empty non-keyword line in top 10
    _skip = re.compile(
        r"^(invoice|tax\s+invoice|bill|date|no\.|number|receipt|page|details|from|to)$",
        re.IGNORECASE,
    )
    for line in ls[:10]:
        line = line.strip()
        if line and not _skip.match(line) and not re.match(r"^\d", line):
            return _clean_vendor(line)

    return None


# ---------------------------------------------------------------------------
# Tax total — multi-line aware
# ---------------------------------------------------------------------------

_TAX_LABEL = re.compile(
    r"\b(CGST|SGST|IGST|GST|VAT|Service\s+Tax|Sales\s+Tax|HST|PST|TDS|Withholding\s+Tax)\b",
    re.IGNORECASE,
)
# A number that looks like a monetary amount (not a pure percentage rate)
# Negative lookahead excludes numbers immediately followed by %
_MONEY_RE = re.compile(r"[£$₹€]?\s*([\d,]+\.\d{1,2})(?!\s*%)\b")
# Percentage pattern — if a line is ONLY a rate, skip it
_ONLY_RATE = re.compile(r"^\s*[\d]+\.?\d*\s*%\s*$")


def _extract_tax_total(text: str) -> Optional[float]:
    ls    = _lines(text)
    total = 0.0
    found = False

    for i, line in enumerate(ls):
        if not _TAX_LABEL.search(line):
            continue
        # Look for a monetary amount in this line or the next non-empty line
        amount = None
        # Collect candidate lines: current line and next line
        for src in [line, (_next_nonempty(ls, i) or "")]:
            if _ONLY_RATE.match(src.strip()):
                continue
            # Find ALL monetary amounts in this source
            amounts = []
            for m in _MONEY_RE.finditer(src):
                v = _parse_number(m.group(1))
                if v is not None and v >= 1:
                    amounts.append(v)
            if amounts:
                # Pick the LARGEST amount (avoids picking the % rate like 8.25)
                # For "Sales tax @ 8.25% $334.13", both appear in line.
                # The tax amount is always larger than the rate.
                amount = max(amounts)
                break
        if amount is not None:
            total += amount
            found  = True

    return round(total, 2) if found else None

# ---------------------------------------------------------------------------
# Total — multi-line aware
# ---------------------------------------------------------------------------

_TOTAL_LABEL = re.compile(
    r"^(grand\s+total|total\s+amount\s+(?:due|payable)|amount\s+(?:due|payable)|"
    r"total\s+payable|net\s+(?:total|amount|payable)|balance\s+due|total)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
_TOTAL_INLINE = re.compile(
    r"\b(grand\s+total|total\s+amount\s+(?:due|payable)|amount\s+(?:due|payable)|"
    r"total\s+payable|net\s+(?:total|amount|payable)|balance\s+due|total)\b.*?([£$₹€]?\s*[\d,]+\.?\d*)",
    re.IGNORECASE,
)


def _extract_total(text: str) -> Optional[float]:
    ls         = _lines(text)
    candidates = []

    for i, line in enumerate(ls):
        stripped = line.strip()

        # Case 1: label alone on a line → value on next line
        if _TOTAL_LABEL.match(stripped):
            label = stripped.lower()
            nxt   = _next_nonempty(ls, i)
            if nxt:
                v = _first_number_in(nxt)
                if v and v > 0:
                    priority = 2 if any(k in label for k in ("grand", "due", "payable")) else 1
                    candidates.append((priority, v))

        # Case 2: label and value inline
        for m in _TOTAL_INLINE.finditer(line):
            label = m.group(1).lower()
            v     = _parse_number(m.group(2))
            if v and v > 0:
                priority = 2 if any(k in label for k in ("grand", "due", "payable")) else 1
                candidates.append((priority, v))

    if not candidates:
        return None
    # Highest priority, then largest value
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Line items — table-row extraction
# ---------------------------------------------------------------------------

_SKIP_ITEM = re.compile(
    r"^\s*(?:subtotal|sub\s*total|grand\s+total|total|tax|cgst|sgst|igst|gst|vat|"
    r"discount|shipping|freight|handling|balance|amount\s+due|invoice|bill|date|"
    r"vendor|seller|buyer|from|to|payment|terms|bank|account|ifsc|pan|gstin|"
    r"description|qty|quantity|unit|price|rate|amount|sr\.?\s*no|s\.?\s*no|"
    r"hsn|sac|hsncode|declaration|authorised|signatory|page)\b",
    re.IGNORECASE,
)
_ONLY_DIGITS  = re.compile(r"^\s*\d[\d,\.]*\s*$")
_NEG_AMOUNT   = re.compile(r"^\s*-[\d,]+\.\d{1,2}\s*$")
_AMOUNT_LINE  = re.compile(r"^\s*[£$₹€]?\s*([\d,]+\.\d{1,2})\s*$")
_ITEM_ROW_QPA = re.compile(
    r"^(?:\d+\s+)?(.{3,60}?)\s+(\d[\d,]*)\s+([\d,]+\.?\d+)\s+([\d,]+\.?\d+)\s*$"
)
_ITEM_ROW_PA = re.compile(
    r"^(?:\d+\s+)?(.{3,60}?)\s+([\d,]+\.?\d+)\s+([\d,]+\.?\d+)\s*$"
)


def _extract_line_items(text: str) -> list:
    """
    Two-strategy heuristic line-item extractor.

    Strategy A (inline): regex for 'desc qty unit amount' or 'desc unit amount'
    on a single line.

    Strategy B (multi-line blocks): PDFs that have one field per line:
        1              <- row number (ascending)
        Description
        HSN_code       <- 4-digit, skipped
        qty
        unit_price
        amount
    Uses ascending row-counter to reliably split blocks.
    """
    ls    = _lines(text)
    items = []

    # ---- Strategy A: inline single-line rows ----------------------------
    for raw in ls:
        line = raw.strip()
        if not line or _SKIP_ITEM.match(line) or _ONLY_DIGITS.match(line):
            continue
        if _NEG_AMOUNT.match(line):
            continue

        m = _ITEM_ROW_QPA.match(line)
        if m:
            desc   = m.group(1).strip()
            qty    = _parse_number(m.group(2))
            amount = _parse_number(m.group(4))
            if desc and amount and amount > 0 and not _SKIP_ITEM.match(desc):
                items.append({"description": desc, "quantity": qty, "amount": amount})
            continue

        m = _ITEM_ROW_PA.match(line)
        if m:
            desc   = m.group(1).strip()
            amount = _parse_number(m.group(3))
            unit   = _parse_number(m.group(2))
            if desc and amount and amount > 0 and unit != amount and not _SKIP_ITEM.match(desc):
                items.append({"description": desc, "quantity": None, "amount": amount})

    if items:
        return items

    # ---- Strategy B: multi-line block reassembly ------------------------
    header_idx = None
    footer_idx = len(ls)
    for idx, line in enumerate(ls):
        if re.search(r"\bDescription\b", line, re.IGNORECASE):
            header_idx = idx
        if header_idx is not None and re.search(
            r"\b(Subtotal|Sub\s*Total|Grand\s+Total)\b", line, re.IGNORECASE
        ):
            footer_idx = idx
            break

    if header_idx is None:
        return []

    body = ls[header_idx + 1 : footer_idx]

    # Walk body using ascending row counter to identify item boundaries
    expected_row = 1
    i = 0
    while i < len(body):
        line = body[i].strip()
        m_row = re.match(r"^(\d{1,3})$", line)
        if m_row and int(m_row.group(1)) == expected_row:
            # Collect lines until the next consecutive row number appears
            block = []
            j = i + 1
            while j < len(body) and len(block) < 12:
                bl = body[j].strip()
                if bl:
                    mr2 = re.match(r"^(\d{1,3})$", bl)
                    if mr2 and int(mr2.group(1)) == expected_row + 1:
                        break
                    block.append(bl)
                j += 1

            if len(block) >= 2:
                desc        = None
                amounts_dec = []  # entries with decimal → line amount
                small_ints  = []  # small plain integers → qty

                for entry in block:
                    am = _AMOUNT_LINE.match(entry)
                    if am:
                        v = _parse_number(am.group(1))
                        if v is not None:
                            amounts_dec.append(v)
                    elif _NEG_AMOUNT.match(entry):
                        pass   # discount — skip
                    elif _ONLY_DIGITS.match(entry):
                        raw_digs = re.sub(r"[,\s]", "", entry.strip())
                        if len(raw_digs) != 4:   # skip 4-digit HSN / SAC
                            v = _parse_number(entry)
                            if v is not None:
                                small_ints.append(v)
                    elif not _SKIP_ITEM.match(entry):
                        if desc is None:
                            desc = entry

                if desc and amounts_dec:
                    amount = amounts_dec[-1]
                    qty = (small_ints[-1]
                           if small_ints and small_ints[-1] < 10000
                           else None)
                    if amount > 0 and not _SKIP_ITEM.match(desc):
                        items.append({
                            "description": desc,
                            "quantity":    qty,
                            "amount":      amount,
                        })

            expected_row += 1
            i = j
        else:
            i += 1

    return items

# ---------------------------------------------------------------------------
# NULL result
# ---------------------------------------------------------------------------

NULL_RESULT = {
    "is_invoice":     False,
    "vendor":         None,
    "invoice_number": None,
    "invoice_date":   None,
    "currency":       None,
    "line_items":     None,   # schema requires null for non-invoices
    "tax_total":      None,
    "total":          None,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract(text: str) -> dict:
    """Given plain text, return a schema-compatible extraction dict."""
    if not text or not text.strip():
        return dict(NULL_RESULT)
    if not _detect_invoice(text):
        return dict(NULL_RESULT)

    return {
        "is_invoice":     True,
        "vendor":         _extract_vendor(text),
        "invoice_number": _extract_invoice_number(text),
        "invoice_date":   _extract_date(text),
        "currency":       _extract_currency(text),
        "line_items":     _extract_line_items(text),
        "tax_total":      _extract_tax_total(text),
        "total":          _extract_total(text),
    }
