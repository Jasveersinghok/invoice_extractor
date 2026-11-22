"""
Prompt management.

PROMPT_VERSION must be incremented whenever the prompt text changes so that
old cache entries are automatically invalidated (version is part of the cache key).
"""

import base64

PROMPT_VERSION: str = "v1"

# ---------------------------------------------------------------------------
# System prompt — all extraction conventions in one place
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are an invoice data extractor. Read the document and return a single JSON object.

RULES
- Output ONLY the JSON object. No explanation, no markdown code fences, no extra text.
- Never invent or guess values. If a field is absent from the document, use null.
- For non-invoice documents set is_invoice to false and every other field to null.

FIELD DEFINITIONS
is_invoice
  true if this is a commercial invoice; false otherwise.

vendor
  The seller or issuer of the invoice — NOT the buyer or customer.

invoice_number
  The invoice reference number exactly as printed. Preserve case, hyphens, and slashes.

invoice_date
  The ISSUE date of the invoice in ISO format YYYY-MM-DD.
  - Use the issue date, not the due date, not the order date.
  - If the day/month order is ambiguous (e.g. "04/05/2026" could be April 5 or May 4),
    set invoice_date to null.

currency
  3-letter ISO currency code, e.g. USD, GBP, INR, EUR.

line_items
  Array of line items in the ORDER they appear in the document.
  Each item has three fields:
    description : item text as printed in the document.
    quantity    : quantity as a plain number; null if not printed on that row.
    amount      : the ROW TOTAL (not the unit price) as a plain number.
                  Discount rows are INCLUDED with a NEGATIVE amount.

tax_total
  Sum of all tax lines as a plain number. Use 0 if no tax is shown.

total
  The final payable amount as a plain number.

NUMBER FORMAT
All numbers must be plain numerics. No currency symbols, no commas, no percent signs.
  Correct : 1200.5
  Wrong   : "$1,200.50"

OUTPUT SCHEMA  (all eight fields are required at the top level)
{
  "is_invoice": true or false,
  "vendor": "Seller Name" or null,
  "invoice_number": "INV-001" or null,
  "invoice_date": "YYYY-MM-DD" or null,
  "currency": "USD" or null,
  "line_items": [
    {"description": "Item description", "quantity": 2 or null, "amount": 100.0}
  ] or null,
  "tax_total": 18.0 or null,
  "total": 118.0 or null
}
"""


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def build_messages(content: str | list, input_mode: str) -> list[dict]:
    """
    Build the messages list for a first-attempt LLM call.

    Args:
        content    : str for text mode; list[bytes] (PNG pages) for vision mode.
        input_mode : "text" | "vision"

    Returns:
        List of message dicts compatible with the OpenAI chat API.
    """
    if input_mode == "text":
        user_text = f"Extract invoice data from this document text:\n\n{content}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    # Vision mode: embed each page image as a base64 data-URI
    image_parts: list[dict] = [
        {"type": "text", "text": "Extract invoice data from these document image(s):"}
    ]
    for page_bytes in content:
        b64 = base64.b64encode(page_bytes).decode()
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": image_parts},
    ]


def build_correction_messages(
    original_messages: list[dict],
    first_response_json: str,
    failure_description: str,
) -> list[dict]:
    """
    Build the messages list for a correction-retry attempt.

    Appends the model's first answer and a correction instruction to the
    original conversation.  The instruction must NOT tell the model to make
    numbers add up — it only asks the model to re-read and report accurately.

    Args:
        original_messages    : The messages sent in the first attempt.
        first_response_json  : The model's first response (as a JSON string).
        failure_description  : Human-readable description of what failed.

    Returns:
        Extended messages list for the correction attempt.
    """
    correction_text = (
        f"Your previous extraction had the following problem:\n"
        f"{failure_description}\n\n"
        "Please re-read the document carefully and extract the data again, "
        "reporting exactly what is printed. "
        "Do not adjust any numbers to force totals to match; "
        "report what the document actually shows."
    )

    return original_messages + [
        {"role": "assistant", "content": first_response_json},
        {"role": "user", "content": correction_text},
    ]

