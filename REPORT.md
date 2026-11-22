# Technical Report — Invoice Extraction Pipeline

**Pipeline version:** v1 (frozen baseline)  
**Model:** `nvidia/nemotron-3.5-super-vl` via OpenRouter  
**Evaluation date:** 2026-11-22  
**Documents:** 20 total (10 labelled with ground truth, 10 unlabelled)

---

## 1. Approach

### Model selection

The pipeline uses `nvidia/nemotron-3.5-super-vl`, a vision-language model available through the OpenRouter API (OpenAI-compatible endpoint). It was selected because it handles both plain text and inline base64 PNG images in a single API call, making it suitable for the auto-detect text/vision workflow without needing separate models per input type.

The model is called at `temperature=0` with `max_tokens=1024` and reasoning explicitly disabled (`"reasoning": {"effort": "none"}`) for deterministic, cost-predictable output.

### Input detection and processing

```
Document (PDF / JPEG)
      │
      ▼
  loader.py
  ├── PyMuPDF text extraction
  │   └── ≥ 100 chars → TEXT MODE  → raw text in user message
  └── < 100 chars    → VISION MODE → pages rendered to PNG, embedded as base64
```

**Text mode** (15 of 20 docs): digital PDFs with extractable embedded text. Compact, low-token-count, high-accuracy path.

**Vision mode** (5 of 20 docs): scanned PDFs (L09, L10), JPEG image (L02), and two unlabelled image documents (U02, U05). Pages are rendered at 2× zoom (≈144 DPI) for legibility.

### Prompting strategy

A single versioned system prompt (`PROMPT_VERSION="v1"`) defines all extraction conventions:
- Output only the JSON object — no markdown fences, no commentary.
- Never invent values; use `null` for absent fields.
- `vendor` = the **seller/issuer**, not the buyer.
- `invoice_date` = the **issue date** in ISO `YYYY-MM-DD`; set to `null` if day/month order is ambiguous.
- `line_items[].amount` = the **row total** (not unit price); discount rows included with negative amounts.
- `tax_total` = **sum of all tax lines** as a plain number.

### Validation and correction-retry

Every extraction is validated against `schema.json` (JSON Schema Draft 2020-12) and checked for financial reconciliation: `sum(line_items[].amount) + tax_total = total` within a tolerance of 0.01. When either check fails, a **single correction attempt** is made: the model receives its first response, the specific error message, and an instruction to re-read and report exactly what is printed without adjusting numbers to reconcile totals.

The corrected result is adopted only if it has strictly fewer errors. The outcome is recorded as `helped`, `no_change`, or `worse` in per-document metadata.

API-level failures (timeout, 429, 5xx) use a separate exponential-backoff retry with up to 3 attempts in `src/llm.py`. All successful responses are cached by SHA-256 of `(file_bytes, input_mode, prompt_version, model, attempt_type)`.

### v2 prompt experiment (unsuccessful)

A second prompt version was drafted to improve tax-total accuracy, specifically to address L02's misread. The v2 prompt added explicit instructions to identify every printed tax component and sum their exact monetary amounts. During testing, v2 caused a **regression on L10**: the model misread a digit in the tax figure on the scanned document, reducing overall `tax_total` accuracy from 8/9 (89%) to 7/9 (78%). Since the net result was worse than v1, the experiment was reverted and v1 was restored as the frozen baseline.

### Regex + local OCR experiment (`regex_experiment/`)

A completely separate folder implements the same extraction pipeline using Python regex and PyMuPDF text extraction, with Tesseract OCR as the fallback for scanned documents (if installed). No LLM or API calls are used.

**Results on the same 10 labelled documents:**

| Field | Regex | Nemotron v1 | Δ |
|---|---|---|---|
| `is_invoice` | 7/10 (70%) | 10/10 (100%) | −30% |
| `vendor` | 6/6 (100%) | 9/9 (100%) | = |
| `invoice_number` | 6/6 (100%) | 9/9 (100%) | = |
| `invoice_date` | 6/6 (100%) | 9/9 (100%) | = |
| `currency` | 6/6 (100%) | 9/9 (100%) | = |
| `tax_total` | 6/6 (100%) | 8/9 (89%) | **+11%** |
| `total` | 6/6 (100%) | 9/9 (100%) | = |
| `schema_valid` | 4/10 (40%) | 10/10 (100%) | −60% |
| Line Items F1 | 0.527 | 1.000 | −0.473 |

> **Denominator note:** Regex denominators are smaller (6, not 9) because L02, L09, L10 are image/scanned documents that could not be read without Tesseract installed on the test machine. These three documents count as full failures across all fields.

**Key finding:** On text-extractable PDFs, regex matches the LLM on every scalar field — vendor, number, date, currency, tax and total are all correct. The critical weaknesses are (a) no OCR capability without Tesseract, and (b) line-item extraction fails on PDFs where the PDF reader emits one column per line rather than one row per line. The regex experiment is archived in `regex_experiment/` and was **not** adopted as the production pipeline.

---

## 2. Accuracy

### Final pipeline: Nemotron v1 on 10 labelled documents

#### Field-level metrics

| Field | Correct | Denominator | Accuracy | Matching rule |
|---|---|---|---|---|
| `vendor` | 9 | 9 | **100%** | Lenient: lowercase + strip punctuation + strip legal suffixes (Ltd, Pvt, etc.) |
| `invoice_number` | 9 | 9 | **100%** | Exact match after strip + lowercase |
| `invoice_date` | 9 | 9 | **100%** | Exact string match (both sides are ISO `YYYY-MM-DD`) |
| `currency` | 9 | 9 | **100%** | Exact match, case-insensitive |
| Line items (F1) | — | 66 GT items | **F1 = 1.000** | Greedy one-to-one matching: description similarity ≥ threshold AND row amount within ±0.01 |
| `tax_total` | 8 | 9 | **89%** | Numeric match within ±0.01 |
| `total` | 9 | 9 | **100%** | Numeric match within ±0.01 |
| `is_invoice` | 10 | 10 | **100%** | Exact boolean match |

**Line-item detail:** TP = 66, FP = 0, FN = 0, Precision = 1.000, Recall = 1.000, F1 = 1.000. All 66 ground-truth line items across 9 invoices were matched exactly; no spurious items were predicted.

#### Denominator explanation

The denominator for `vendor`, `invoice_number`, `invoice_date`, `currency`, `tax_total`, and `total` is **9, not 10**, because L08 is a Purchase Order, not an invoice. Its ground-truth values for all those fields are `null`. A `null`/`null` pair counts as a correct non-invoice classification under `is_invoice`, but it is excluded from the per-field denominators to avoid inflating accuracy with trivially-correct null predictions.

#### Supplementary metrics

| Metric | Value |
|---|---|
| Schema valid | 10/10 (100%) |
| Financial reconciliation (`sum(lines) + tax = total`, ±0.01) | 9/10 (90%) |
| Exact document match (all fields simultaneously correct) | 9/10 (90%) |

The one document that fails financial reconciliation is **L02**: the tax figure is off by 20.00, which means `sum(lines) + tax_total ≠ total`. The total itself is correct; the error is in the tax subtotal only (see Failure Analysis).

---

## 3. Failure Analysis

### L02 — Tax misread (the only field-level failure)

**Document type:** JPEG scan of a hand-prepared Indian invoice (vision mode)  
**Vendor:** Shree Ganesh Stationers | **Invoice:** SGS/1847

| | `tax_total` | `total` |
|---|---|---|
| **Ground truth** | 3,529.80 | 23,139.80 |
| **v1 prediction** | 3,549.80 | 23,139.80 |
| **Error** | +20.00 | — |

The invoice contains two tax components — CGST and SGST — each of ₹1,764.90, summing to ₹3,529.80. The model returned ₹3,549.80. The error of exactly 20.00 is consistent with misreading one component as ₹1,784.90 instead of ₹1,764.90 — a likely OCR-style digit substitution on the scanned image. The grand total (₹23,139.80) was read correctly, which demonstrates that the error is specific to the tax sub-row rather than a systematic arithmetic failure.

**Correction retry behaviour:** The totals mismatch triggered a correction attempt. The correction message stated the exact arithmetic error and asked the model to re-read the tax lines. The correction returned an **identical result** (`correction_outcome: no_change`), indicating the model consistently misreads the same digit across two independent reads from the same image. The misread persisted through v2 as well.

**Root cause (tentative):** The digit `6` in `1,764.90` is likely rendered ambiguously in the scan or compressed JPEG artefact. We cannot confirm without access to the original image at higher resolution. What we can confirm is that neither the correction retry nor the v2 prompt change resolved it.

**What to try next:** Test a stronger vision model (e.g. `google/gemini-2.0-flash` or `anthropic/claude-3.5-sonnet`) on L02 specifically. If a stronger model reads the digit correctly, the fix is a model upgrade for vision-mode documents. If the error persists, the scan quality itself is the bottleneck, and the fix is pre-processing (image sharpening, contrast enhancement) before the API call.

### v2 prompt regression — L10 digit misread

The v2 prompt added tax-extraction instructions intended to fix L02. During testing on L10 (a different scanned PDF, vision mode), the model introduced a new digit error in the tax figure that was not present under v1. This demonstrates that prompt changes intended to fix one vision-mode document can introduce regressions on others when the root cause is image quality rather than prompting. v2 was reverted immediately; the net effect was a reduction from 8/9 (89%) to 7/9 (78%) on `tax_total`.

### Regex experiment — classification, schema and line-item failures

**`is_invoice` at 70% (3 wrong, all OCR-related):** L02, L09, L10 are image/scanned documents. Without Tesseract installed, the loader cannot extract any text from them, so the extractor returns a null result and `is_invoice` defaults to `False`. All three ground-truth labels are `is_invoice: True`. This is entirely a missing-dependency failure, not a regex logic failure. The regex classifier itself correctly handled all 7 text-mode documents including L08 (Purchase Order → `is_invoice: False`).

**`schema_valid` at 40%:** Four of the ten documents fail financial reconciliation. L02, L09, L10 fail because their `line_items` and `tax_total` are null (OCR unavailable), so `sum([]) + null ≠ total`. L04, L06, L07 fail because the multi-line block reassembler does not recover all line items from PDFs with extra columns (e.g., HSN/SAC codes, discount rows, service-line layouts without row numbers). The result: `sum(recovered_items) + tax ≠ total`.

**Line items at F1 = 0.527:** The regex extractor uses two strategies — inline single-line row parsing and multi-line block reassembly using ascending row-counter detection. The first strategy works for compact single-line table rows; the second works for uniform column-per-line layouts (L01, L03). It fails for irregular layouts such as service invoices without row numbers (L06, L07) and invoices with inline discount rows (L07). The LLM handles all these layouts correctly because it understands document structure semantically rather than syntactically.

### What to fix first

1. **L02 tax misread:** Try a stronger vision model on the specific JPEG. If the model upgrade fixes it, `tax_total` accuracy rises to 9/9 (100%) and financial reconciliation to 10/10.
2. **Regex line items:** Replace the row-counter heuristic with a layout-aware table parser (e.g., `pdfplumber` table extraction) for PDFs with irregular table structures.
3. **Regex OCR:** Install Tesseract (`winget install UB-Mannheim.TesseractOCR`) to unlock L02, L09, L10 and test whether regex extraction is accurate on those documents once text is available.

---

## 4. Cost per Document

### Nemotron v1 — API costs

All 20 documents (10 labelled + 10 unlabelled) were processed. Costs are computed from recorded token counts at the stated pricing. The `cost_usd` field in metadata files is 0.00 for all documents because all runs in the final evaluation used the disk cache (no live API calls). Token counts reflect the **original live call** usage, stored at cache write time.

**Pricing:** \$0.06 per million input tokens, \$0.35 per million output tokens (OpenRouter, `nvidia/nemotron-3.5-super-vl`, checked 2026-11-22). No reasoning tokens — reasoning was explicitly disabled.

#### Aggregate token and cost summary (all 20 documents)

| Metric | Value |
|---|---|
| Total input tokens | 31,910 |
| Total output tokens | 4,950 |
| Total cost (20 docs) | **\$0.003647** |
| Average cost per document | **\$0.000182** |
| Corrections triggered | 1 (L02) |
| Correction outcome | `no_change` |
| Live API calls | 0 (all served from cache) |
| API retries (infrastructure) | 0 |

#### Breakdown by document set

| Set | Docs | Input tok | Output tok | Cost (USD) |
|---|---|---|---|---|
| Labelled (L01–L10) | 10 | ~20,616 | ~3,312 | \$0.002373 |
| Unlabelled (U01–U10) | 10 | ~11,294 | ~1,638 | \$0.001274 |
| **Total** | **20** | **31,910** | **4,950** | **\$0.003647** |

#### Cost calculation

```
Cost = (input_tokens / 1,000,000 × 0.06) + (output_tokens / 1,000,000 × 0.35)
     = (31,910 / 1,000,000 × 0.06) + (4,950 / 1,000,000 × 0.35)
     = 0.001915 + 0.001733
     = $0.003647 (rounded to 6 decimal places)
```

#### Per-mode token profile

| Mode | Docs | Avg input tok/doc | Avg output tok/doc | Notes |
|---|---|---|---|---|
| Text | 15 | ~953 | ~198 | Compact; low vision overhead |
| Vision | 5 | ~3,151 | ~341 | Pages encoded as base64 PNG; L02 also includes correction attempt tokens |

Vision-mode documents consume ~3.3× more input tokens than text-mode documents, primarily because base64-encoded PNG images are much larger than the equivalent extracted text. L02 is the most expensive single document at 7,687 input tokens and 589 output tokens (includes one correction attempt), costing \$0.000668 alone.

#### Regex experiment — cost

**\$0.00.** The regex extractor uses only local computation — PyMuPDF for text extraction and Python regex for field parsing. There are no API calls and no cloud services. If Tesseract were installed, OCR processing would add CPU time but no monetary cost.
