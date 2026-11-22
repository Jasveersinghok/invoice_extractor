# Invoice Extractor

An invoice-extraction pipeline that uses a vision-language model (OpenRouter → `nvidia/nemotron-3.5-super-vl`) to extract structured data from PDF and image invoices. Supports digital PDFs (text mode) and scanned PDFs / JPEG images (vision mode), with automatic mode detection, schema validation, financial reconciliation, and a correction-retry loop.

For full methodology, accuracy metrics, failure analysis, and cost breakdown see **[REPORT.md](REPORT.md)**.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill in your OPENROUTER_API_KEY
```

**`.env` variables:**

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3.5-super-vl` | Model identifier |
| `PRICE_IN_PER_MTOK` | `0.06` | Input token price (USD per million) |
| `PRICE_OUT_PER_MTOK` | `0.35` | Output token price (USD per million) |
| `PRICE_DATE` | `2026-11-22` | Date you verified the prices |

---

## Datasets

The document sets are **not included** in this repository (confidential). Before running the pipeline, place the following directories in the project root:

- `labelled/` — 10 labelled documents (PDF/JPG) with matching ground-truth JSON files (same directory, same stem).
- `unlabelled/` — 10 unlabelled documents (PDF/JPG).

Both directories are listed in `.gitignore` and will never be committed.

---

## Run

### Extract one document

```bash
python extract.py path/to/invoice.pdf
```

Prints extraction JSON to stdout. Automatically selects text or vision mode.

```bash
python extract.py --mode text   path/to/invoice.pdf   # force text mode (PDF only)
python extract.py --mode vision path/to/invoice.pdf   # force vision mode
```

**Exit codes:**
- `0` — success. JSON printed to stdout.
- `1` — hard failure. Nothing on stdout; error JSON on stderr.

### Evaluate against 10 ground-truth labels

```bash
python evaluate.py               # run extractor + compare against ground truth
python evaluate.py --no-run      # compare using cached extractions (no API calls)
```

Prints per-field and per-document accuracy tables. Saves:
- `outputs/eval_results.json`
- `outputs/eval_results.csv`
- Appends a run record to `experiments.md`

### Run tests

```bash
python -m pytest tests/test_core.py -v
```

36 tests covering: corrupt/empty file handling, CLI exit codes, API timeout and unparseable response handling, schema validation, date validation, totals arithmetic, and retry flag logic.

---

## Web Interface

The project includes a local Flask-based interface for interactive invoice extraction — no command line needed.

### Start the server

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

### How to use it

1. Click **Choose File** and select a PDF, JPG, or PNG invoice (up to 20 MB).
2. Click **Extract**.
3. The extracted fields appear on the right alongside a document preview — vendor, invoice number, date, currency, line items, tax total, and grand total.
4. A metadata panel shows the input mode (text / vision), model used, token counts, estimated cost, and any validation or review flags.

### Notes

- Requires the same `.env` with `OPENROUTER_API_KEY` as the CLI.
- Uploaded files are saved temporarily to `outputs/uploads/` (not committed — covered by `.gitignore`).
- The same correction-retry and schema-validation logic used by the CLI runs here too.
- Runs on port `5000` by default; pass `--port` or set `FLASK_RUN_PORT` to change it.

---

## Architecture

| File | Responsibility |
|---|---|
| `src/loader.py` | File-type detection; PyMuPDF text extraction; auto-detect text vs vision; `LoadError` for bad inputs |
| `src/llm.py` | OpenRouter API client, `temperature=0`, SHA-256 disk cache, 3-attempt exponential-backoff retry, usage recording |
| `src/prompts.py` | Versioned prompt (`PROMPT_VERSION="v1"`), extraction conventions, `build_messages()`, `build_correction_messages()` |
| `src/validate.py` | JSON-schema validation, real calendar-date check, Decimal totals arithmetic (tolerance 0.01), `ValidationResult.should_retry()` |
| `src/review.py` | Heuristic review flags: missing fields, short text, value-not-in-text. Never an LLM confidence score |
| `src/matching.py` | Ground-truth comparison: vendor strict/lenient, numeric matching (±0.01), line-item F1 with description-similarity threshold |
| `extract.py` | CLI entry point; two-stage correction-retry loop; metadata writing |
| `evaluate.py` | Batch evaluation harness; per-field and per-document tables; CSV/JSON export |
| `tests/test_core.py` | 36 unit/integration tests |

### Input mode detection

`src/loader.py` tries PyMuPDF text extraction first. If the extracted text has ≥ 100 characters the document is classified as a **digital PDF** and processed in **text mode** — the raw text is sent to the model. If fewer than 100 characters are extracted (scanned PDF, image-only PDF, or JPEG/PNG) every page is rendered to a PNG and sent in **vision mode**.

You can override with `--mode text` or `--mode vision`.

### Model configuration

The model is set via `OPENROUTER_MODEL` in `.env`. The pipeline was developed and evaluated with:

```
nvidia/nemotron-3.5-super-vl
```

This model handles both plain text and inline base64 PNG images in the same API call. Pricing used: **\$0.06/M input tokens**, **\$0.35/M output tokens** (checked 2026-11-22).

Reasoning tokens are disabled explicitly (`"reasoning": {"effort": "none"}` in `extra_body`) to keep output tokens — and therefore cost — predictable.

---

## Output locations

| Path | Contents |
|---|---|
| `outputs/metadata/<stem>.json` | Tokens, cost, input mode, validation errors, correction outcome, timing |
| `outputs/extractions/<stem>.json` | Final extraction JSON (written by `evaluate.py`) |
| `outputs/cache/` | SHA-256 disk cache (avoids duplicate API calls) |
| `outputs/eval_results.json` | Aggregate and per-document evaluation metrics |
| `outputs/eval_results.csv` | Same data in CSV format |
| `experiments.md` | Auto-appended run log |

---

## Output schema

```json
{
  "is_invoice": true,
  "vendor": "Acme Corp",
  "invoice_number": "INV-001",
  "invoice_date": "2026-01-15",
  "currency": "USD",
  "line_items": [
    {"description": "Widget", "quantity": 10, "amount": 100.0}
  ],
  "tax_total": 10.0,
  "total": 110.0
}
```

Non-invoices: `is_invoice: false`, every other field `null`.
