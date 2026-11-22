# regex_experiment

A fully self-contained invoice extraction pipeline that replaces the LLM
(Nemotron) with **traditional Python regex** and compares results against
the frozen Nemotron v1 baseline.

---

## Folder structure

```
regex_experiment/
├── src/
│   ├── __init__.py
│   ├── loader.py        # Document loader (PyMuPDF text + Tesseract OCR)
│   └── extractor.py     # Pure-regex field extractor
├── outputs/
│   ├── extractions/     # Per-document JSON predictions
│   ├── text/            # Raw extracted text per document
│   └── metadata/        # Processing metadata per document
├── extract_regex.py     # CLI: single-document extraction
├── evaluate_regex.py    # Batch evaluation + Nemotron comparison
├── requirements.txt
└── README.md            ← you are here
```

---

## Setup

```bash
pip install -r requirements.txt
```

### Tesseract OCR (for scanned PDFs and JPEG images)

Without Tesseract, documents L02, L09, L10 cannot be read and will be
counted as failures.  To enable OCR on Windows:

```bash
winget install UB-Mannheim.TesseractOCR
# After installation, tesseract should be in PATH automatically.
# If not, set TESSDATA_PREFIX or add to PATH manually.
```

---

## Run

**Single document:**
```bash
python extract_regex.py ..\documents\L01.pdf
python extract_regex.py ..\documents\L01.pdf --save
```

**Full evaluation (all 10 labelled docs, comparison with v1):**
```bash
cd regex_experiment
python evaluate_regex.py
```

---

## Architecture

```
Document (PDF / JPG)
       │
       ▼
  loader.py
  ├── PyMuPDF text extraction  ──▶  text mode (≥100 chars)
  └── Tesseract OCR            ──▶  ocr mode
       │ unavailable ──▶ ocr_unavailable (failure)
       ▼
  extractor.py  (pure regex)
  ├── is_invoice    keyword match
  ├── vendor        label + company-suffix heuristic
  ├── invoice_number  labeled pattern
  ├── invoice_date  multi-format date patterns
  ├── currency      symbol → ISO code
  ├── tax_total     sum all GST/VAT/Tax lines
  ├── total         grand total / amount due (highest priority)
  └── line_items    table row heuristic
       │
       ▼
  parent src/validate.py  (schema + financial checks)
       │
       ▼
  outputs/extractions/<stem>.json
```

---

## Limitations

| Limitation | Impact |
|-----------|--------|
| Tesseract not installed | L02, L09, L10 fail (3/10 labelled) |
| Regex cannot handle layout tables | Line item extraction is approximate |
| No semantic understanding | Vendor / date may be wrong on unusual layouts |
| Indian number format (1,00,000) | Handled by `_parse_number()` helper |

---

## Comparison with Nemotron v1

The evaluation script automatically loads `../outputs/eval_results.json`
(frozen v1 baseline) and prints a side-by-side table.  Nemotron v1 results
(reference):

| Field          | v1 Score |
|----------------|----------|
| is_invoice     | 10/10    |
| vendor         | 9/9      |
| invoice_number | 9/9      |
| invoice_date   | 9/9      |
| currency       | 9/9      |
| tax_total      | 8/9      |
| total          | 9/9      |
| schema_valid   | 10/10    |
| Line Items F1  | 1.000    |

