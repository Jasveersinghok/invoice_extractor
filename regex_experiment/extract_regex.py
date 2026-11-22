"""
extract_regex.py — Run regex extraction on a single document.

Usage:
    python extract_regex.py <path_to_document> [--save]

Options:
    --save   Write outputs/extractions/<stem>.json, outputs/text/<stem>.txt,
             and outputs/metadata/<stem>.json

Outputs JSON to stdout.
"""

import sys
import os
import json
import time
import argparse

# Allow importing parent project's validate.py
_PARENT_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_PARENT_SRC))

# Local regex modules
_THIS_SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, os.path.abspath(_THIS_SRC))

from loader    import load_document
from extractor import extract, NULL_RESULT
import validate as _validate  # from parent src/


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def run(doc_path: str, save: bool = False) -> dict:
    stem = os.path.splitext(os.path.basename(doc_path))[0]
    t0   = time.time()

    # --- Load ---
    loaded = load_document(doc_path)
    input_mode = loaded["input_mode"]

    if input_mode == "ocr_unavailable":
        result = dict(NULL_RESULT)
        result["is_invoice"] = None  # unknown — could not read doc
        status = "ocr_unavailable"
        validation_errors = []
        needs_review = True
        review_reasons = ["Tesseract OCR not installed; document could not be read"]
        text = ""
    else:
        text   = loaded["content"] or ""
        result = extract(text)
        status = "completed"

        # --- Validate ---
        try:
            vr = _validate.validate_extraction(result)
            errs = vr.errors
        except Exception as e:
            errs = [str(e)]
        validation_errors = errs
        needs_review = bool(errs)
        review_reasons = errs if errs else []

    elapsed = time.time() - t0

    # --- Save ---
    if save:
        # extraction JSON
        extr_path = os.path.join(OUTPUTS_DIR, "extractions", f"{stem}.json")
        with open(extr_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # raw text
        if text:
            text_path = os.path.join(OUTPUTS_DIR, "text", f"{stem}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

        # metadata
        meta = {
            "document":          os.path.basename(doc_path),
            "processing_status": status,
            "input_mode":        input_mode,
            "ocr_available":     loaded["ocr_available"],
            "needs_review":      needs_review,
            "review_reasons":    review_reasons,
            "validation_errors": validation_errors,
            "text_chars":        loaded["text_chars"],
            "pages":             loaded["pages"],
            "seconds":           round(elapsed, 3),
            "extractor":         "regex",
        }
        meta_path = os.path.join(OUTPUTS_DIR, "metadata", f"{stem}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Regex invoice extractor")
    parser.add_argument("path", help="Path to invoice document")
    parser.add_argument("--save", action="store_true", help="Save outputs")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    result = run(args.path, save=args.save)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
