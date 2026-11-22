"""
Simple Flask frontend for the invoice extractor.

Usage:
    python app.py
Then open http://localhost:5000 in your browser.

Upload a PDF or image; the extracted fields are shown beside the document preview.
"""

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from loader import LoadError, load_document
from llm import MODEL, PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK, call_llm
from prompts import PROMPT_VERSION, build_correction_messages, build_messages
from validate import validate_extraction
from review import add_heuristic_flags

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

UPLOAD_DIR = Path("outputs") / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _extract(file_path: Path) -> tuple[dict, dict]:
    """
    Run the full extraction pipeline on file_path.
    Returns (result_dict, metadata_dict).
    result_dict has an extra key "_error" if extraction failed hard.
    """
    meta = {
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
        "needs_review": False,
        "review_reasons": [],
        "validation_errors": [],
        "processing_status": "failed",
    }

    # 1. Load
    try:
        doc = load_document(str(file_path))
    except LoadError as exc:
        return {"_error": str(exc)}, meta

    file_bytes = file_path.read_bytes()
    meta["input_mode"] = doc["input_mode"]

    # 2. First LLM call
    messages = build_messages(doc["content"], doc["input_mode"])
    try:
        result, usage, from_cache = call_llm(
            messages=messages,
            file_bytes=file_bytes,
            input_mode=doc["input_mode"],
            prompt_version=PROMPT_VERSION,
            attempt_type="first",
        )
    except RuntimeError as exc:
        return {"_error": str(exc)}, meta

    _add_usage(meta, usage, from_cache)

    # 3. Validate
    val = validate_extraction(result)

    # 4. Correction retry (once, only on schema/totals fail)
    if val.should_retry():
        meta["correction_attempted"] = True
        parts = []
        if val.schema_errors:
            parts.append("Schema errors:\n  - " + "\n  - ".join(val.schema_errors))
        if val.totals_errors:
            parts.append("Totals errors:\n  - " + "\n  - ".join(val.totals_errors))
        correction_desc = "\n".join(parts)

        correction_msgs = build_correction_messages(
            original_messages=messages,
            first_response_json=json.dumps(result),
            failure_description=correction_desc,
        )
        try:
            r2, u2, c2 = call_llm(
                messages=correction_msgs,
                file_bytes=file_bytes,
                input_mode=doc["input_mode"],
                prompt_version=PROMPT_VERSION,
                attempt_type="correction",
                correction_msg=correction_desc,
            )
            _add_usage(meta, u2, c2)
            val2 = validate_extraction(r2)
            if val2.retry_error_count() < val.retry_error_count():
                result, val = r2, val2
                meta["correction_outcome"] = "helped"
            elif val2.retry_error_count() == val.retry_error_count():
                meta["correction_outcome"] = "no_change"
            else:
                meta["correction_outcome"] = "worse"
        except RuntimeError:
            meta["correction_outcome"] = "n/a"

    # 5. Final review flags
    meta["validation_errors"] = val.errors
    if val.needs_review:
        meta["needs_review"] = True
    if val.schema_errors:
        _add_reason(meta, "schema_validation_failed")
    if val.totals_errors:
        _add_reason(meta, "totals_mismatch")

    extracted_text = doc.get("content") if doc["input_mode"] == "text" else None
    meta["review_reasons"], h_flagged = add_heuristic_flags(
        result=result,
        review_reasons=meta["review_reasons"],
        extracted_text=extracted_text,
        input_mode=doc["input_mode"],
    )
    if h_flagged:
        meta["needs_review"] = True

    meta["processing_status"] = "completed"
    return result, meta


def _add_usage(meta: dict, usage: dict, from_cache: bool) -> None:
    if not from_cache:
        meta["api_calls"] += 1
    meta["input_tokens"] += usage["input_tokens"]
    meta["output_tokens"] += usage["output_tokens"]
    meta["api_retry_count"] += usage["api_retry_count"]
    meta["cost_usd"] += usage["cost_usd"]


def _add_reason(meta: dict, code: str) -> None:
    if code not in meta["review_reasons"]:
        meta["review_reasons"].append(code)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": "Unsupported file type. Upload a PDF, JPG, or PNG."}), 400

    # Save to a unique path so concurrent uploads don't collide
    suffix = Path(f.filename).suffix.lower()
    unique_name = uuid.uuid4().hex + suffix
    save_path = UPLOAD_DIR / unique_name
    f.save(str(save_path))

    result, meta = _extract(save_path)

    if "_error" in result:
        return jsonify({
            "ok": False,
            "error": result["_error"],
            "file_id": unique_name,
        }), 200  # 200 so the browser JS can read the body

    return jsonify({
        "ok": True,
        "result": result,
        "meta": {
            "input_mode": meta["input_mode"],
            "model": meta["model"],
            "cost_usd": round(meta["cost_usd"], 6),
            "input_tokens": meta["input_tokens"],
            "output_tokens": meta["output_tokens"],
            "needs_review": meta["needs_review"],
            "review_reasons": meta["review_reasons"],
            "validation_errors": meta["validation_errors"],
            "correction_outcome": meta["correction_outcome"],
        },
        "file_id": unique_name,
    })


@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    """Serve the uploaded file so the browser can preview it."""
    return send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

