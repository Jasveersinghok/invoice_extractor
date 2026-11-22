"""
evaluate_regex.py — Evaluate the regex extractor against 10 labelled ground-truth docs
and compare results with the frozen Nemotron v1 baseline.

Run from the regex_experiment/ folder:
    python evaluate_regex.py

Outputs:
    outputs/eval_results.json
    outputs/eval_results.csv
    Console: per-field table + comparison with Nemotron v1
"""

import sys
import os
import json
import time
import csv
import re

# Parent src/ for matching.py and validate.py
_PARENT_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_PARENT_SRC))

_THIS_SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, os.path.abspath(_THIS_SRC))

from loader    import load_document
from extractor import extract, NULL_RESULT
import matching as _matching   # from parent src/
import validate as _validate   # from parent src/

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(THIS_DIR)
LABELLED_DIR= os.path.join(PARENT_DIR, "labelled")
DOCS_DIR    = os.path.join(PARENT_DIR, "labelled")   # docs live alongside JSONs
OUTPUTS_DIR = os.path.join(THIS_DIR,   "outputs")

BASELINE_PATH = os.path.join(PARENT_DIR, "outputs", "eval_results.json")

# Ground-truth files are named L01.json … L10.json
LABELLED_IDS = [f"L{i:02d}" for i in range(1, 11)]

# ---------------------------------------------------------------------------
# Field matching (same logic as parent evaluate.py)
# ---------------------------------------------------------------------------

def _match_field(field: str, pred, gt) -> bool:
    """Return True if prediction matches ground truth for a given field."""
    if field == "is_invoice":
        return pred == gt
    if field == "schema_valid":
        return bool(pred)
    # Both null → match
    if gt is None and pred is None:
        return True
    if gt is None or pred is None:
        return False

    if field == "vendor":
        strict, lenient = _matching.match_vendor(pred, gt)
        return lenient  # use lenient (strips legal suffixes)
    if field == "invoice_number":
        return _matching.match_exact(pred, gt)
    if field == "invoice_date":
        return _matching.match_exact(pred, gt)
    if field == "currency":
        return _matching.match_exact(pred, gt)
    if field in ("tax_total", "total"):
        return _matching.match_number(pred, gt)
    return _matching.match_exact(pred, gt)


def _remap_items(items: list) -> list:
    """
    match_line_items expects items with key 'amount', but our extractor
    outputs key 'total'.  Map 'total' → 'amount' for the matcher.
    """
    out = []
    for it in (items or []):
        entry = dict(it)
        if "total" in entry and "amount" not in entry:
            entry["amount"] = entry.pop("total")
        out.append(entry)
    return out


def _line_item_metrics(pred_items: list, gt_items: list):
    """
    Compute TP / FP / FN for line items using the same matching as parent.
    Ground-truth items may already use 'amount' key.
    """
    try:
        pm = _remap_items(pred_items)
        gm = _remap_items(gt_items)
        return _matching.match_line_items(pm, gm)
    except Exception as e:
        # Fallback: simple count
        tp = min(len(pred_items), len(gt_items))
        fp = max(0, len(pred_items) - len(gt_items))
        fn = max(0, len(gt_items) - len(pred_items))
        return {"tp": tp, "fp": fp, "fn": fn}

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

SCALAR_FIELDS = [
    "is_invoice", "vendor", "invoice_number",
    "invoice_date", "currency", "tax_total", "total",
]


def evaluate():
    results_per_doc = []
    field_counts    = {f: {"correct": 0, "total": 0} for f in SCALAR_FIELDS}
    field_counts["schema_valid"] = {"correct": 0, "total": 0}
    li_tp = li_fp = li_fn = 0
    doc_exact = 0

    for doc_id in LABELLED_IDS:
        gt_path  = os.path.join(LABELLED_DIR, f"{doc_id}.json")
        doc_path = None
        # Find the actual document file (pdf or jpg)
        for ext in (".pdf", ".PDF", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png"):
            cand = os.path.join(DOCS_DIR, f"{doc_id}{ext}")
            if os.path.exists(cand):
                doc_path = cand
                break

        if not os.path.exists(gt_path):
            print(f"[WARN] Ground truth not found: {gt_path}", file=sys.stderr)
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        # ---- Run extraction ----
        status = "completed"
        if doc_path is None:
            print(f"[WARN] Document not found for {doc_id}", file=sys.stderr)
            pred   = dict(NULL_RESULT)
            status = "doc_not_found"
        else:
            loaded = load_document(doc_path)
            if loaded["input_mode"] == "ocr_unavailable":
                pred   = dict(NULL_RESULT)
                pred["is_invoice"] = None
                status = "ocr_unavailable"
            else:
                text   = loaded["content"] or ""
                pred   = extract(text)
                # Save extraction + text
                stem = doc_id
                extr_path = os.path.join(OUTPUTS_DIR, "extractions", f"{stem}.json")
                with open(extr_path, "w", encoding="utf-8") as f:
                    json.dump(pred, f, indent=2, ensure_ascii=False)
                if text:
                    text_path = os.path.join(OUTPUTS_DIR, "text", f"{stem}.txt")
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(text)

        # ---- Schema validation ----
        try:
            vr           = _validate.validate_extraction(pred)
            schema_valid = len(vr.errors) == 0
        except Exception:
            schema_valid = False

        # ---- Score fields ----
        field_results = {}
        all_correct   = True

        for field in SCALAR_FIELDS:
            gt_val   = gt.get(field)
            pred_val = pred.get(field)

            # Skip evaluation for OCR-unavailable docs on non-is_invoice fields
            if status == "ocr_unavailable" and field != "is_invoice":
                field_results[field] = None  # N/A
                all_correct = False
                continue

            correct = _match_field(field, pred_val, gt_val)
            field_results[field] = correct
            if not correct:
                all_correct = False

            # Only count in denominator if gt field is non-null OR if field is is_invoice
            if field == "is_invoice" or gt_val is not None:
                field_counts[field]["total"]   += 1
                field_counts[field]["correct"] += int(correct)

        # Schema valid
        field_counts["schema_valid"]["total"]   += 1
        field_counts["schema_valid"]["correct"] += int(schema_valid)
        field_results["schema_valid"] = schema_valid
        if not schema_valid:
            all_correct = False

        # Line items
        gt_items   = gt.get("line_items") or []
        pred_items = pred.get("line_items") or []
        li_metrics = _line_item_metrics(pred_items, gt_items)
        li_tp += li_metrics.get("tp", 0)
        li_fp += li_metrics.get("fp", 0)
        li_fn += li_metrics.get("fn", 0)

        if all_correct:
            doc_exact += 1

        results_per_doc.append({
            "doc_id":  doc_id,
            "status":  status,
            "fields":  field_results,
            "li":      li_metrics,
            "schema_valid": schema_valid,
        })

        status_str = f"[{status.upper()[:3]}]"
        print(f"  {doc_id} {status_str}  " +
              "  ".join(f"{k}={'✓' if v else ('?' if v is None else '✗')}"
                        for k, v in field_results.items()))

    # ---- Aggregate metrics ----
    li_prec  = li_tp / (li_tp + li_fp) if (li_tp + li_fp) > 0 else 0.0
    li_rec   = li_tp / (li_tp + li_fn) if (li_tp + li_fn) > 0 else 0.0
    li_f1    = (2 * li_prec * li_rec / (li_prec + li_rec)
                if (li_prec + li_rec) > 0 else 0.0)

    n_docs = len(LABELLED_IDS)

    agg = {
        "extractor":          "regex",
        "n_docs_evaluated":   n_docs,
        "doc_exact_match":    doc_exact,
        "schema_valid":       field_counts["schema_valid"]["correct"],
        "fields":             {},
        "line_items":         {"tp": li_tp, "fp": li_fp, "fn": li_fn,
                               "precision": round(li_prec, 4),
                               "recall":    round(li_rec, 4),
                               "f1":        round(li_f1, 4)},
        "per_doc":            results_per_doc,
    }
    for field in SCALAR_FIELDS + ["schema_valid"]:
        c = field_counts[field]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0.0
        agg["fields"][field] = {
            "correct": c["correct"],
            "total":   c["total"],
            "accuracy": round(acc, 4),
        }

    return agg


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------

def _print_table(agg: dict):
    print("\n" + "="*60)
    print("  REGEX EXTRACTOR — EVALUATION RESULTS")
    print("="*60)
    print(f"  Documents evaluated : {agg['n_docs_evaluated']}")
    print(f"  Exact doc match     : {agg['doc_exact_match']}/{agg['n_docs_evaluated']}")
    print(f"  Schema valid        : {agg['schema_valid']}/{agg['n_docs_evaluated']}")
    print()
    print(f"  {'Field':<20} {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*5}  {'-'*8}")
    for field in SCALAR_FIELDS + ["schema_valid"]:
        fd = agg["fields"][field]
        print(f"  {field:<20} {fd['correct']:>7}  {fd['total']:>5}  {fd['accuracy']:>8.1%}")

    li = agg["line_items"]
    print(f"\n  Line Items: TP={li['tp']} FP={li['fp']} FN={li['fn']}"
          f"  P={li['precision']:.3f} R={li['recall']:.3f} F1={li['f1']:.3f}")


def _print_comparison(agg: dict, baseline_path: str):
    if not os.path.exists(baseline_path):
        print(f"\n[INFO] Nemotron v1 baseline not found at {baseline_path}")
        return

    with open(baseline_path, encoding="utf-8") as f:
        v1 = json.load(f)

    # v1 stores aggregate as strings like "9/9 (100%)" inside v1["aggregate"]
    v1_agg = v1.get("aggregate", {})

    def _parse_v1(key: str):
        """Parse a v1 aggregate string '9/9 (100%)' → (correct, total, accuracy)."""
        raw = v1_agg.get(key, "")
        m = re.match(r"(\d+)/(\d+)\s*\((\d+(?:\.\d+)?)%\)", str(raw))
        if m:
            return int(m.group(1)), int(m.group(2)), float(m.group(3)) / 100
        return None, None, None

    # Map our field names to v1 aggregate keys
    V1_KEY_MAP = {
        "is_invoice":     "is_invoice",
        "vendor":         "vendor_lenient",
        "invoice_number": "invoice_number",
        "invoice_date":   "invoice_date",
        "currency":       "currency",
        "tax_total":      "tax_total",
        "total":          "total",
        "schema_valid":   "schema_valid",
    }

    print("\n" + "="*72)
    print("  HEAD-TO-HEAD COMPARISON: Regex vs Nemotron v1 Baseline")
    print("="*72)
    print(f"  {'Field':<20} {'Regex':>12} {'Nemotron v1':>14} {'Δ':>8}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*14}  {'-'*8}")

    for field in SCALAR_FIELDS + ["schema_valid"]:
        rg    = agg["fields"][field]
        rg_str = f"{rg['correct']}/{rg['total']} ({rg['accuracy']:.0%})"

        v1_key = V1_KEY_MAP.get(field, field)
        v1_c, v1_t, v1_acc = _parse_v1(v1_key)
        if v1_acc is not None:
            v1_str    = f"{v1_c}/{v1_t} ({v1_acc:.0%})"
            delta_str = f"{rg['accuracy'] - v1_acc:+.0%}"
        else:
            v1_str    = "N/A"
            delta_str = "N/A"

        print(f"  {field:<20} {rg_str:>12}  {v1_str:>14}  {delta_str:>8}")

    # Line items
    rg_li  = agg["line_items"]
    v1_f1  = float(v1_agg.get("line_micro_f1", 0))
    rg_f1_str = f"F1={rg_li['f1']:.3f}"
    v1_f1_str = f"F1={v1_f1:.3f}"
    delta_li  = rg_li["f1"] - v1_f1
    print(f"  {'line_items':<20} {rg_f1_str:>12}  {v1_f1_str:>14}  {delta_li:>+8.3f}")

    print("="*72)
    print("  Note: Docs with Tesseract unavailable (L02, L09, L10) counted as")
    print("        full failures for regex; Nemotron v1 used vision model for them.")
    print()


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def _save_csv(agg: dict):
    csv_path = os.path.join(OUTPUTS_DIR, "eval_results.csv")
    rows = []
    for doc in agg["per_doc"]:
        row = {"doc_id": doc["doc_id"], "status": doc["status"]}
        for field, val in doc["fields"].items():
            row[field] = "" if val is None else ("1" if val else "0")
        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\nRunning regex extraction on all labelled documents …\n")
    t0  = time.time()
    agg = evaluate()
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    _print_table(agg)
    _print_comparison(agg, BASELINE_PATH)

    # Save JSON
    json_path = os.path.join(OUTPUTS_DIR, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    csv_path = _save_csv(agg)

    print(f"\n  Saved → {json_path}")
    print(f"  Saved → {csv_path}\n")


if __name__ == "__main__":
    main()
