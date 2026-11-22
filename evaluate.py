#!/usr/bin/env python3
"""
Evaluation harness.

Usage:
    python evaluate.py                    # run on labelled docs, print tables
    python evaluate.py --mode vision      # force vision mode on all docs
    python evaluate.py --no-run           # compare existing outputs to ground truth

Outputs:
    outputs/extractions/<stem>.json       saved extraction JSON per doc
    outputs/eval_results.json             raw per-doc per-field results
    outputs/eval_results.csv             CSV version
    experiments.md                        appended with baseline entry (first run)

Ground truth is loaded for comparison ONLY inside this script and is NEVER
passed to the LLM (extract.py is called as a subprocess).
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from matching import match_exact, match_line_items, match_number, match_vendor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
LABELLED_DIR = PROJECT_ROOT / "labelled"
EXTRACTIONS_DIR = PROJECT_ROOT / "outputs" / "extractions"
METADATA_DIR = PROJECT_ROOT / "outputs" / "metadata"
EVAL_JSON = PROJECT_ROOT / "outputs" / "eval_results.json"
EVAL_CSV = PROJECT_ROOT / "outputs" / "eval_results.csv"
EXPERIMENTS_MD = PROJECT_ROOT / "experiments.md"

LABELLED_DOCS = [f"L{i:02d}" for i in range(1, 11)]   # L01..L10

# Small-dataset note to prepend to every reported table
_SMALL_DATASET_NOTE = (
    "NOTE: 10 labelled docs (9 invoices, 1 non-invoice). "
    "One document = 10 percentage points. Differences of 1 doc are within noise. "
    "Numbers after prompt tuning are optimistic (tuned on these same documents)."
)


# ---------------------------------------------------------------------------
# Running the extractor
# ---------------------------------------------------------------------------

def run_extractor(stem: str, file_path: Path, mode: str | None = None) -> dict | None:
    """
    Run extract.py on a single file; return parsed JSON or None on hard failure.
    Saves the extraction to outputs/extractions/<stem>.json.
    """
    cmd = [sys.executable, "extract.py"]
    if mode:
        cmd += ["--mode", mode]
    cmd.append(str(file_path))

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"  [FAILED] {stem}: {result.stderr.strip()}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"  [PARSE ERROR] {stem}: {exc}", file=sys.stderr)
        return None

    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (EXTRACTIONS_DIR / f"{stem}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    return data


# ---------------------------------------------------------------------------
# Per-document evaluation
# ---------------------------------------------------------------------------

def evaluate_doc(stem: str, predicted: dict | None, truth: dict) -> dict:
    """
    Compare predicted extraction against ground truth for one document.
    Returns a dict of per-field results.
    'predicted' is None for hard-failed documents.
    """
    if predicted is None:
        # Hard failure: every field wrong
        return {
            "stem": stem,
            "status": "failed",
            "is_invoice": False,
            "vendor_strict": False,
            "vendor_lenient": False,
            "invoice_number": False,
            "invoice_date": False,
            "currency": False,
            "tax_total": False,
            "total": False,
            "line_items": None,  # no matching possible
            "line_tp": 0,
            "line_fp": 0,
            "line_fn": int(len(truth.get("line_items") or [])),
            "line_precision": None,
            "line_recall": None,
            "line_f1": None,
            "line_right_item_wrong_amount": 0,
            "line_whole_correct": False,
            "line_order_correct": False,
            "line_both_null": False,
            "doc_exact_match": False,
            "schema_valid": False,
            "totals_ok": False,
        }

    # --- Scalar fields ---
    is_inv_ok = predicted.get("is_invoice") == truth.get("is_invoice")

    # Vendor
    v_strict, v_lenient = match_vendor(predicted.get("vendor"), truth.get("vendor"))

    # Invoice number: exact after trim+lower
    inv_num_ok = match_exact(predicted.get("invoice_number"), truth.get("invoice_number"))

    # Date: exact on normalised value
    date_ok = match_exact(predicted.get("invoice_date"), truth.get("invoice_date"))

    # Currency: exact uppercase
    curr_ok = match_exact(predicted.get("currency"), truth.get("currency"))

    # Numeric fields: within 0.01
    tax_ok = match_number(predicted.get("tax_total"), truth.get("tax_total"))
    total_ok = match_number(predicted.get("total"), truth.get("total"))

    # --- Line items ---
    li_match = match_line_items(predicted.get("line_items"), truth.get("line_items"))

    # --- Schema validity ---
    meta_path = METADATA_DIR / f"{stem}.json"
    schema_valid = True
    totals_ok = True
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        schema_valid = not any(
            e.startswith("schema:") for e in meta.get("validation_errors", [])
        )
        totals_ok = not any(
            e.startswith("totals_mismatch") for e in meta.get("validation_errors", [])
        )

    # --- Document-level exact match ---
    # All scalar fields correct (strict vendor) + whole_list_correct
    doc_exact = (
        is_inv_ok and v_strict and inv_num_ok and date_ok and curr_ok
        and tax_ok and total_ok and li_match["whole_list_correct"]
    )

    return {
        "stem": stem,
        "status": "ok",
        "is_invoice": is_inv_ok,
        "vendor_strict": v_strict,
        "vendor_lenient": v_lenient,
        "invoice_number": inv_num_ok,
        "invoice_date": date_ok,
        "currency": curr_ok,
        "tax_total": tax_ok,
        "total": total_ok,
        "line_tp": li_match["tp"],
        "line_fp": li_match["fp"],
        "line_fn": li_match["fn"],
        "line_precision": li_match["precision"],
        "line_recall": li_match["recall"],
        "line_f1": li_match["f1"],
        "line_right_item_wrong_amount": li_match["right_item_wrong_amount"],
        "line_whole_correct": li_match["whole_list_correct"],
        "line_order_correct": li_match["order_correct"],
        "line_both_null": li_match["both_null"],
        "doc_exact_match": doc_exact,
        "schema_valid": schema_valid,
        "totals_ok": totals_ok,
        "matched_pairs_detail": li_match["matched_pairs"],
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate metrics over all evaluated documents."""
    n = len(results)                # 10 (all labelled docs)
    n_inv = sum(1 for r in results if json.loads(
        (LABELLED_DIR / f"{r['stem']}.json").read_text()
    ).get("is_invoice"))            # 9 labelled invoices

    def pct(num, den): return f"{num}/{den} ({100*num/den:.0f}%)" if den else "n/a"

    # Over all 10 docs
    is_inv_ok = sum(1 for r in results if r["is_invoice"])
    schema_ok = sum(1 for r in results if r["schema_valid"])
    totals_ok = sum(1 for r in results if r["totals_ok"])
    doc_exact = sum(1 for r in results if r["doc_exact_match"])

    # Over 9 labelled invoices (excluding the non-invoice L08)
    inv_results = [r for r in results if json.loads(
        (LABELLED_DIR / f"{r['stem']}.json").read_text()
    ).get("is_invoice")]

    def inv_pct(field):
        ok = sum(1 for r in inv_results if r[field])
        return pct(ok, len(inv_results))

    # Micro line-item pooled
    total_tp = sum(r["line_tp"] for r in inv_results)
    total_fp = sum(r["line_fp"] for r in inv_results)
    total_fn = sum(r["line_fn"] for r in inv_results)
    total_pred = total_tp + total_fp
    total_truth = total_tp + total_fn
    micro_p = total_tp / total_pred if total_pred > 0 else None
    micro_r = total_tp / total_truth if total_truth > 0 else None
    micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p and micro_r and micro_p+micro_r > 0) else None
    rwa = sum(r["line_right_item_wrong_amount"] for r in inv_results)

    return {
        "n_docs": n,
        "n_invoices": n_inv,
        "is_invoice": pct(is_inv_ok, n),
        "schema_valid": pct(schema_ok, n),
        "totals_check_ok": pct(totals_ok, n),
        "doc_exact_match": pct(doc_exact, n),
        "vendor_strict": inv_pct("vendor_strict"),
        "vendor_lenient": inv_pct("vendor_lenient"),
        "invoice_number": inv_pct("invoice_number"),
        "invoice_date": inv_pct("invoice_date"),
        "currency": inv_pct("currency"),
        "tax_total": inv_pct("tax_total"),
        "total": inv_pct("total"),
        "line_micro_tp": total_tp,
        "line_micro_fp": total_fp,
        "line_micro_fn": total_fn,
        "line_micro_precision": f"{micro_p:.3f}" if micro_p is not None else "n/a",
        "line_micro_recall": f"{micro_r:.3f}" if micro_r is not None else "n/a",
        "line_micro_f1": f"{micro_f1:.3f}" if micro_f1 is not None else "n/a",
        "right_item_wrong_amount": rwa,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_per_field_table(agg: dict) -> None:
    print("\n" + "=" * 60)
    print("PER-FIELD ACCURACY")
    print(_SMALL_DATASET_NOTE)
    print("=" * 60)
    rows = [
        ("is_invoice",     agg["is_invoice"],      "over 10 docs"),
        ("vendor (strict)", agg["vendor_strict"],   "over 9 invoices"),
        ("vendor (lenient)",agg["vendor_lenient"],  "over 9 invoices"),
        ("invoice_number",  agg["invoice_number"],  "over 9 invoices"),
        ("invoice_date",    agg["invoice_date"],    "over 9 invoices"),
        ("currency",        agg["currency"],        "over 9 invoices"),
        ("tax_total",       agg["tax_total"],       "over 9 invoices"),
        ("total",           agg["total"],           "over 9 invoices"),
        ("schema_valid",    agg["schema_valid"],    "over 10 docs"),
        ("totals_check_ok", agg["totals_check_ok"],"over 10 docs"),
        ("doc_exact_match", agg["doc_exact_match"],"over 10 docs"),
    ]
    for name, val, note in rows:
        print(f"  {name:<20} {val:<12} ({note})")
    print()
    print("LINE-ITEM METRICS (micro-pooled over labelled invoices)")
    print(f"  TP={agg['line_micro_tp']}  FP={agg['line_micro_fp']}  FN={agg['line_micro_fn']}")
    print(f"  Precision : {agg['line_micro_precision']}")
    print(f"  Recall    : {agg['line_micro_recall']}")
    print(f"  F1        : {agg['line_micro_f1']}")
    print(f"  Right-item, wrong-amount (diagnostic): {agg['right_item_wrong_amount']}")


def print_per_doc_table(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("PER-DOCUMENT RESULTS")
    print("=" * 60)
    header = f"{'Doc':<6} {'OK?':<6} {'inv':<4} {'vnd':<4} {'num':<4} {'date':<5} {'cur':<4} {'tax':<5} {'tot':<5} {'li_f1':<7} {'exact':<6}"
    print(header)
    print("-" * 60)
    for r in results:
        def b(v): return "Y" if v else "N"
        li_f1 = f"{r['line_f1']:.2f}" if r.get("line_f1") is not None else "null"
        status = r["status"]
        if status == "failed":
            print(f"{r['stem']:<6} {'FAIL':<6} {'?'*4:<4} {'?'*4:<4} {'?':<4} {'?':<5} {'?':<4} {'?':<5} {'?':<5} {'?':<7} {'N':<6}")
        else:
            print(f"{r['stem']:<6} {'ok':<6} {b(r['is_invoice']):<4} {b(r['vendor_strict']):<4} "
                  f"{b(r['invoice_number']):<4} {b(r['invoice_date']):<5} {b(r['currency']):<4} "
                  f"{b(r['tax_total']):<5} {b(r['total']):<5} {li_f1:<7} {b(r['doc_exact_match']):<6}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: list[dict], agg: dict, run_meta: dict) -> None:
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output = {"run_meta": run_meta, "aggregate": agg, "per_doc": results}
    EVAL_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    fieldnames = [
        "stem", "status", "is_invoice", "vendor_strict", "vendor_lenient",
        "invoice_number", "invoice_date", "currency", "tax_total", "total",
        "line_tp", "line_fp", "line_fn", "line_f1", "line_whole_correct",
        "line_order_correct", "doc_exact_match", "schema_valid", "totals_ok",
    ]
    with open(EVAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\nSaved: {EVAL_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {EVAL_CSV.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Experiment log
# ---------------------------------------------------------------------------

def append_experiment_log(run_meta: dict, agg: dict, results: list[dict]) -> None:
    """Append an entry to experiments.md. Never reconstructed from memory."""
    is_first = not EXPERIMENTS_MD.exists() or "## BASELINE" not in EXPERIMENTS_MD.read_text(encoding="utf-8")
    label = "## BASELINE" if is_first else f"## Run: {run_meta.get('label', 'unlabelled')}"

    lines = [
        f"\n{label}",
        f"Date: {run_meta['timestamp']}",
        f"Prompt version: {run_meta['prompt_version']}",
        f"Mode override: {run_meta.get('mode_override', 'none (auto-detect)')}",
        "",
        "### Per-field accuracy",
        f"is_invoice    : {agg['is_invoice']}",
        f"vendor strict : {agg['vendor_strict']}",
        f"vendor lenient: {agg['vendor_lenient']}",
        f"invoice_number: {agg['invoice_number']}",
        f"invoice_date  : {agg['invoice_date']}",
        f"currency      : {agg['currency']}",
        f"tax_total     : {agg['tax_total']}",
        f"total         : {agg['total']}",
        f"doc_exact_match: {agg['doc_exact_match']}",
        "",
        "### Line-item metrics (micro)",
        f"TP={agg['line_micro_tp']} FP={agg['line_micro_fp']} FN={agg['line_micro_fn']}",
        f"P={agg['line_micro_precision']} R={agg['line_micro_recall']} F1={agg['line_micro_f1']}",
        f"Right-item wrong-amount: {agg['right_item_wrong_amount']}",
        "",
        "### Per-document",
    ]
    for r in results:
        lines.append(
            f"  {r['stem']}: status={r['status']} "
            f"vendor={r.get('vendor_strict','?')} date={r.get('invoice_date','?')} "
            f"total={r.get('total','?')} li_f1={r.get('line_f1','?')}"
        )

    with open(EXPERIMENTS_MD, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended to {EXPERIMENTS_MD.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate invoice extractor on labelled docs.")
    parser.add_argument("--mode", choices=["text", "vision"], default=None,
                        help="Force input mode for all labelled docs.")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip extraction; compare existing outputs/extractions/ files.")
    parser.add_argument("--label", default=None,
                        help="Label for this run in experiments.md (e.g. 'prompt_v2').")
    args = parser.parse_args()

    from prompts import PROMPT_VERSION
    from llm import MODEL

    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "mode_override": args.mode or "none (auto-detect)",
        "label": args.label or ("BASELINE" if not EXPERIMENTS_MD.exists() else args.label or "run"),
    }

    print(f"\nEvaluating {len(LABELLED_DOCS)} labelled documents")
    print(f"Prompt: {PROMPT_VERSION}  Model: {MODEL}  Mode: {run_meta['mode_override']}")
    print(_SMALL_DATASET_NOTE)

    results = []
    for stem in LABELLED_DOCS:
        # Find the file (PDF or image)
        file_path = None
        for ext in [".pdf", ".jpg", ".jpeg", ".png"]:
            candidate = LABELLED_DIR / f"{stem}{ext}"
            if candidate.exists():
                file_path = candidate
                break
        if file_path is None:
            print(f"  [MISSING] {stem}: no file found", file=sys.stderr)
            continue

        truth_path = LABELLED_DIR / f"{stem}.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))

        if args.no_run:
            ext_path = EXTRACTIONS_DIR / f"{stem}.json"
            if ext_path.exists():
                predicted = json.loads(ext_path.read_text(encoding="utf-8-sig"))
            else:
                predicted = None
        else:
            print(f"  Running {stem} ({file_path.suffix})...")
            predicted = run_extractor(stem, file_path, mode=args.mode)

        doc_result = evaluate_doc(stem, predicted, truth)
        results.append(doc_result)

    agg = compute_aggregate(results)

    print_per_field_table(agg)
    print_per_doc_table(results)
    save_results(results, agg, run_meta)
    append_experiment_log(run_meta, agg, results)


if __name__ == "__main__":
    main()
