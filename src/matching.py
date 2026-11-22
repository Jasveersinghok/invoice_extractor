"""
Line-item matching and evaluation metrics.

Matching algorithm (fixed before the baseline run — do not change afterwards):
  1. Normalize descriptions: lowercase, strip, collapse whitespace.
  2. Similarity = difflib.SequenceMatcher ratio on normalized descriptions.
  3. Eligible pair: amounts equal within 0.01 AND similarity >= 0.6.
  4. Build all eligible pairs, sort by similarity desc (ties: smaller index distance).
  5. Greedy one-to-one assignment.
  6. TP = matched pairs; FP = unmatched predicted; FN = unmatched truth.
  7. Precision = TP/predicted; Recall = TP/truth; F1 as usual.
  8. Diagnostic: "right item, wrong amount" = pairs with similarity >= 0.6 but
     amount differs > 0.01 that were NOT matched.
  9. whole_list_correct: same length, same order, all desc/qty/amount exact.
     order_correct: matched pairs appear in increasing index order on both sides.
 10. Both null → correct. One null and other non-null → all missing or all extra.
"""

import difflib
import re
from decimal import Decimal

AMOUNT_TOLERANCE = Decimal("0.01")
SIM_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_desc(text: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _amounts_equal(a, b) -> bool:
    try:
        return abs(Decimal(str(a)) - Decimal(str(b))) <= AMOUNT_TOLERANCE
    except Exception:
        return False


def _qty_equal(a, b) -> bool:
    """Both null counts as equal; both numeric compared with Decimal."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(Decimal(str(a)) - Decimal(str(b))) <= AMOUNT_TOLERANCE
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------

def match_line_items(predicted: list | None, truth: list | None) -> dict:
    """
    Match predicted line items against ground-truth line items.

    Returns a dict with keys:
        tp, fp, fn                  int counts
        precision, recall, f1       float (None when denominator is 0)
        matched_pairs               list of {pred_idx, truth_idx, sim,
                                             qty_match, desc_exact_match}
        right_item_wrong_amount     int (diagnostic only)
        whole_list_correct          bool
        order_correct               bool
        both_null                   bool
    """
    # Both null → correct
    if predicted is None and truth is None:
        return _empty_result(both_null=True, whole_list_correct=True, order_correct=True)

    # One null → all extra or all missing
    if predicted is None:
        fn = len(truth)
        return _empty_result(fn=fn, both_null=False, whole_list_correct=False, order_correct=False)
    if truth is None:
        fp = len(predicted)
        return _empty_result(fp=fp, both_null=False, whole_list_correct=False, order_correct=False)

    n_pred = len(predicted)
    n_truth = len(truth)

    # Precompute normalised descriptions
    pred_norms = [_norm_desc(item["description"]) for item in predicted]
    truth_norms = [_norm_desc(item["description"]) for item in truth]

    # Build eligible pairs
    pairs = []
    for pi, pitem in enumerate(predicted):
        for ti, titem in enumerate(truth):
            if _amounts_equal(pitem["amount"], titem["amount"]):
                sim = difflib.SequenceMatcher(None, pred_norms[pi], truth_norms[ti]).ratio()
                if sim >= SIM_THRESHOLD:
                    pairs.append((sim, pi, ti))

    # Sort: similarity desc, then smaller index distance first
    pairs.sort(key=lambda x: (-x[0], abs(x[1] - x[2])))

    # Greedy one-to-one assignment
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matched_pairs = []

    for sim, pi, ti in pairs:
        if pi in used_pred or ti in used_truth:
            continue
        used_pred.add(pi)
        used_truth.add(ti)
        matched_pairs.append({
            "pred_idx": pi,
            "truth_idx": ti,
            "sim": round(sim, 4),
            "qty_match": _qty_equal(predicted[pi].get("quantity"), titem.get("quantity")),
            "desc_exact_match": pred_norms[pi] == truth_norms[ti],
        })
        # find the titem that matches ti
        matched_pairs[-1]["qty_match"] = _qty_equal(
            predicted[pi].get("quantity"), truth[ti].get("quantity")
        )

    tp = len(matched_pairs)
    fp = n_pred - tp
    fn = n_truth - tp

    precision = tp / n_pred if n_pred > 0 else None
    recall = tp / n_truth if n_truth > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    # Diagnostic: eligible pairs that had similar desc but diff amount
    right_item_wrong_amount = 0
    matched_pred = {p["pred_idx"] for p in matched_pairs}
    matched_truth = {p["truth_idx"] for p in matched_pairs}
    for pi, pitem in enumerate(predicted):
        if pi in matched_pred:
            continue
        for ti, titem in enumerate(truth):
            if ti in matched_truth:
                continue
            sim = difflib.SequenceMatcher(None, pred_norms[pi], truth_norms[ti]).ratio()
            if sim >= SIM_THRESHOLD and not _amounts_equal(pitem["amount"], titem["amount"]):
                right_item_wrong_amount += 1

    # whole_list_correct
    whole = False
    if n_pred == n_truth and tp == n_pred:
        # Check all in order and exact match
        order_pairs = sorted(matched_pairs, key=lambda x: x["pred_idx"])
        whole = all(
            p["desc_exact_match"] and p["qty_match"]
            for p in order_pairs
        )
        # Also verify index order is preserved
        if whole:
            truth_indices = [p["truth_idx"] for p in order_pairs]
            whole = truth_indices == sorted(truth_indices) == list(range(n_truth))

    # order_correct: matched pairs in increasing index order on both sides
    order_correct = False
    if matched_pairs:
        pred_idxs = [p["pred_idx"] for p in sorted(matched_pairs, key=lambda x: x["pred_idx"])]
        truth_idxs = [p["truth_idx"] for p in sorted(matched_pairs, key=lambda x: x["pred_idx"])]
        order_correct = pred_idxs == sorted(pred_idxs) and truth_idxs == sorted(truth_idxs)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_pairs": matched_pairs,
        "right_item_wrong_amount": right_item_wrong_amount,
        "whole_list_correct": whole,
        "order_correct": order_correct,
        "both_null": False,
    }


def _empty_result(tp=0, fp=0, fn=0, both_null=False,
                  whole_list_correct=False, order_correct=False) -> dict:
    n_pred = tp + fp
    n_truth = tp + fn
    precision = tp / n_pred if n_pred > 0 else None
    recall = tp / n_truth if n_truth > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_pairs": [],
        "right_item_wrong_amount": 0,
        "whole_list_correct": whole_list_correct,
        "order_correct": order_correct,
        "both_null": both_null,
    }


# ---------------------------------------------------------------------------
# Scalar field matching helpers (used by evaluate.py)
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = re.compile(
    r"\b(pvt|private|ltd|limited|llp|llc|inc|corp|co|gmbh)\.?\s*$",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]")


def match_vendor(pred: str | None, truth: str | None) -> tuple[bool, bool]:
    """
    Returns (strict_match, lenient_match).
    strict : lowercase + trim + collapse whitespace
    lenient: additionally strip punctuation and trailing legal suffixes
    """
    if pred is None and truth is None:
        return True, True
    if pred is None or truth is None:
        return False, False

    def strict_norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def lenient_norm(s: str) -> str:
        s = strict_norm(s)
        s = _LEGAL_SUFFIXES.sub("", s).strip()
        s = _PUNCT.sub("", s)
        return re.sub(r"\s+", " ", s).strip()

    return (
        strict_norm(pred) == strict_norm(truth),
        lenient_norm(pred) == lenient_norm(truth),
    )


def match_number(pred, truth) -> bool:
    """Match numeric fields within AMOUNT_TOLERANCE. Both null = correct."""
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    return _amounts_equal(pred, truth)


def match_exact(pred, truth) -> bool:
    """Exact match after strip + lowercase. Both null = correct."""
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    return str(pred).strip().lower() == str(truth).strip().lower()

