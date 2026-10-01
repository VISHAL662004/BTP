import numpy as np
import pandas as pd

from src.evaluation.froc import FP_RATES, evaluate


def _ann(rows):
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])


def _cand(rows):
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"])


def test_perfect_detector_cpm_one():
    ann = _ann([("a", 0, 0, 0, 10), ("b", 5, 5, 5, 6)])
    cand = _cand([("a", 1, 0, 0, 0.9), ("a", 50, 0, 0, 0.1), ("b", 5, 5, 6, 0.8)])
    r = evaluate(cand, ["a", "b"], ann)
    assert r.true_positives == 2 and r.false_negatives == 0 and r.false_positives == 1
    assert abs(r.cpm - 1.0) < 1e-9   # the FP has lower score than all TPs


def test_missed_nodule_counts_in_denominator():
    ann = _ann([("a", 0, 0, 0, 10), ("a", 100, 0, 0, 10)])
    cand = _cand([("a", 0, 0, 0, 0.9)])
    r = evaluate(cand, ["a"], ann)
    assert r.n_nodules == 2 and r.false_negatives == 1 and r.max_sensitivity == 0.5
    assert max(r.sens) <= 0.5 + 1e-9


def test_double_detections_ignored_and_not_false_positives():
    ann = _ann([("a", 0, 0, 0, 10)])
    cand = _cand([("a", 0, 0, 0, 0.9), ("a", 1, 1, 0, 0.8), ("a", 2, 0, 0, 0.7)])
    r = evaluate(cand, ["a"], ann)
    assert r.true_positives == 1 and r.false_positives == 0 and r.ignored_double == 2


def test_excluded_findings_ignored():
    ann = _ann([("a", 0, 0, 0, 10)])
    exc = _ann([("a", 100, 0, 0, -1)])
    cand = _cand([("a", 0, 0, 0, 0.9), ("a", 101, 0, 0, 0.99), ("a", 200, 0, 0, 0.5)])
    r = evaluate(cand, ["a"], ann, exc)
    assert r.ignored_excluded == 1 and r.false_positives == 1


def test_mark_cap_keeps_top_k():
    ann = _ann([("a", 0, 0, 0, 10)])
    cand = _cand([("a", 500 + i, 0, 0, 0.5 - i * 1e-3) for i in range(5)] + [("a", 0, 0, 0, 0.1)])
    capped = evaluate(cand, ["a"], ann, max_marks_per_scan=3)
    assert capped.true_positives == 0 and capped.n_candidates == 3   # the TP mark is outside the top 3
    assert evaluate(cand, ["a"], ann, max_marks_per_scan=100).true_positives == 1


def test_fp_per_scan_normalisation():
    ann = _ann([("a", 0, 0, 0, 10)])
    cand = _cand([("a", 0, 0, 0, 0.5)] + [("a", 100 + i, 0, 0, 0.9) for i in range(4)])
    r = evaluate(cand, ["a", "b"], ann)       # 2 scans, 4 FPs scored above the TP
    assert abs(r.fps.max() - 4 / 2) < 1e-9
    assert len(FP_RATES) == 7
