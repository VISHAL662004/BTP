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


def test_assemble_matches_evaluate_and_bootstrap_paired():
    from src.evaluation.froc import assemble, bootstrap_cpm, scan_vectors, sensitivity_by_size
    rng = np.random.default_rng(0)
    ann = _ann([(f"s{i}", 0, 0, 0, 8.0 + i) for i in range(6)])
    rows = []
    for i in range(6):
        rows.append((f"s{i}", 0, 0, 0, 0.9 - 0.05 * i))
        rows += [(f"s{i}", 100 + j, 0, 0, rng.random() * 0.6) for j in range(10)]
    cand = _cand(rows)
    uids = [f"s{i}" for i in range(6)]
    vecs = scan_vectors(cand, uids, ann)
    a, b = assemble(vecs, uids), evaluate(cand, uids, ann)
    assert a.cpm == b.cpm and a.true_positives == 6
    boot = bootstrap_cpm([vecs, vecs], uids, n_boot=20, seed=1)
    assert boot.shape == (20, 2) and np.allclose(boot[:, 0], boot[:, 1])          # same model -> identical, paired resamples
    assert 0 <= boot.min() and boot.max() <= 1
    sz = sensitivity_by_size(vecs, uids, fp_rate=1.0)
    assert sum(v["nodules"] for v in sz.values()) == 6
