"""LUNA16 FROC / CPM evaluation.

Python 3 port of the official `noduleCADEvaluationLUNA16.py` (evaluationScript.zip), validated
against the official example output (see tests/test_evaluation.py and
scripts/validate_evaluation.py). Protocol:
  * a nodule is detected when a candidate lies within the nodule radius (Euclidean, world mm);
  * extra candidates on an already-detected nodule are ignored (not false positives);
  * candidates on "excluded" (irrelevant) findings are ignored (radius 5 mm: diameter -1 -> 10);
  * nodules without any candidate stay in the sensitivity denominator;
  * at most `max_marks_per_scan` (100, as in the shipped script) highest-probability marks per
    scan are kept; the rest are discarded entirely (neither TP nor FP);
  * FP rate = false positives / number of scans; CPM = mean sensitivity at
    FP/scan in {1/8, 1/4, 1/2, 1, 2, 4, 8}.

`legacy=True` reproduces the older script that generated the example files bundled in the zip
(excluded findings with diameter -1 get radius 0.5 mm, no mark cap); used only for validation.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import metrics as skl_metrics

FP_RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)
MIN_PROB = -1e9


@dataclass
class FrocResult:
    fps: np.ndarray
    sens: np.ndarray
    thresholds: np.ndarray
    n_nodules: int
    true_positives: int
    false_positives: int
    false_negatives: int
    n_candidates: int
    ignored_excluded: int
    ignored_double: int
    n_scans: int

    @property
    def sensitivity_at(self) -> dict:
        return {r: float(np.interp(r, self.fps, self.sens)) for r in FP_RATES}

    @property
    def cpm(self) -> float:
        return float(np.mean(list(self.sensitivity_at.values())))

    @property
    def max_sensitivity(self) -> float:
        return self.true_positives / self.n_nodules if self.n_nodules else 0.0

    def summary(self) -> dict:
        d = {"cpm": self.cpm, "max_sensitivity": self.max_sensitivity, "n_scans": self.n_scans,
             "n_nodules": self.n_nodules, "n_candidates": self.n_candidates,
             "true_positives": self.true_positives, "false_negatives": self.false_negatives,
             "false_positives": self.false_positives, "ignored_excluded": self.ignored_excluded,
             "ignored_double": self.ignored_double}
        d.update({f"sens@{r}": v for r, v in self.sensitivity_at.items()})
        return d


def compute_froc_curve(gt, prob, n_images, exclude):
    gt, prob, exclude = np.asarray(gt, float), np.asarray(prob, float), np.asarray(exclude, bool)
    keep = ~exclude
    gt_l, prob_l = gt[keep], prob[keep]
    n_detected = gt_l.sum()
    n_lesions = gt.sum()
    fpr, tpr, thr = skl_metrics.roc_curve(gt_l, prob_l, drop_intermediate=False)
    if n_lesions == len(gt):  # no false positives at all
        fps = np.zeros(len(fpr))
    else:
        fps = fpr * (len(prob_l) - n_detected) / n_images
    sens = tpr * n_detected / n_lesions
    return fps, sens, thr


def _cap_marks(g: pd.DataFrame, k: int) -> pd.DataFrame:
    return g if k is None or len(g) <= k else g.sort_values("probability", ascending=False, kind="stable").iloc[:k]


def evaluate(candidates: pd.DataFrame, seriesuids, annotations: pd.DataFrame,
             excluded: pd.DataFrame = None, max_marks_per_scan: int = 100, legacy: bool = False) -> FrocResult:
    """candidates: seriesuid, coordX, coordY, coordZ, probability. annotations: LUNA16
    annotations.csv rows (diameter_mm). excluded: annotations_excluded.csv (optional)."""
    seriesuids = list(seriesuids)
    if legacy:
        max_marks_per_scan = None
    cand_g = {k: _cap_marks(g, max_marks_per_scan) for k, g in candidates.groupby("seriesuid")}
    neg_r2 = 0.25 if legacy else 25.0   # radius^2 for annotations with diameter < 0
    ann_g = {k: g for k, g in annotations.groupby("seriesuid")}
    exc_g = {k: g for k, g in excluded.groupby("seriesuid")} if excluded is not None else {}
    gt, prob, excl = [], [], []
    tp = fn = fp = n_cands = n_nod = ign_exc = ign_dbl = 0
    for uid in seriesuids:
        g = cand_g.get(uid)
        if g is None:
            xyz, p = np.zeros((0, 3)), np.zeros(0)
        else:
            xyz, p = g[["coordX", "coordY", "coordZ"]].to_numpy(float), g["probability"].to_numpy(float)
        n_cands += len(p)
        remaining = np.ones(len(p), bool)
        if uid in ann_g:
            for a in ann_g[uid][["coordX", "coordY", "coordZ", "diameter_mm"]].itertuples(index=False):
                n_nod += 1
                r2 = (a.diameter_mm / 2.0) ** 2 if a.diameter_mm >= 0 else neg_r2
                hit = ((xyz - np.array(a[:3])) ** 2).sum(1) < r2
                if hit.any():
                    ign_dbl += int(hit.sum()) - 1
                    remaining &= ~hit
                    gt.append(1.0); prob.append(p[hit].max()); excl.append(False); tp += 1
                else:
                    gt.append(1.0); prob.append(MIN_PROB); excl.append(True); fn += 1
        if uid in exc_g:
            for a in exc_g[uid][["coordX", "coordY", "coordZ", "diameter_mm"]].itertuples(index=False):
                r2 = (a.diameter_mm / 2.0) ** 2 if a.diameter_mm >= 0 else neg_r2
                hit = (((xyz - np.array(a[:3])) ** 2).sum(1) < r2) & remaining
                ign_exc += int(hit.sum())
                remaining &= ~hit
        fp_p = p[remaining]
        fp += len(fp_p)
        gt.extend([0.0] * len(fp_p)); prob.extend(fp_p.tolist()); excl.extend([False] * len(fp_p))
    fps, sens, thr = compute_froc_curve(gt, prob, len(seriesuids), excl)
    return FrocResult(fps, sens, thr, n_nod, tp, fp, fn, n_cands, ign_exc, ign_dbl, len(seriesuids))
