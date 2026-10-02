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


@dataclass
class ScanVec:
    """FROC vectors of one scan (concatenated across scans to get the dataset FROC)."""
    gt: np.ndarray
    prob: np.ndarray
    excl: np.ndarray
    diam: np.ndarray        # nodule diameter (mm) for GT entries, NaN for false positives
    tp: int = 0
    fn: int = 0
    fp: int = 0
    n_cands: int = 0
    n_nod: int = 0
    ign_exc: int = 0
    ign_dbl: int = 0


def scan_vectors(candidates: pd.DataFrame, seriesuids, annotations: pd.DataFrame, excluded: pd.DataFrame = None,
                 max_marks_per_scan: int = 100, legacy: bool = False) -> dict:
    """Per-scan FROC vectors (uid -> ScanVec). Matching/ignoring rules as described in the module doc."""
    if legacy:
        max_marks_per_scan = None
    cand_g = {k: _cap_marks(g, max_marks_per_scan) for k, g in candidates.groupby("seriesuid")}
    neg_r2 = 0.25 if legacy else 25.0   # radius^2 for annotations with diameter < 0
    ann_g = {k: g for k, g in annotations.groupby("seriesuid")}
    exc_g = {k: g for k, g in excluded.groupby("seriesuid")} if excluded is not None else {}
    out = {}
    for uid in dict.fromkeys(seriesuids):
        g = cand_g.get(uid)
        if g is None:
            xyz, p = np.zeros((0, 3)), np.zeros(0)
        else:
            xyz, p = g[["coordX", "coordY", "coordZ"]].to_numpy(float), g["probability"].to_numpy(float)
        v = ScanVec([], [], [], [], n_cands=len(p))
        gt, prob, excl, diam = [], [], [], []
        remaining = np.ones(len(p), bool)
        if uid in ann_g:
            for a in ann_g[uid][["coordX", "coordY", "coordZ", "diameter_mm"]].itertuples(index=False):
                v.n_nod += 1
                r2 = (a.diameter_mm / 2.0) ** 2 if a.diameter_mm >= 0 else neg_r2
                hit = ((xyz - np.array(a[:3])) ** 2).sum(1) < r2
                diam.append(a.diameter_mm); gt.append(1.0); excl.append(not hit.any())
                if hit.any():
                    v.ign_dbl += int(hit.sum()) - 1
                    remaining &= ~hit
                    prob.append(p[hit].max()); v.tp += 1
                else:
                    prob.append(MIN_PROB); v.fn += 1
        if uid in exc_g:
            for a in exc_g[uid][["coordX", "coordY", "coordZ", "diameter_mm"]].itertuples(index=False):
                r2 = (a.diameter_mm / 2.0) ** 2 if a.diameter_mm >= 0 else neg_r2
                hit = (((xyz - np.array(a[:3])) ** 2).sum(1) < r2) & remaining
                v.ign_exc += int(hit.sum())
                remaining &= ~hit
        fp_p = p[remaining]
        v.fp = len(fp_p)
        gt += [0.0] * v.fp; prob += fp_p.tolist(); excl += [False] * v.fp; diam += [np.nan] * v.fp
        v.gt, v.prob, v.excl, v.diam = (np.array(x, float) for x in (gt, prob, excl, diam))
        v.excl = v.excl.astype(bool)
        out[uid] = v
    return out


def assemble(vecs: dict, uids) -> FrocResult:
    """Dataset-level FROC from per-scan vectors; `uids` may contain duplicates (bootstrap)."""
    uids = list(uids)
    cat = lambda f: np.concatenate([getattr(vecs[u], f) for u in uids]) if uids else np.zeros(0)
    fps, sens, thr = compute_froc_curve(cat("gt"), cat("prob"), len(uids), cat("excl"))
    tot = lambda f: sum(getattr(vecs[u], f) for u in uids)
    return FrocResult(fps, sens, thr, tot("n_nod"), tot("tp"), tot("fp"), tot("fn"), tot("n_cands"),
                      tot("ign_exc"), tot("ign_dbl"), len(uids))


def evaluate(candidates: pd.DataFrame, seriesuids, annotations: pd.DataFrame,
             excluded: pd.DataFrame = None, max_marks_per_scan: int = 100, legacy: bool = False) -> FrocResult:
    seriesuids = list(seriesuids)
    return assemble(scan_vectors(candidates, seriesuids, annotations, excluded, max_marks_per_scan, legacy), seriesuids)


def _cpm(fps, sens) -> float:
    return float(np.mean([np.interp(r, fps, sens) for r in FP_RATES]))


def bootstrap_cpm(vecs_list, uids, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Scan-level bootstrap (as in the official script): resample scans with replacement and
    recompute CPM for every model on the SAME resamples -> paired differences. Returns (n_boot, n_models)."""
    uids = np.array(list(uids))
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(vecs_list)))
    for b in range(n_boot):
        samp = uids[rng.integers(0, len(uids), len(uids))]
        for m, vecs in enumerate(vecs_list):
            r = assemble(vecs, samp)
            out[b, m] = _cpm(r.fps, r.sens)
    return out


SIZE_BINS = (("<6 mm", 0, 6), ("6-10 mm", 6, 10), (">=10 mm", 10, 1e9))


def sensitivity_by_size(vecs: dict, uids, fp_rate: float = 1.0, bins=SIZE_BINS) -> dict:
    """Sensitivity per nodule-diameter bin at the operating threshold giving `fp_rate` FP/scan overall."""
    uids = list(uids)
    res = assemble(vecs, uids)
    i = max(int(np.searchsorted(res.fps, fp_rate, side="right")) - 1, 0)
    thr = res.thresholds[i]
    gt = np.concatenate([vecs[u].gt for u in uids]); prob = np.concatenate([vecs[u].prob for u in uids])
    diam = np.concatenate([vecs[u].diam for u in uids]); ex = np.concatenate([vecs[u].excl for u in uids])
    out = {}
    for name, lo, hi in bins:
        m = (gt == 1) & (diam >= lo) & (diam < hi)
        det = m & (prob >= thr) & ~ex
        out[name] = {"nodules": int(m.sum()), "detected": int(det.sum()), "sensitivity": float(det.sum() / max(m.sum(), 1))}
    return out
