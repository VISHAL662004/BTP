"""Validate src/evaluation/froc.py against the official LUNA16 example output
(evaluationScript.zip: sampleSubmission.csv -> froc_sampleSubmission.txt, CADAnalysis.txt)."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.froc import evaluate

Z = "evaluationScript/"


def main():
    z = zipfile.ZipFile("data/evaluationScript.zip")
    rd = lambda n, **kw: pd.read_csv(io.BytesIO(z.read(Z + n)), **kw)
    sub = rd("exampleFiles/submission/sampleSubmission.csv")
    ann, exc = rd("annotations/annotations.csv"), rd("annotations/annotations_excluded.csv")
    uids = rd("annotations/seriesuids.csv", header=None)[0].tolist()
    ref = pd.read_csv(io.BytesIO(z.read(Z + "exampleFiles/evaluation/froc_sampleSubmission.txt")), header=None).to_numpy()
    res = evaluate(sub, uids, ann, exc, legacy=True)
    ok = True
    for name, mine, theirs in [("TP", res.true_positives, 1120), ("FP", res.false_positives, 548420), ("FN", res.false_negatives, 66),
                               ("candidates", res.n_candidates, 551065), ("nodules", res.n_nodules, 1186),
                               ("ignored excluded", res.ignored_excluded, 1294), ("ignored double", res.ignored_double, 231)]:
        print(f"{name:18s} mine={mine} official={theirs} {'OK' if mine == theirs else 'MISMATCH'}")
        ok &= mine == theirs
    # point counts differ slightly between scikit-learn versions (tied thresholds), so compare the
    # curves where they are used: interpolated sensitivity over the FROC operating range.
    grid = np.linspace(0.125, 8, 2000)
    mine_i, ref_i = np.interp(grid, res.fps, res.sens), np.interp(grid, ref[:, 0], ref[:, 1])
    d = float(np.abs(mine_i - ref_i).max())
    ref_cpm = float(np.mean([np.interp(r, ref[:, 0], ref[:, 1]) for r in (0.125, 0.25, 0.5, 1, 2, 4, 8)]))
    print(f"FROC curve points mine={len(res.fps)} official={len(ref)}; max |sens diff| on [1/8, 8] FP/scan = {d:.2e}")
    print(f"CPM mine={res.cpm:.6f} official-file={ref_cpm:.6f}")
    ok &= d < 1e-3 and abs(res.cpm - ref_cpm) < 1e-3
    print("CPM of the official sample submission:", round(res.cpm, 4))
    cur = evaluate(sub, uids, ann, exc)   # default = shipped-script protocol (5 mm excluded radius, 100 marks/scan)
    print("Shipped-script protocol on the same file: TP", cur.true_positives, "FP", cur.false_positives, "CPM", round(cur.cpm, 4))
    print("RESULT:", "official evaluation reproduced" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
