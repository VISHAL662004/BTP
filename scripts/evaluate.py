"""Evaluate a trained experiment on val and test (LUNA16 FROC/CPM) + efficiency.

usage: python scripts/evaluate.py experiments/baseline/EXP-001-baseline-2d [--ckpt best.pt]
The test split is evaluated ONCE here, after model selection on validation.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.patch_store import PatchStore
from src.efficiency.flops import count_flops
from src.efficiency.latency import measure_latency
from src.efficiency.parameters import count_parameters
from src.evaluation.evaluator import predict_proba, score
from src.evaluation.froc import FP_RATES
from src.models.detector import build_detector
from src.utils.device import get_device
from src.utils.progress import Progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()
    exp = Path(args.exp_dir)
    ck = torch.load(exp / args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    device = get_device(cfg["device"]["preferred"])
    model = build_detector(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    ev = cfg["evaluation"]
    ann, exc = pd.read_csv(ev["annotations"]), pd.read_csv(ev["excluded"])
    name = cfg["experiment"]["name"]
    pred_dir = Path("results/predictions") / name
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"experiment": name, "checkpoint": args.ckpt, "epoch": ck["epoch"], "seed": ck["seed"],
               "device": str(device), "protocol": f"LUNA16 FROC/CPM, max {ev['max_marks_per_scan']} marks/scan"}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    rng = np.random.default_rng(0)
    for split, color in Progress([("val", "#2563EB"), ("test", "#0F766E")], desc="evaluating splits", unit="split", step_pct=50):
        store = PatchStore(cfg["data"]["cache_dir"], split, cfg["data"]["channels"])
        uids = pd.read_csv(Path(ev["splits_dir"]) / f"{split}.csv").seriesuid.tolist()
        prob = predict_proba(model, store, device)
        res, pred = score(store, prob, uids, ann, exc, ev["max_marks_per_scan"])
        rand, _ = score(store, rng.random(len(store)), uids, ann, exc, ev["max_marks_per_scan"])
        pred.to_csv(pred_dir / f"{split}_predictions.csv", index=False)
        pd.DataFrame({"fp_per_scan": res.fps, "sensitivity": res.sens, "threshold": res.thresholds}).to_csv(exp / f"froc_{split}.csv", index=False)
        metrics[split] = {**res.summary(), "random_scores_cpm_reference": rand.cpm}
        g = np.geomspace(FP_RATES[0], FP_RATES[-1], 200)
        ax.plot(g, np.interp(g, res.fps, res.sens), color=color, lw=2, label=f"{split} (CPM {res.cpm:.3f})")
        print(split, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics[split].items()})
    ax.set(xscale="log", xlim=(0.125, 8), ylim=(0, 1), xlabel="Average false positives per scan", ylabel="Sensitivity",
           title=f"FROC — {name}")
    ax.set_xticks(FP_RATES); ax.set_xticklabels([str(r) for r in FP_RATES]); ax.xaxis.set_minor_formatter(NullFormatter()); ax.grid(alpha=.3); ax.legend(loc="lower right")
    plt.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"results/figures/{name}_froc.png", dpi=150)

    c = len(cfg["data"]["channels"] or range(5))
    metrics["efficiency"] = {**count_parameters(model), **count_flops(model, (1, c, 64, 64)),
                             "latency_cpu_batch1": measure_latency(model, (1, c, 64, 64), "cpu"),
                             "latency_cpu_batch256": measure_latency(model, (256, c, 64, 64), "cpu", runs=20)}
    if device.type == "mps":
        metrics["efficiency"]["latency_mps_batch256"] = measure_latency(model, (256, c, 64, 64), "mps", runs=20)
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["efficiency"], indent=1))


if __name__ == "__main__":
    main()
