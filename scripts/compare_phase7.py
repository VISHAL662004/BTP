"""Phase 7 comparison: 2.5D CNN (5 slices) vs CNN+Transformer vs capacity-matched conv control.

Decisions use the VALIDATION split; test is reported alongside. Seed noise = mean +- std over 3 seeds;
evaluation-set noise = paired scan-level bootstrap on identical resamples.
Outputs: results/tables/phase7_*.{csv,json}, results/figures/phase7_*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.froc import FP_RATES, assemble, bootstrap_cpm, scan_vectors, sensitivity_by_size
from src.utils.progress import Progress

MODELS = {  # label -> (experiment dir glob parent, name prefix)
    "CNN (5 slices)": ("experiments/baseline", "EXP-002-slices5-s"),
    "CNN + Transformer": ("experiments/cnn_transformer", "EXP-003-cnn-transformer-s"),
    "CNN + extra conv (control)": ("experiments/cnn_transformer", "EXP-003c-cnn-extraconv-s"),
}
COLORS = {"CNN (5 slices)": "#2563EB", "CNN + Transformer": "#DC2626", "CNN + extra conv (control)": "#64748B"}
SEEDS = (42, 43, 44)
PRED = Path("results/predictions")


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    ann, exc = pd.read_csv("data/annotations.csv"), pd.read_csv("data/annotations/annotations_excluded.csv")
    uids = {s: pd.read_csv(f"data/splits/{s}.csv").seriesuid.tolist() for s in ("val", "test")}
    runs = []
    for label, (parent, prefix) in MODELS.items():
        for seed in SEEDS:
            d = Path(parent) / f"{prefix}{seed}"
            if (d / "metrics.json").exists():
                runs.append({"label": label, "seed": seed, "name": d.name, "dir": d, "m": json.loads((d / "metrics.json").read_text())})
    labels = [l for l in MODELS if any(r["label"] == l for r in runs)]
    rows = [{"model": r["label"], "experiment": r["name"], "seed": r["seed"], "best_epoch": r["m"]["epoch"],
             "val_cpm": r["m"]["val"]["cpm"], "test_cpm": r["m"]["test"]["cpm"], "val_sens@1": r["m"]["val"]["sens@1"],
             "test_sens@1": r["m"]["test"]["sens@1"], "params": r["m"]["efficiency"]["parameters"],
             "MFLOPs": r["m"]["efficiency"]["flops_per_sample"] / 1e6,
             "cpu_ms_b256": r["m"]["efficiency"]["latency_cpu_batch256"]["latency_ms_mean"],
             "mps_ms_b256": r["m"]["efficiency"].get("latency_mps_batch256", {}).get("latency_ms_mean", np.nan)} for r in runs]
    tab = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True); Path("results/figures").mkdir(parents=True, exist_ok=True)
    tab.to_csv("results/tables/phase7_runs.csv", index=False)
    agg = tab.groupby("model", sort=False).agg(runs=("seed", "size"), val_cpm_mean=("val_cpm", "mean"), val_cpm_std=("val_cpm", "std"),
                                               test_cpm_mean=("test_cpm", "mean"), test_cpm_std=("test_cpm", "std"),
                                               val_sens1=("val_sens@1", "mean"), test_sens1=("test_sens@1", "mean"),
                                               params=("params", "first"), MFLOPs=("MFLOPs", "first"),
                                               cpu_ms_b256=("cpu_ms_b256", "mean"), mps_ms_b256=("mps_ms_b256", "mean")).reset_index()
    agg.to_csv("results/tables/phase7_summary.csv", index=False)
    out = {"summary": agg.to_dict("records"), "runs": tab.to_dict("records")}

    vecs = {}
    for r in Progress(runs, desc="loading predictions", unit="run", step_pct=25):
        for s in ("val", "test"):
            vecs[(r["name"], s)] = scan_vectors(pd.read_csv(PRED / r["name"] / f"{s}_predictions.csv"), uids[s], ann, exc)

    out["bootstrap"], seed_avg = {}, {}
    for s in ("val", "test"):
        boot = bootstrap_cpm([vecs[(r["name"], s)] for r in runs], uids[s], n_boot=1000, seed=0)
        by = {l: boot[:, [i for i, r in enumerate(runs) if r["label"] == l]].mean(1) for l in labels}
        seed_avg[s] = by
        res = {l: {"cpm_mean": float(by[l].mean()), "ci95": ci(by[l])} for l in labels}
        for a, b in (("CNN + Transformer", "CNN (5 slices)"), ("CNN + Transformer", "CNN + extra conv (control)"),
                     ("CNN + extra conv (control)", "CNN (5 slices)")):
            if a in by and b in by:
                d = by[a] - by[b]
                res[f"{a} minus {b}"] = {"delta_cpm": float(d.mean()), "ci95": ci(d), "p(delta<=0)": float((d <= 0).mean())}
        out["bootstrap"][s] = res

    out["by_size_at_1fp"] = {}
    for s in ("val", "test"):
        per = {}
        for l in labels:
            lst = [sensitivity_by_size(vecs[(r["name"], s)], uids[s]) for r in runs if r["label"] == l]
            per[l] = {b: {"nodules": lst[0][b]["nodules"], "sensitivity": float(np.mean([x[b]["sensitivity"] for x in lst]))} for b in lst[0]}
        out["by_size_at_1fp"][s] = per
    Path("results/tables/phase7_comparison.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for k, s in enumerate(("val", "test")):
        for j, l in enumerate(labels):
            t = tab[tab.model == l][f"{s}_cpm"]
            ax[k].scatter([j] * len(t), t, color=COLORS[l], s=45, zorder=3)
            ax[k].plot([j - .25, j + .25], [t.mean()] * 2, color=COLORS[l], lw=3)
        ax[k].set(xticks=range(len(labels)), xticklabels=[l.replace(" (", "\n(").replace(" + ", "\n+ ") for l in labels], ylabel="CPM",
                  title=f"CPM — {s} (dots = seeds, bar = mean)"); ax[k].grid(alpha=.3)
    names, mids, lo, hi = [], [], [], []
    for key, v in out["bootstrap"]["val"].items():
        if " minus " in key:
            a, b = key.split(" minus ")
            names.append(f"{a.replace('CNN + ', '')}\n− {b.replace('CNN + ', '').replace(' (control)', '')}")
            mids.append(v["delta_cpm"]); lo.append(v["delta_cpm"] - v["ci95"][0]); hi.append(v["ci95"][1] - v["delta_cpm"])
    ax[2].errorbar(mids, range(len(names)), xerr=[lo, hi], fmt="o", color="#0F766E", capsize=4)
    ax[2].axvline(0, color="k", lw=1); ax[2].set(yticks=range(len(names)), yticklabels=names, xlabel="ΔCPM, paired bootstrap 95% CI", title="Validation ΔCPM (seed-averaged)")
    ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("results/figures/phase7_cpm_comparison.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for r in runs:
        h = pd.read_csv(r["dir"] / "history.csv")
        ax[0].plot(h.epoch, h.val_cpm, color=COLORS[r["label"]], alpha=.8, lw=1.4)
        ax[1].plot(h.epoch, h.train_loss, color=COLORS[r["label"]], alpha=.8, lw=1.4)
    for l in labels:
        ax[0].plot([], [], color=COLORS[l], label=l)
    ax[0].set(xlabel="epoch", ylabel="validation CPM", title="Validation CPM during training"); ax[0].legend(fontsize=8)
    ax[1].set(xlabel="epoch", ylabel="train loss", title="Training loss"); [a.grid(alpha=.3) for a in ax]
    plt.tight_layout(); plt.savefig("results/figures/phase7_training_curves.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for k, s in enumerate(("val", "test")):
        bins = list(next(iter(out["by_size_at_1fp"][s].values())).keys()); w = 0.26
        for j, l in enumerate(labels):
            v = [out["by_size_at_1fp"][s][l][b]["sensitivity"] for b in bins]
            ax[k].bar(np.arange(len(bins)) + (j - 1) * w, v, w, color=COLORS[l], label=l)
        nn = [out["by_size_at_1fp"][s][labels[0]][b]["nodules"] for b in bins]
        ax[k].set_xticks(range(len(bins))); ax[k].set_xticklabels([f"{b}\n(n={c})" for b, c in zip(bins, nn)])
        ax[k].set(ylim=(0, 1), ylabel="Sensitivity @ 1 FP/scan", title=f"By nodule size — {s}"); ax[k].grid(alpha=.3, axis="y")
    ax[0].legend(fontsize=7, loc="lower left")
    plt.tight_layout(); plt.savefig("results/figures/phase7_by_size.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, s in enumerate(("val", "test")):
        g = np.geomspace(0.125, 8, 200)
        for l in labels:
            curves = [np.interp(g, *(lambda x: (x.fps, x.sens))(assemble(vecs[(r["name"], s)], uids[s]))) for r in runs if r["label"] == l]
            ax[k].plot(g, np.mean(curves, 0), color=COLORS[l], lw=2, label=l)
        ax[k].set(xscale="log", xlim=(0.125, 8), ylim=(0.3, 1), xlabel="FP per scan", ylabel="Sensitivity", title=f"FROC — {s} (seed-mean)")
        ax[k].set_xticks(FP_RATES); ax[k].set_xticklabels([str(x) for x in FP_RATES]); ax[k].grid(alpha=.3)
    ax[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig("results/figures/phase7_froc.png", dpi=140); plt.close()
    print(agg.round(4).to_string(index=False)); print(json.dumps(out["bootstrap"], indent=1, default=float))


if __name__ == "__main__":
    main()
