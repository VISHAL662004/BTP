"""Compare the 1/3/5-slice CNN experiments (Phase 6).

Decisions are based on the VALIDATION split; test numbers are reported alongside, not used for selection.
Two sources of uncertainty are shown separately:
  * seed-to-seed training noise  (mean +- std over seeds)
  * evaluation-set noise         (scan-level paired bootstrap, 95% CI)
Outputs: results/tables/phase6_comparison.{csv,json}, results/figures/phase6_*.png
"""
import json
import re
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

GROUP, PRED = Path("experiments/baseline"), Path("results/predictions")
COLORS = {1: "#64748B", 3: "#D97706", 5: "#2563EB"}


def discover():
    runs = []
    for d in sorted(GROUP.iterdir()):
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        if d.name == "EXP-001-baseline-2d":
            n, seed = 1, 42
        else:
            mm = re.match(r"EXP-002-slices(\d)-s(\d+)", d.name)
            if not mm:
                continue
            n, seed = int(mm.group(1)), int(mm.group(2))
        runs.append({"name": d.name, "n": n, "seed": seed, "metrics": m, "dir": d})
    return runs


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    runs = discover()
    ann, exc = pd.read_csv("data/annotations.csv"), pd.read_csv("data/annotations/annotations_excluded.csv")
    uids = {s: pd.read_csv(f"data/splits/{s}.csv").seriesuid.tolist() for s in ("val", "test")}
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    rows = []
    for r in runs:
        m = r["metrics"]; e = m["efficiency"]
        rows.append({"experiment": r["name"], "slices": r["n"], "seed": r["seed"], "epoch": m["epoch"],
                     "val_cpm": m["val"]["cpm"], "test_cpm": m["test"]["cpm"],
                     "val_sens@1": m["val"]["sens@1"], "test_sens@1": m["test"]["sens@1"],
                     "val_max_sens": m["val"]["max_sensitivity"],
                     "params": e["parameters"], "MFLOPs": e.get("flops_per_sample", np.nan) / 1e6,
                     "cpu_ms_b1": e["latency_cpu_batch1"]["latency_ms_mean"]})
    tab = pd.DataFrame(rows).sort_values(["slices", "seed"])
    tab.to_csv("results/tables/phase6_runs.csv", index=False)

    agg = tab.groupby("slices").agg(runs=("seed", "size"), val_cpm_mean=("val_cpm", "mean"), val_cpm_std=("val_cpm", "std"),
                                    test_cpm_mean=("test_cpm", "mean"), test_cpm_std=("test_cpm", "std"),
                                    val_sens1_mean=("val_sens@1", "mean"), test_sens1_mean=("test_sens@1", "mean"),
                                    params=("params", "first"), MFLOPs=("MFLOPs", "first"), cpu_ms_b1=("cpu_ms_b1", "mean")).reset_index()
    agg.to_csv("results/tables/phase6_by_slices.csv", index=False)
    out = {"runs": tab.to_dict("records"), "by_slices": agg.to_dict("records")}

    # per-scan FROC vectors for every run/split
    vecs = {}
    for r in Progress(runs, desc="loading predictions", unit="run", step_pct=25):
        for s in ("val", "test"):
            pred = pd.read_csv(PRED / r["name"] / f"{s}_predictions.csv")
            vecs[(r["name"], s)] = scan_vectors(pred, uids[s], ann, exc)

    # paired bootstrap: all runs share the same scan resamples; average CPM over seeds per slice count
    out["bootstrap"] = {}
    ns = sorted(tab.slices.unique())
    for s in ("val", "test"):
        names = [r["name"] for r in runs]
        boot = bootstrap_cpm([vecs[(n, s)] for n in names], uids[s], n_boot=1000, seed=0)
        by_n = {n: boot[:, [i for i, r in enumerate(runs) if r["n"] == n]].mean(1) for n in ns}
        res = {f"{n}-slice": {"cpm_mean": float(by_n[n].mean()), "ci95": ci(by_n[n])} for n in ns}
        for a, b in ((3, 1), (5, 1), (5, 3)):
            if a in by_n and b in by_n:
                d = by_n[a] - by_n[b]
                res[f"{a}-slice minus {b}-slice"] = {"delta_cpm": float(d.mean()), "ci95": ci(d), "p(delta<=0)": float((d <= 0).mean())}
        out["bootstrap"][s] = res
        if s == "val":
            boot_val = by_n

    # sensitivity by nodule size at 1 FP/scan, averaged over seeds
    out["by_size_at_1fp"] = {}
    for s in ("val", "test"):
        per = {}
        for n in ns:
            lst = [sensitivity_by_size(vecs[(r["name"], s)], uids[s]) for r in runs if r["n"] == n]
            per[f"{n}-slice"] = {b: {"nodules": lst[0][b]["nodules"], "sensitivity": float(np.mean([x[b]["sensitivity"] for x in lst]))} for b in lst[0]}
        out["by_size_at_1fp"][s] = per
    Path("results/tables/phase6_comparison.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, s in enumerate(("val", "test")):
        for r in runs:
            if r["seed"] != 42:
                continue
            res = assemble(vecs[(r["name"], s)], uids[s])
            g = np.geomspace(0.125, 8, 200)
            ax[k].plot(g, np.interp(g, res.fps, res.sens), color=COLORS[r["n"]], lw=2, label=f"{r['n']}-slice (CPM {res.cpm:.3f})")
        ax[k].set(xscale="log", xlim=(0.125, 8), ylim=(0.2, 1), title=f"FROC — {s} (seed 42)", xlabel="FP per scan", ylabel="Sensitivity")
        ax[k].set_xticks(FP_RATES); ax[k].set_xticklabels([str(x) for x in FP_RATES]); ax[k].grid(alpha=.3); ax[k].legend(loc="lower right")
    plt.tight_layout(); plt.savefig("results/figures/phase6_froc.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, s in enumerate(("val", "test")):
        for n in ns:
            t = tab[tab.slices == n]
            ax[k].scatter([n] * len(t), t[f"{s}_cpm"], color=COLORS[n], s=40, zorder=3)
            ax[k].plot([n - .25, n + .25], [t[f"{s}_cpm"].mean()] * 2, color=COLORS[n], lw=3)
        ax[k].set(xticks=ns, xlabel="input slices", ylabel="CPM", title=f"CPM by slice count — {s}\n(dots = seeds, bar = mean)")
        ax[k].grid(alpha=.3)
    labels, mids, lo, hi = [], [], [], []
    for key, v in out["bootstrap"]["val"].items():
        if "minus" in key:
            labels.append(key.replace("-slice", "")); mids.append(v["delta_cpm"]); lo.append(v["delta_cpm"] - v["ci95"][0]); hi.append(v["ci95"][1] - v["delta_cpm"])
    ax[2].errorbar(mids, range(len(labels)), xerr=[lo, hi], fmt="o", color="#0F766E", capsize=4)
    ax[2].axvline(0, color="k", lw=1); ax[2].set(yticks=range(len(labels)), yticklabels=labels, xlabel="ΔCPM (paired bootstrap, 95% CI)", title="Validation ΔCPM (seed-averaged)")
    ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("results/figures/phase6_cpm_comparison.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, s in enumerate(("val", "test")):
        bins = list(next(iter(out["by_size_at_1fp"][s].values())).keys())
        w = 0.25
        for j, n in enumerate(ns):
            v = [out["by_size_at_1fp"][s][f"{n}-slice"][b]["sensitivity"] for b in bins]
            ax[k].bar(np.arange(len(bins)) + (j - 1) * w, v, w, color=COLORS[n], label=f"{n}-slice")
        nn = [out["by_size_at_1fp"][s][f"{ns[0]}-slice"][b]["nodules"] for b in bins]
        ax[k].set_xticks(range(len(bins))); ax[k].set_xticklabels([f"{b}\n(n={c})" for b, c in zip(bins, nn)])
        ax[k].set(ylim=(0, 1), ylabel="Sensitivity @ 1 FP/scan", title=f"By nodule size — {s}"); ax[k].legend(); ax[k].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig("results/figures/phase6_by_size.png", dpi=140); plt.close()
    print(agg.round(4).to_string(index=False))
    print(json.dumps(out["bootstrap"], indent=1, default=float))


if __name__ == "__main__":
    main()
