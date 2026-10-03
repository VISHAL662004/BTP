"""Phase 8 comparison: attention (SE, CBAM) vs the 2.5D CNN reference (CNN+Transformer shown for context).

Decisions use VALIDATION; test is reported alongside. Seed noise = std over 3 seeds; evaluation-set noise =
paired scan-level bootstrap. Also: efficiency table for ALL models (params, FLOPs, activation memory, latency
re-measured with interleaved rounds so drift averages out) and attention-map interpretability.
Outputs: results/tables/phase8_*.{csv,json}, results/figures/phase8_*.png
Run it when the machine is otherwise idle (latency).
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.patch_store import PatchStore
from src.efficiency.flops import count_flops
from src.efficiency.latency import measure_latency
from src.efficiency.memory import activation_memory
from src.efficiency.parameters import count_parameters
from src.evaluation.froc import FP_RATES, assemble, bootstrap_cpm, scan_vectors, sensitivity_by_size
from src.models.detector import build_detector
from src.utils.config import load_experiment_config
from src.utils.device import get_device
from src.utils.progress import Progress

MODELS = {  # label -> (parent dir, prefix, config used for the architecture)
    "CNN (5 slices)": ("experiments/baseline", "EXP-002-slices5-s", "baseline"),
    "CNN + SE": ("experiments/attention", "EXP-004-attention-se-s", "attention_se"),
    "CNN + CBAM": ("experiments/attention", "EXP-004-attention-cbam-s", "attention_cbam"),
    "CNN + Transformer (ref.)": ("experiments/cnn_transformer", "EXP-003-cnn-transformer-s", "cnn_transformer"),
}
COLORS = {"CNN (5 slices)": "#2563EB", "CNN + SE": "#D97706", "CNN + CBAM": "#0F766E", "CNN + Transformer (ref.)": "#94A3B8"}
SEEDS = (42, 43, 44)
PRED, FIG, TAB = Path("results/predictions"), Path("results/figures"), Path("results/tables")


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def arch_cfg(cfg_name):
    c = load_experiment_config(f"configs/experiments/{cfg_name}.yaml")
    c["model"]["in_channels"] = 5
    return c


def efficiency_table():
    rows, models = [], {}
    for label, (_, _, cfg_name) in MODELS.items():
        m = build_detector(arch_cfg(cfg_name)).eval()
        models[label] = m
        r = {"model": label, **count_parameters(m), **count_flops(m, (1, 5, 64, 64)),
             **activation_memory(m, (1, 5, 64, 64))}
        rows.append(r)
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    lat = {(l, k): [] for l in models for k in ("cpu_b1", "cpu_b256", "mps_b256")}
    for rnd in Progress(range(3), desc="latency rounds", unit="round", step_pct=34):   # interleave to average out drift
        for label, m in models.items():
            lat[(label, "cpu_b1")].append(measure_latency(m, (1, 5, 64, 64), "cpu", warmup=20, runs=100)["latency_ms_mean"])
            lat[(label, "cpu_b256")].append(measure_latency(m, (256, 5, 64, 64), "cpu", warmup=3, runs=10)["latency_ms_mean"])
            if "mps" in devs:
                lat[(label, "mps_b256")].append(measure_latency(m, (256, 5, 64, 64), "mps", warmup=10, runs=40)["latency_ms_mean"])
    for r in rows:
        for k in ("cpu_b1", "cpu_b256", "mps_b256"):
            v = lat[(r["model"], k)]
            r[f"latency_ms_{k}"] = float(np.median(v)) if v else np.nan
    return pd.DataFrame(rows)


def attention_analysis(ann_by_run):
    """CBAM seed-42 model on validation positives (with boxes) and random negatives."""
    ck = torch.load("experiments/attention/EXP-004-attention-cbam-s42/best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_detector(cfg); model.load_state_dict(ck["model"]); model.eval()
    store = PatchStore("data/processed", "val", cfg["data"]["channels"])
    pos = np.where(store.y.numpy() == 1)[0]
    rng = np.random.default_rng(0)
    neg = rng.choice(np.where(store.y.numpy() == 0)[0], 400, replace=False)
    mods = model.backbone.attention_modules()
    for m in mods:
        m.keep = True

    def run(idx):
        gates, maps = [[] for _ in mods], [[] for _ in mods]
        for i in range(0, len(idx), 100):
            x = store.x[idx[i:i + 100]].float() / 255
            with torch.no_grad():
                model(x)
            for k, m in enumerate(mods):
                gates[k].append(m.last_gate.clone())
                maps[k].append(F.interpolate(m.last_spatial, size=64, mode="bilinear", align_corners=False)[:, 0])
        return [torch.cat(g) for g in gates], [torch.cat(m) for m in maps]
    gp, mp = run(pos); gn, mn = run(neg)
    box = store.box[pos].numpy()
    yy, xx = np.mgrid[:64, :64]
    inside = np.stack([np.hypot(xx - (32 + b[0]), yy - (32 + b[1])) <= max(b[2] / 2, 1) for b in box])    # (Np, 64, 64)
    res = {"stages": []}
    for k in range(len(mods)):
        s = mp[k].numpy()
        ins = np.array([s[i][inside[i]].mean() for i in range(len(pos))]); out = np.array([s[i][~inside[i]].mean() for i in range(len(pos))])
        neg_mean = float(mn[k].numpy().mean())
        d = (gp[k].mean(0) - gn[k].mean(0)).abs()
        res["stages"].append({"stage": k, "spatial_inside_nodule": float(ins.mean()), "spatial_outside_nodule": float(out.mean()),
                              "ratio_inside_over_outside": float((ins / out).mean()), "frac_positives_inside_gt_outside": float((ins > out).mean()),
                              "mean_spatial_on_negatives": neg_mean, "channel_gate_mean_pos": float(gp[k].mean()),
                              "channel_gate_mean_neg": float(gn[k].mean()), "channel_gate_mean_abs_diff_pos_vs_neg": float(d.mean()),
                              "channel_gate_max_abs_diff_pos_vs_neg": float(d.max())})
    # figure: example positives, spatial maps for each stage
    pick = rng.choice(len(pos), 5, replace=False)
    fig, ax = plt.subplots(len(pick), 1 + len(mods), figsize=(2.1 * (1 + len(mods)), 2.1 * len(pick)))
    for r, i in enumerate(pick):
        ax[r, 0].imshow(store.x[pos[i]][2], cmap="gray", vmin=0, vmax=255); ax[r, 0].axis("off")
        b = store.box[pos[i]]
        ax[r, 0].add_patch(plt.Rectangle((32 + b[0] - b[2] / 2, 32 + b[1] - b[2] / 2), b[2], b[2], fill=False, ec="#DC2626", lw=1))
        for k in range(len(mods)):
            ax[r, k + 1].imshow(store.x[pos[i]][2], cmap="gray", vmin=0, vmax=255)
            ax[r, k + 1].imshow(mp[k][i].numpy(), cmap="jet", alpha=.5, vmin=0, vmax=1); ax[r, k + 1].axis("off")
            if r == 0:
                ax[r, k + 1].set_title(f"stage {k + 1} spatial", fontsize=8)
    ax[0, 0].set_title("centre slice + box", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "phase8_cbam_attention_maps.png", dpi=130); plt.close()
    # figure: inside/outside ratio per stage + channel-gate class difference
    fig, a = plt.subplots(1, 2, figsize=(9, 3.4))
    st = res["stages"]
    a[0].bar(np.arange(len(st)) - .18, [s["spatial_inside_nodule"] for s in st], .36, color="#DC2626", label="inside nodule box")
    a[0].bar(np.arange(len(st)) + .18, [s["spatial_outside_nodule"] for s in st], .36, color="#94A3B8", label="outside")
    a[0].set(xticks=range(len(st)), xticklabels=[f"stage {s['stage'] + 1}" for s in st], ylabel="mean spatial attention", ylim=(0, 1), title="CBAM spatial attention, validation positives"); a[0].legend(fontsize=8)
    a[1].bar(range(len(st)), [s["channel_gate_mean_abs_diff_pos_vs_neg"] for s in st], color="#0F766E")
    a[1].set(xticks=range(len(st)), xticklabels=[f"stage {s['stage'] + 1}" for s in st], ylabel="mean |gate(pos) − gate(neg)|", title="Channel gates: positives vs negatives")
    [x.grid(alpha=.3, axis="y") for x in a]
    plt.tight_layout(); plt.savefig(FIG / "phase8_attention_stats.png", dpi=130); plt.close()
    return res


def main():
    ann, exc = pd.read_csv("data/annotations.csv"), pd.read_csv("data/annotations/annotations_excluded.csv")
    uids = {s: pd.read_csv(f"data/splits/{s}.csv").seriesuid.tolist() for s in ("val", "test")}
    runs = []
    for label, (parent, prefix, _) in MODELS.items():
        for seed in SEEDS:
            d = Path(parent) / f"{prefix}{seed}"
            if (d / "metrics.json").exists():
                runs.append({"label": label, "seed": seed, "name": d.name, "dir": d, "m": json.loads((d / "metrics.json").read_text())})
    labels = [l for l in MODELS if any(r["label"] == l for r in runs)]
    tab = pd.DataFrame([{"model": r["label"], "experiment": r["name"], "seed": r["seed"], "best_epoch": r["m"]["epoch"],
                         "val_cpm": r["m"]["val"]["cpm"], "test_cpm": r["m"]["test"]["cpm"],
                         "val_sens@1": r["m"]["val"]["sens@1"], "test_sens@1": r["m"]["test"]["sens@1"]} for r in runs])
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TAB / "phase8_runs.csv", index=False)
    agg = tab.groupby("model", sort=False).agg(runs=("seed", "size"), val_cpm_mean=("val_cpm", "mean"), val_cpm_std=("val_cpm", "std"),
                                               test_cpm_mean=("test_cpm", "mean"), test_cpm_std=("test_cpm", "std"),
                                               val_sens1=("val_sens@1", "mean"), test_sens1=("test_sens@1", "mean")).reset_index()
    agg.to_csv(TAB / "phase8_summary.csv", index=False)
    out = {"summary": agg.to_dict("records"), "runs": tab.to_dict("records")}

    vecs = {}
    for r in Progress(runs, desc="loading predictions", unit="run", step_pct=25):
        for s in ("val", "test"):
            vecs[(r["name"], s)] = scan_vectors(pd.read_csv(PRED / r["name"] / f"{s}_predictions.csv"), uids[s], ann, exc)
    out["bootstrap"] = {}
    for s in ("val", "test"):
        boot = bootstrap_cpm([vecs[(r["name"], s)] for r in runs], uids[s], n_boot=1000, seed=0)
        by = {l: boot[:, [i for i, r in enumerate(runs) if r["label"] == l]].mean(1) for l in labels}
        res = {l: {"cpm_mean": float(by[l].mean()), "ci95": ci(by[l])} for l in labels}
        for a, b in (("CNN + SE", "CNN (5 slices)"), ("CNN + CBAM", "CNN (5 slices)"), ("CNN + CBAM", "CNN + SE"),
                     ("CNN + CBAM", "CNN + Transformer (ref.)")):
            if a in by and b in by:
                d = by[a] - by[b]
                res[f"{a} minus {b}"] = {"delta_cpm": float(d.mean()), "ci95": ci(d), "p(delta<=0)": float((d <= 0).mean())}
        out["bootstrap"][s] = res
    out["by_size_at_1fp"] = {s: {l: (lambda lst: {b: {"nodules": lst[0][b]["nodules"], "sensitivity": float(np.mean([x[b]["sensitivity"] for x in lst]))} for b in lst[0]})(
        [sensitivity_by_size(vecs[(r["name"], s)], uids[s]) for r in runs if r["label"] == l]) for l in labels} for s in ("val", "test")}

    eff = efficiency_table()
    eff.to_csv(TAB / "phase8_efficiency.csv", index=False)
    out["efficiency"] = eff.to_dict("records")
    out["attention_analysis"] = attention_analysis(None)
    (TAB / "phase8_comparison.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 3.9))
    for k, s in enumerate(("val", "test")):
        for j, l in enumerate(labels):
            t = tab[tab.model == l][f"{s}_cpm"]
            ax[k].scatter([j] * len(t), t, color=COLORS[l], s=45, zorder=3); ax[k].plot([j - .25, j + .25], [t.mean()] * 2, color=COLORS[l], lw=3)
        ax[k].set(xticks=range(len(labels)), xticklabels=[l.replace(" (", "\n(").replace(" + ", "\n+ ") for l in labels], ylabel="CPM", title=f"CPM — {s} (dots = seeds)"); ax[k].grid(alpha=.3)
    names, mids, lo, hi = [], [], [], []
    for key, v in out["bootstrap"]["val"].items():
        if " minus " in key:
            a, b = key.split(" minus ")
            names.append(f"{a.replace('CNN + ', '')}\n− {b.replace('CNN + ', '').replace(' (5 slices)', '').replace(' (ref.)', '')}")
            mids.append(v["delta_cpm"]); lo.append(v["delta_cpm"] - v["ci95"][0]); hi.append(v["ci95"][1] - v["delta_cpm"])
    ax[2].errorbar(mids, range(len(names)), xerr=[lo, hi], fmt="o", color="#0F766E", capsize=4); ax[2].axvline(0, color="k", lw=1)
    ax[2].set(yticks=range(len(names)), yticklabels=names, xlabel="ΔCPM, paired bootstrap 95% CI", title="Validation ΔCPM"); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "phase8_cpm_comparison.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for r in runs:
        h = pd.read_csv(r["dir"] / "history.csv")
        ax[0].plot(h.epoch, h.val_cpm, color=COLORS[r["label"]], alpha=.8, lw=1.3); ax[1].plot(h.epoch, h.train_loss, color=COLORS[r["label"]], alpha=.8, lw=1.3)
    for l in labels:
        ax[0].plot([], [], color=COLORS[l], label=l)
    ax[0].set(xlabel="epoch", ylabel="validation CPM", title="Validation CPM during training"); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="epoch", ylabel="train loss", title="Training loss"); [a.grid(alpha=.3) for a in ax]
    plt.tight_layout(); plt.savefig(FIG / "phase8_training_curves.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for k, s in enumerate(("val", "test")):
        bins = list(next(iter(out["by_size_at_1fp"][s].values())).keys()); w = 0.8 / len(labels)
        for j, l in enumerate(labels):
            ax[k].bar(np.arange(len(bins)) + (j - (len(labels) - 1) / 2) * w, [out["by_size_at_1fp"][s][l][b]["sensitivity"] for b in bins], w, color=COLORS[l], label=l)
        nn = [out["by_size_at_1fp"][s][labels[0]][b]["nodules"] for b in bins]
        ax[k].set_xticks(range(len(bins))); ax[k].set_xticklabels([f"{b}\n(n={c})" for b, c in zip(bins, nn)])
        ax[k].set(ylim=(0, 1), ylabel="Sensitivity @ 1 FP/scan", title=f"By nodule size — {s}"); ax[k].grid(alpha=.3, axis="y")
    ax[0].legend(fontsize=6.5, loc="lower left")
    plt.tight_layout(); plt.savefig(FIG / "phase8_by_size.png", dpi=140); plt.close()
    print(agg.round(4).to_string(index=False)); print(eff.round(3).to_string(index=False))
    print(json.dumps(out["bootstrap"], indent=1, default=float)); print(json.dumps(out["attention_analysis"], indent=1, default=float))


if __name__ == "__main__":
    main()
