"""Phase 9 comparison: progressive pruning of every model (unstructured vs structured).

Reads experiments/pruning/EXP-005-*/level_*/level.json, re-measures latency of every pruned checkpoint on an
IDLE machine (interleaved rounds), runs paired scan-level bootstraps (selected level vs unpruned reference),
and writes results/tables/phase9_*.{csv,json} and results/figures/phase9_*.png.
Selection uses the pre-registered VALIDATION rule (see src/pruning/scheduler.py); test is reported, never used to choose.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.efficiency.latency import measure_latency
from src.evaluation.froc import bootstrap_cpm, scan_vectors
from src.models.detector import build_detector
from src.pruning.scheduler import select_final_level
from src.utils.progress import Progress

GROUP, PRED, FIG, TAB = Path("experiments/pruning"), Path("results/predictions"), Path("results/figures"), Path("results/tables")
REF = {"cnn": "EXP-002-slices5-s42", "se": "EXP-004-attention-se-s42", "cbam": "EXP-004-attention-cbam-s42",
       "transformer": "EXP-003-cnn-transformer-s42", "convctrl": "EXP-003c-cnn-extraconv-s42"}
REFDIR = {"cnn": "baseline", "se": "attention", "cbam": "attention", "transformer": "cnn_transformer", "convctrl": "cnn_transformer"}
LABEL = {"cnn": "CNN", "se": "CNN + SE", "cbam": "CNN + CBAM", "transformer": "CNN + Transformer", "convctrl": "CNN + extra conv"}
COLOR = {"cnn": "#2563EB", "se": "#D97706", "cbam": "#0F766E", "transformer": "#DC2626", "convctrl": "#64748B"}
MAX_DROP = 0.02


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def load_runs():
    rows = []
    for d in sorted(GROUP.glob("EXP-005-*")):
        _, _, model, kind = d.name.split("-", 3)
        ref = json.loads((Path("experiments") / REFDIR[model] / REF[model] / "metrics.json").read_text())
        for lv in sorted(d.glob("level_*/level.json")):
            r = json.loads(lv.read_text()); r.update(model=model, type=kind, exp=d.name, ckpt=str(lv.parent / "best.pt"))
            rows.append(r)
        # level 0 = unpruned reference
        e = ref["efficiency"]
        rows.append({"level": 0, "target": 0.0, "model": model, "type": kind, "exp": d.name, "ckpt": str(Path("experiments") / REFDIR[model] / REF[model] / "best.pt"),
                     "val_cpm": ref["val"]["cpm"], "test_cpm": ref["test"]["cpm"], "val_sens@1": ref["val"]["sens@1"], "test_sens@1": ref["test"]["sens@1"],
                     "params_total": e["parameters"], "params_nonzero": e["nonzero"], "nonzero_reduction": 0.0,
                     "flops_dense": e["flops_per_sample"], "flops_effective": e["flops_per_sample"], "activations_mb": np.nan})
    return pd.DataFrame(rows).drop_duplicates(["exp", "level"]).sort_values(["exp", "level"]).reset_index(drop=True)


def measure_all_latency(df):
    """Interleaved rounds over every checkpoint (dense kernels; unstructured zeros do NOT make them faster)."""
    models = {}
    for i, r in df.iterrows():
        ck = torch.load(r["ckpt"], map_location="cpu", weights_only=False)
        m = build_detector(ck["config"]); m.load_state_dict(ck["model"]); models[i] = m.eval()
    res = {(i, k): [] for i in models for k in ("cpu_b1", "cpu_b256", "mps_b256")}
    mps = torch.backends.mps.is_available()
    for _ in Progress(range(3), desc="latency rounds", unit="round", step_pct=34):
        for i, m in models.items():
            res[(i, "cpu_b1")].append(measure_latency(m, (1, 5, 64, 64), "cpu", warmup=10, runs=60)["latency_ms_mean"])
            res[(i, "cpu_b256")].append(measure_latency(m, (256, 5, 64, 64), "cpu", warmup=2, runs=6)["latency_ms_mean"])
            if mps:
                res[(i, "mps_b256")].append(measure_latency(m, (256, 5, 64, 64), "mps", warmup=5, runs=20)["latency_ms_mean"])
    for k in ("cpu_b1", "cpu_b256", "mps_b256"):
        df[f"lat_{k}"] = [float(np.median(res[(i, k)])) if res[(i, k)] else np.nan for i in df.index]
    return df


def main():
    df = load_runs()
    if df.empty:
        raise SystemExit("no pruning results yet")
    df = measure_all_latency(df)
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    keep = ["model", "type", "level", "target", "val_cpm", "test_cpm", "val_sens@1", "test_sens@1", "pre_ft_val_cpm", "best_epoch_in_level",
            "params_total", "params_nonzero", "nonzero_reduction", "flops_dense", "flops_effective", "activations_mb", "lat_cpu_b1", "lat_cpu_b256", "lat_mps_b256"]
    df[[c for c in keep if c in df.columns]].to_csv(TAB / "phase9_levels.csv", index=False)

    ann, exc = pd.read_csv("data/annotations.csv"), pd.read_csv("data/annotations/annotations_excluded.csv")
    uids = {s: pd.read_csv(f"data/splits/{s}.csv").seriesuid.tolist() for s in ("val", "test")}
    sel_rows, out = [], {"selection_rule": f"highest level with val CPM >= reference val CPM - {MAX_DROP} (fixed before results)", "selected": []}
    for (model, kind), g in df[df.type != "finetune_only"].groupby(["model", "type"]):
        ref_row = g[g.level == 0].iloc[0]
        chosen = select_final_level(g[g.level > 0].to_dict("records"), ref_row["val_cpm"], MAX_DROP)
        if chosen is None:
            sel_rows.append({"model": model, "type": kind, "selected_level": None}); continue
        s = {"model": model, "type": kind, "selected_level": chosen["level"], "target": chosen["target"],
             "val_cpm_ref": ref_row["val_cpm"], "val_cpm": chosen["val_cpm"], "test_cpm_ref": ref_row["test_cpm"], "test_cpm": chosen["test_cpm"],
             "params_nonzero_ref": ref_row["params_nonzero"], "params_nonzero": chosen["params_nonzero"],
             "param_reduction": 1 - chosen["params_nonzero"] / ref_row["params_total"],
             "flops_effective_ref": ref_row["flops_effective"], "flops_effective": chosen["flops_effective"],
             "lat_cpu_b1_ref": ref_row["lat_cpu_b1"], "lat_cpu_b1": chosen["lat_cpu_b1"], "lat_mps_b256_ref": ref_row["lat_mps_b256"], "lat_mps_b256": chosen["lat_mps_b256"]}
        # paired bootstrap: pruned (selected) vs unpruned reference, same scan resamples
        for split in ("val", "test"):
            v_ref = scan_vectors(pd.read_csv(PRED / REF[model] / f"{split}_predictions.csv"), uids[split], ann, exc)
            v_pr = scan_vectors(pd.read_csv(PRED / f"{g.exp.iloc[0]}_L{chosen['level']}" / f"{split}_predictions.csv"), uids[split], ann, exc)
            b = bootstrap_cpm([v_ref, v_pr], uids[split], n_boot=1000, seed=0)
            d = b[:, 1] - b[:, 0]
            s[f"{split}_delta_cpm"], s[f"{split}_delta_ci_low"], s[f"{split}_delta_ci_high"] = float(d.mean()), ci(d)[0], ci(d)[1]
        sel_rows.append(s)
    # ---- POST-HOC (not pre-registered): each pruned CNN level vs the no-pruning fine-tune control at the same level
    ctrl = df[(df.model == "cnn") & (df.type == "finetune_only")].set_index("level")
    vc = []
    for kind in ("unstructured", "structured"):
        g = df[(df.model == "cnn") & (df.type == kind) & (df.level > 0)]
        for r in g.itertuples():
            row = {"type": kind, "level": r.level, "target": r.target, "val_cpm": r.val_cpm, "control_val_cpm": ctrl.loc[r.level, "val_cpm"],
                   "test_cpm": r.test_cpm, "control_test_cpm": ctrl.loc[r.level, "test_cpm"], "nonzero_reduction": r.nonzero_reduction}
            for split in ("val", "test"):
                v_c = scan_vectors(pd.read_csv(PRED / f"EXP-005-cnn-finetune_only_L{r.level}" / f"{split}_predictions.csv"), uids[split], ann, exc)
                v_p = scan_vectors(pd.read_csv(PRED / f"EXP-005-cnn-{kind}_L{r.level}" / f"{split}_predictions.csv"), uids[split], ann, exc)
                b = bootstrap_cpm([v_c, v_p], uids[split], n_boot=500, seed=0)
                d = b[:, 1] - b[:, 0]
                row[f"{split}_delta_vs_control"], row[f"{split}_delta_ci_low"], row[f"{split}_delta_ci_high"] = float(d.mean()), ci(d)[0], ci(d)[1]
            vc.append(row)
    pd.DataFrame(vc).to_csv(TAB / "phase9_vs_control.csv", index=False)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(TAB / "phase9_selected.csv", index=False)
    out["selected"] = sel.to_dict("records")
    (TAB / "phase9_summary.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- figures
    for kind in ("unstructured", "structured"):
        d = df[df.type == kind]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for model, g in d.groupby("model"):
            x = g.nonzero_reduction * 100
            ax[0].plot(x, g.val_cpm, "o-", color=COLOR[model], label=LABEL[model])
            ax[1].plot(x, g.test_cpm, "o-", color=COLOR[model])
            ax[2].plot(x, g.lat_mps_b256, "o-", color=COLOR[model])
        for k in range(2):
            ax[k].set(xlabel="non-zero parameter reduction (%)", ylabel="CPM", title=("Validation" if k == 0 else "Test") + f" CPM — {kind} pruning"); ax[k].grid(alpha=.3)
        ax[2].set(xlabel="non-zero parameter reduction (%)", ylabel="MPS latency, batch 256 (ms)", title="Measured latency"); ax[2].grid(alpha=.3)
        ax[0].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(FIG / f"phase9_{kind}_tradeoff.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for kind, mk in (("unstructured", "o--"), ("structured", "s-")):
        g = df[(df.model == "cnn") & (df.type == kind)]
        ax[0].plot(g.nonzero_reduction * 100, g.test_cpm, mk, color="#2563EB" if kind == "structured" else "#DC2626", label=kind)
        ax[1].plot(g.nonzero_reduction * 100, g.lat_cpu_b1, mk, color="#2563EB" if kind == "structured" else "#DC2626", label=kind)
        ax[2].plot(g.nonzero_reduction * 100, g.flops_effective / 1e6, mk, color="#2563EB" if kind == "structured" else "#DC2626", label=kind)
    ax[0].set(ylabel="test CPM", title="CNN: accuracy"); ax[1].set(ylabel="CPU latency, batch 1 (ms)", title="CNN: measured latency (dense kernels)")
    ax[2].set(ylabel="MFLOPs per sample", title="CNN: (effective) FLOPs")
    for a in ax:
        a.set_xlabel("non-zero parameter reduction (%)"); a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(FIG / "phase9_sparsity_vs_speed.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in sel.dropna(subset=["val_cpm"]).iterrows():
        ax.scatter(r["lat_mps_b256"], r["test_cpm"], color=COLOR[r["model"]], marker="s" if r["type"] == "structured" else "o", s=70)
        ax.annotate(f"{LABEL[r['model']].replace('CNN + ', '+')} ({r['type'][:6]})", (r["lat_mps_b256"], r["test_cpm"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    for model, g in df[df.level == 0].groupby("model"):
        ax.scatter(g.lat_mps_b256.iloc[0], g.test_cpm.iloc[0], color=COLOR[model], marker="*", s=160, edgecolor="k")
    ax.set(xlabel="MPS latency, batch 256 (ms)", ylabel="test CPM", title="Selected pruned models vs unpruned (stars)"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "phase9_selected_models.png", dpi=140); plt.close()
    print(sel.round(4).to_string(index=False))
    print(df[df.model == "cnn"][["type", "level", "target", "val_cpm", "test_cpm", "params_nonzero", "flops_effective", "lat_cpu_b1", "lat_mps_b256"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
