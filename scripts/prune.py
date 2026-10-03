"""Phase 9: progressive pruning of ONE model with ONE method.

    python scripts/prune.py --model cnn --type unstructured
    --model {cnn,se,cbam,transformer,convctrl}   (each starts from its trained seed-42 checkpoint)
    --type  {unstructured,structured,finetune_only}   (finetune_only = no pruning; control for the extra training)

Per level: prune -> fine-tune (best-validation epoch within the level) -> evaluate (val + test) -> record.
Pruning decisions use the VALIDATION split only; test numbers are recorded for every level for reporting.
The final model is chosen by the pre-registered rule in src/pruning/scheduler.select_final_level.
Resumable: finished levels (level.json) are skipped. Output: experiments/pruning/EXP-005-<model>-<type>/
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.patch_store import PatchStore
from src.efficiency.flops import effective_flops
from src.efficiency.memory import activation_memory
from src.efficiency.parameters import count_parameters
from src.evaluation.evaluator import predict_proba, score
from src.models.detector import build_detector
from src.pruning.masks import apply_masks, compute_global_masks, prunable, sparsity_report
from src.pruning.scheduler import levels as make_levels, select_final_level
from src.pruning.strategy import stage_widths, structured_prune, target_widths
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.utils.progress import Progress

SOURCES = {
    "cnn": "experiments/baseline/EXP-002-slices5-s42",
    "se": "experiments/attention/EXP-004-attention-se-s42",
    "cbam": "experiments/attention/EXP-004-attention-cbam-s42",
    "transformer": "experiments/cnn_transformer/EXP-003-cnn-transformer-s42",
    "convctrl": "experiments/cnn_transformer/EXP-003c-cnn-extraconv-s42",
}
GROUP = Path("experiments/pruning")
FT = {"epochs_per_level": 4, "learning_rate": 3e-4, "max_val_cpm_drop": 0.02}     # fixed for ALL models / methods


def evaluate_model(model, stores, uids, ann, exc, device, tag):
    out = {}
    for split in ("val", "test"):
        prob = predict_proba(model, stores[split], device)
        res, pred = score(stores[split], prob, uids[split], ann, exc, 100)
        d = Path("results/predictions") / tag
        d.mkdir(parents=True, exist_ok=True)
        pred.sort_values("probability", ascending=False).groupby("seriesuid").head(100).to_csv(d / f"{split}_predictions.csv", index=False)
        s = res.summary()
        out[split] = {"cpm": s["cpm"], "sens@1": s["sens@1"], "max_sens": s["max_sensitivity"]}
    return out


def efficiency(model, ref_params):
    model = model.to("cpu")
    p = count_parameters(model)
    f = effective_flops(model, (1, 5, 64, 64))
    m = activation_memory(model, (1, 5, 64, 64))
    return {"params_total": p["parameters"], "params_nonzero": p["nonzero"], "sparsity_total": p["sparsity"],
            "nonzero_reduction": 1 - p["nonzero"] / ref_params, "flops_dense": f["flops_dense"],
            "flops_effective": f["flops_effective"], "activations_mb": m["activations_mb_per_sample"],
            "parameters_mb_dense": m["parameters_mb"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=SOURCES)
    ap.add_argument("--type", required=True, choices=("unstructured", "structured", "finetune_only"))
    ap.add_argument("--epochs-per-level", type=int, default=FT["epochs_per_level"])
    ap.add_argument("--levels", type=float, nargs="*", help="override schedule (testing)")
    ap.add_argument("--out", default=None, help="override output dir (testing)")
    args = ap.parse_args()

    src = Path(SOURCES[args.model])
    ck = torch.load(src / "best.pt", map_location="cpu", weights_only=False)
    cfg0 = ck["config"]
    ref = json.loads((src / "metrics.json").read_text())
    name = f"EXP-005-{args.model}-{args.type}"
    out = Path(args.out) if args.out else GROUP / name
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg0["device"]["preferred"])
    ev = cfg0["evaluation"]
    ann, exc = pd.read_csv(ev["annotations"]), pd.read_csv(ev["excluded"])
    uids = {s: pd.read_csv(f"{ev['splits_dir']}/{s}.csv").seriesuid.tolist() for s in ("val", "test")}
    stores = {s: PatchStore(cfg0["data"]["cache_dir"], s, cfg0["data"]["channels"]) for s in ("val", "test")}

    model = build_detector(cfg0)
    model.load_state_dict(ck["model"])
    cfg = copy.deepcopy(cfg0)
    ref_params = count_parameters(model)["parameters"]
    mid0, out0 = stage_widths(cfg0)
    sched = make_levels("structured" if args.type == "structured" else "unstructured", args.levels) if args.type != "finetune_only" \
        else tuple(0.0 for _ in range(5))
    rows = [{"level": 0, "target": 0.0, "val_cpm": ref["val"]["cpm"], "test_cpm": ref["test"]["cpm"],
             "val_sens@1": ref["val"]["sens@1"], "test_sens@1": ref["test"]["sens@1"], "pre_ft_val_cpm": ref["val"]["cpm"],
             "best_epoch_in_level": ck["epoch"], **efficiency(model, ref_params)}]
    masks = None
    print(f"{name}: reference val CPM {ref['val']['cpm']:.4f}, params {ref_params:,}; schedule {sched}", flush=True)
    bar = Progress(list(enumerate(sched, 1)), desc=f"{name} levels", unit="level", step_pct=1)
    for i, target in bar:
        lvl = out / f"level_{i:02d}"
        done = lvl / "level.json"
        if done.exists():                                               # resume: restore state, keep going
            row = json.loads(done.read_text()); rows.append(row)
            c = torch.load(lvl / "best.pt", map_location="cpu", weights_only=False)
            cfg = c["config"]; model = build_detector(cfg); model.load_state_dict(c["model"])
            if args.type == "unstructured":
                masks = {n: (w != 0) for n, w in prunable(model).items()}
            continue
        t0 = time.time()
        post = None
        if args.type == "unstructured":
            masks = compute_global_masks(model, target, masks)
            apply_masks(model, masks)
            post = lambda: apply_masks(model, masks)                    # noqa: E731
        elif args.type == "structured":
            model, cfg = structured_prune(model, cfg, target_widths(mid0, target), target_widths(out0, target, keep_last=True))
        pre = evaluate_model(model.to(device), stores, uids, ann, exc, device, f"{name}_L{i}_prefinetune")["val"]["cpm"]
        fcfg = copy.deepcopy(cfg)
        fcfg["experiment"] = {"name": f"{name}-L{i}", "seed": cfg0["experiment"]["seed"] + i}
        fcfg["training"].update({"epochs": args.epochs_per_level, "learning_rate": FT["learning_rate"]})
        tr = Trainer(fcfg, lvl, model=model, post_step=post)
        tr.fit()
        c = torch.load(lvl / "best.pt", map_location="cpu", weights_only=False)      # best validation epoch of this level
        model = build_detector(c["config"]); model.load_state_dict(c["model"]); cfg = c["config"]
        met = evaluate_model(model.to(device), stores, uids, ann, exc, device, f"{name}_L{i}")
        row = {"level": i, "target": target, "val_cpm": met["val"]["cpm"], "test_cpm": met["test"]["cpm"],
               "val_sens@1": met["val"]["sens@1"], "test_sens@1": met["test"]["sens@1"], "pre_ft_val_cpm": pre,
               "best_epoch_in_level": c["epoch"], "seconds": time.time() - t0, **efficiency(model, ref_params)}
        if args.type == "unstructured":
            row["prunable_sparsity"] = sparsity_report(model)["prunable_sparsity"]
        done.write_text(json.dumps(row, indent=1))
        rows.append(row)
        bar.set_postfix(val_cpm=row["val_cpm"], params=row["params_nonzero"])
    df = pd.DataFrame(rows)
    df.to_csv(out / "levels.csv", index=False)
    chosen = select_final_level(rows[1:], ref["val"]["cpm"], FT["max_val_cpm_drop"]) if args.type != "finetune_only" else None
    (out / "summary.json").write_text(json.dumps({
        "experiment": name, "source": str(src), "type": args.type, "fine_tune": FT, "reference_val_cpm": ref["val"]["cpm"],
        "selected_level": chosen["level"] if chosen else None, "selected": chosen}, indent=1, default=float))
    print(df[["level", "target", "pre_ft_val_cpm", "val_cpm", "test_cpm", "params_total", "params_nonzero", "flops_effective"]].round(4).to_string(index=False))
    print("selected level (val rule):", chosen["level"] if chosen else None)


if __name__ == "__main__":
    main()
