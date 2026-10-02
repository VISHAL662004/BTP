"""Phase 6 sweep: CNN baseline with 1 / 3 / 5 input slices x several seeds, identical settings
otherwise (configs/experiments/baseline.yaml). Resumable: finished runs (metrics.json) are skipped.

usage: python scripts/run_sweep.py            (log it:  > experiments/baseline/phase6_sweep.log)
Experiment names: EXP-002-slices{n}-s{seed}  (EXP-001-baseline-2d is the existing 1-slice, seed-42 run).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.patch_store import centered_channels
from src.training.trainer import Trainer
from src.utils.config import load_experiment_config
from src.utils.progress import Progress

GROUP = Path("experiments/baseline")
# (n_slices, seed); EXP-001 already covers (1, 42). Primary comparison first.
# 1 vs 5 slices get 3 seeds each (headline comparison, seed noise); 3 slices one seed (dose-response hint).
RUNS = [(5, 42), (3, 42), (1, 43), (5, 43), (1, 44), (5, 44)]


def make_cfg(n: int, seed: int) -> dict:
    cfg = load_experiment_config("configs/experiments/baseline.yaml")
    cfg["experiment"] = {"name": f"EXP-002-slices{n}-s{seed}", "seed": seed}
    cfg["data"]["channels"] = centered_channels(5, n)
    cfg["model"]["in_channels"] = n
    return cfg


def main():
    bar = Progress(RUNS, desc="phase-6 sweep", unit="run", step_pct=1)
    for n, seed in bar:
        cfg = make_cfg(n, seed)
        out = GROUP / cfg["experiment"]["name"]
        if (out / "metrics.json").exists():
            print(f"skip {out.name} (done)", flush=True)
            continue
        print(f"\n===== {out.name}: {n} slice(s), seed {seed} =====", flush=True)
        Trainer(cfg, out).fit()
        subprocess.run([sys.executable, "scripts/evaluate.py", str(out)], check=True)
        bar.set_postfix(last=out.name)


if __name__ == "__main__":
    main()
