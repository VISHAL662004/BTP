"""Phase 8 runs: attention on the 2.5D CNN. SE (channel) and CBAM (channel+spatial), 3 seeds each,
identical protocol to the Phase 6 5-slice reference. Resumable.

usage: python scripts/run_phase8.py > experiments/attention/phase8_run.log
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import Trainer
from src.utils.config import load_experiment_config
from src.utils.progress import Progress

GROUP = Path("experiments/attention")
# CBAM first (richest variant; channel + spatial), then SE (channel-only ablation).
RUNS = [("attention_cbam", "EXP-004-attention-cbam", s) for s in (42, 43, 44)] + \
       [("attention_se", "EXP-004-attention-se", s) for s in (42, 43, 44)]


def main():
    bar = Progress(RUNS, desc="phase-8 runs", unit="run", step_pct=1)
    for cfg_name, prefix, seed in bar:
        cfg = load_experiment_config(f"configs/experiments/{cfg_name}.yaml")
        cfg["experiment"] = {"name": f"{prefix}-s{seed}", "seed": seed}
        out = GROUP / cfg["experiment"]["name"]
        if (out / "metrics.json").exists():
            print(f"skip {out.name} (done)", flush=True)
            continue
        print(f"\n===== {out.name} (seed {seed}) =====", flush=True)
        Trainer(cfg, out).fit()
        subprocess.run([sys.executable, "scripts/evaluate.py", str(out)], check=True)
        bar.set_postfix(last=out.name)


if __name__ == "__main__":
    main()
