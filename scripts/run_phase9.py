"""Phase 9 orchestrator: progressive pruning of every model, one by one. Resumable (finished levels skipped).

usage: python scripts/run_phase9.py > experiments/pruning/phase9_run.log
Order: plain CNN first (unstructured, structured, no-pruning fine-tune control), then the other models.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.progress import Progress

RUNS = [("cnn", "unstructured"), ("cnn", "structured"), ("cnn", "finetune_only"),
        ("cbam", "unstructured"), ("cbam", "structured"), ("se", "unstructured"), ("se", "structured"),
        ("transformer", "unstructured"), ("transformer", "structured"),
        ("convctrl", "unstructured"), ("convctrl", "structured")]


def main():
    bar = Progress(RUNS, desc="phase-9 pruning runs", unit="run", step_pct=1)
    for model, kind in bar:
        print(f"\n########## {model} / {kind} ##########", flush=True)
        subprocess.run([sys.executable, "scripts/prune.py", "--model", model, "--type", kind], check=True)
        bar.set_postfix(last=f"{model}/{kind}")


if __name__ == "__main__":
    main()
