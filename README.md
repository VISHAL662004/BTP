# A 2.5D CNN–Transformer Framework with Attention and Progressive Pruning for Lung Nodule Detection

B.Tech. project, Department of Computer Science and Engineering, Netaji Subhas University of Technology.

**Authors:** Vishal Kumar (2023UCA1854), Rydham Bansal (2023UCS1599), Kushagra Sabharwal (2023UCA1851)
**Supervisor:** Dr. Surendra Nagar, Assistant Professor

## Objective

Investigate whether a 2.5D CNN–Transformer lung nodule detector, enhanced with attention and progressive pruning, achieves a useful balance between detection performance (FROC, CPM, sensitivity, FPs/scan) and computational efficiency (parameters, FLOPs, memory, latency, sparsity, throughput) on LUNA16.

The value of each component is determined through single-variable ablation (E0 2D CNN → E1 2.5D CNN → E2 + Transformer → E3 + Attention → E4 + Progressive Pruning), not assumed.

> Research prototype only. Not a clinical diagnostic tool.

## Documentation

| File | Content |
| ---- | ------- |
| [Project_Report.pdf](Project_Report.pdf) | Source-of-truth project report |
| [prd.md](prd.md) | Requirements and research objectives |
| [architecture.md](architecture.md) | System architecture, repository layout, loss formulation |
| [rules.md](rules.md) | Development and research rules |
| [phases.md](phases.md) | Phase roadmap (Phases 0–17) |
| [design.md](design.md) | Visual and presentation standards |
| [memory.md](memory.md) | Current project state |

## Environment

Python 3.11, PyTorch (Apple MPS where stable, CPU fallback), NumPy, SciPy, pandas, scikit-learn, SimpleITK, nibabel, PyYAML, matplotlib, pytest. Developed on a MacBook Pro (M3, 8 GB RAM).

## Setup

```bash
conda env create -f environment.yml
conda activate btp-lung
python scripts/check_env.py   # library, CPU and MPS checks
pytest
```

## Status

Phases 0–8 complete. Next: Phase 9 — progressive pruning.

Phase 8 (SE / CBAM attention on the 5-slice CNN, 3 seeds): val CPM 0.785 / 0.800 vs 0.799 for the CNN — no measurable gain, CBAM +89% MPS latency (negative result recorded).

Phase 7 (CNN + Transformer, 3 seeds, capacity-matched control): val CPM 0.774 ± 0.005 vs 0.799 ± 0.019 for the 5-slice CNN — no measured benefit (negative result recorded; see `notebooks/05_model_analysis.ipynb`).

Baselines (official LUNA16 FROC/CPM, validation CPM, 3 seeds): 2D CNN 0.620 ± 0.012; 2.5D CNN with 5 slices 0.799 ± 0.019 (test 0.668 → 0.869). Sweep: `python scripts/run_sweep.py`, summary: `python scripts/compare_experiments.py`. See `notebooks/03_baseline_analysis.ipynb`.

Dataset validation: `python scripts/validate_dataset.py` (outputs in `data/metadata/`, splits in `data/splits/`).
2.5D check: `python scripts/validate_25d.py` (notebook `04_25d_pipeline_validation.ipynb`). Training progress is shown as live bars and logged to `experiments/<group>/<EXP>/training.log`.
Baseline: `python scripts/build_patch_cache.py`, `python scripts/train.py configs/experiments/baseline.yaml baseline`, `python scripts/evaluate.py experiments/baseline/EXP-001-baseline-2d`; evaluation check: `python scripts/validate_evaluation.py`.
Preprocessing: `python scripts/build_candidate_index.py`, `python scripts/validate_preprocessing.py` (see `configs/preprocessing.yaml`). See [memory.md](memory.md).
