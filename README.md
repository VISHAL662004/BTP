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

Phases 0–1 complete. Next: Phase 2 — dataset understanding and validation. See [memory.md](memory.md).
