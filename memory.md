# Project Memory

**File:** `memory.md`
**Project:** 2.5D CNN–Transformer Framework for Lung Nodule Detection
**Version:** 1.1
**Status:** Phase 0 complete (documentation synced with `Project_Report.pdf`)

---

# Purpose

This file is the **persistent project-state memory** for the AI working on this project.

It should contain only concise information needed to understand the **current state of the project** and continue work correctly across sessions.

The AI should update this file whenever a significant project decision, implementation milestone, experiment result, issue, or next step changes the project state.

---

# Current State

**Phase:** Phase 3 — CT Preprocessing (not started)

**Completed:** Phases 0, 1, 2 (dataset validated, Gate 1 passed)

**In Progress:** None

**Next Task:** Begin Phase 3 — CT preprocessing (HU, windowing, normalization, resampling, lung ROI, slice extraction, patch/candidate generation); must apply per-scan direction matrix for world→voxel

---

# Project Decisions

* Primary dataset: **LUNA16**
* Primary representation: **2.5D CT**
* Primary architecture direction: **CNN–Transformer**
* Research components: **Attention + Progressive Pruning**
* Nodule Feature Diversity (NFD) regularization, the consistency/feature loss, and the LightSeNet-inspired diversity loss have been **removed** (revised approach per `Project_Report.pdf`).
* Ablation plan: E0 2D CNN → E1 2.5D CNN → E2 + Transformer → E3 + Attention → E4 + Progressive Pruning (final model).
* Final architecture is **not yet fixed**.
* Exact CNN, Transformer, attention, and pruning configurations must be determined experimentally.
* Efficiency is an important evaluation dimension.
* Research claims must be supported by experiments and literature.

---

# Current Loss Function Decision

The model is trained with the detection objective only:

```text
L_total = L_det = λ_cls L_focal + λ_loc L_loc
```

`L_focal` is a focal classification/objectness loss and `L_loc` is an
IoU-based localization loss (`1 - IoU(B, B*)`). The exact localization
formulation is finalized once the detection head is selected.

No NFD or other auxiliary regularization term is used.

**Next task:** Begin Phase 1 and proceed through the baselines.

---

# Environment

* Conda env `btp-lung` (`/opt/homebrew/Caskroom/miniconda/base/envs/btp-lung`), Jupyter kernel "Python 3.11 (btp-lung)".
* Versions: Python 3.11.16, PyTorch 2.10.0 (conda-forge), NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.6, scikit-learn 1.9.1, SimpleITK 2.5.6, nibabel 5.4.2, matplotlib 3.11.2, PyYAML 6.0.3, pytest 9.1.1.
* CPU and Apple MPS both verified (`python scripts/check_env.py`, `pytest`).
* Gotcha: torch must come from conda-forge, not pip — the pip wheel's bundled libomp clashes with conda numpy (OMP Error #15).

---

# Dataset (Phase 2 results)

* All 10 subsets present as zips in `data/` (not extracted; headers read in place via `src/data/luna16.py`; ~29 GB free disk). CRC of all zips OK.
* 888 scans (all 512×512; 95–764 slices; in-plane 0.46–0.98 mm; z-spacing 0.45–2.5 mm), 1186 annotations (601 scans with nodules, 287 without), diameter 3.3–32.3 mm (median 6.4).
* candidates.csv: 551,065 rows / 1,351 positives; candidates_V2: 754,975 / 1,557 (extracted to `data/candidates/`). Lung masks: seg-lungs-LUNA16.zip covers all scans.
* 14 scans have x/y axes flipped (TransformMatrix diag −1,−1,1). World→voxel must use the direction matrix (`ScanInfo.world_to_voxel`); ignoring it puts 24 annotations out of bounds. After correction: 0 issues.
* Split (scan level, by subset, no randomness): train subsets 0–7 (712 scans, 963 nodules), val subset 8 (88, 118), test subset 9 (88, 105). Files in `data/splits/`.
* Regenerate everything: `python scripts/validate_dataset.py [--crc]`; exploration in `notebooks/01_data_exploration.ipynb`.
* Official evaluation script is inside `data/evaluationScript.zip` (needed to validate FROC/CPM in Phase 4).

---

# How AI Should Use This File

Before continuing project work, the AI should use this file together with:

```text
prd.md
architecture.md
rules.md
phases.md
design.md
```

to understand the current project state.

The AI should:

* update completed work,
* record important decisions,
* record current work,
* record blockers,
* record important experiment results,
* record the next actionable task.

Keep entries **short and factual**.

Do not turn this file into a detailed project report. Detailed information belongs in the appropriate project documentation files.

---

# Memory Update Format

Use this structure when updating the file:

```text
## Current State

Phase:
Completed:
In Progress:
Next Task:

## Recent Decisions

- ...

## Important Results

- ...

## Known Issues

- ...

## Next Actions

1. ...
2. ...
3. ...
```

Only add sections when they contain useful project-state information.
