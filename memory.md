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

**Phase:** Phase 5 — 2.5D Data Pipeline (not started)

**Completed:** Phases 0, 1, 2, 3, 4 (Gate 1 passed; 2D baseline established)

**In Progress:** None

**Next Task:** Phase 5 — 2.5D pipeline. Most of it already exists (`extract_25d_patch`, 5-slice patch cache, `slice_stride`); remaining: verify slice order/boundaries/annotation alignment on the cache, visualize 2D vs 2.5D samples, and compare. Phase 6 trains the 2.5D CNN on `channels: null` (all 5) with identical settings and also reports 1/3/5 slices.

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

# Preprocessing (Phase 3 results)

* Pipeline (`src/preprocessing/`, config `configs/preprocessing.yaml`): load from zip → canonicalize orientation (flip x/y back; array (z,y,x), identity direction) → HU clip [-1024, 3071] → resample to 1 mm isotropic (separable linear; masks nearest; origin preserved) → window [-1000, 400] HU → [0,1] (fixed constants, no dataset stats) → lung mask labels 3+4.
* Raw LUNA16 data is already in HU. Lung mask zraw is zlib int16 with labels 3 left, 4 right, 5 trachea.
* Full volumes are NOT cached (would be >100 GB); preprocess per scan on demand (~2 s/scan, ~1.7 GB peak RAM) or cache patches only.
* 2.5D patches: `extract_25d_patch` (64×64, 5 slices z-2..z+2 at 1 mm (decision: user chose 5×1 mm; `slice_stride` option allows wider spacing for the Phase 5/6 sweep), edge boundary, pad 0 = -1000 HU). Patch centre = candidate world coord.
* Candidate index: `data/candidates/candidate_index.csv` (candidates_V2 + split + canonical voxel coords + in_lung flag; git-ignored, rebuild with `scripts/build_candidate_index.py`). Train 604,563 cands/1,218 pos; val 74,488/195; test 75,924/144. 99.7% of candidates (and all but 4 of 1,557 positives) lie in the 5 mm-dilated lung mask.
* Validation (`scripts/validate_preprocessing.py`, 41 scans/79 nodules): 91% of annotation sites dense (>-500 HU mean in 1.5 mm sphere) vs 3.8% for random lung voxels; flipped scans 100% aligned vs 58% at mirrored position; 100% inside dilated lung. Remaining ~9% plausibly ground-glass nodules. Visual check in `notebooks/02_preprocessing_validation.ipynb`.
* Bug fixed during phase: canonical origin shift must apply only to axes with negative direction sign (regression test added).

---

# 2D Baseline (Phase 4 results) — EXP-001-baseline-2d

* Formulation: **candidate scoring** on the supplied LUNA16 `candidates_V2` (official FROC protocol), not full-scan detection. Model: SimpleCNN (4 conv blocks 16-32-64-128, 302,228 params, 1.2 MB), single centre slice of the 5-slice patch, objectness logit + box (dx, dy, log d). Loss `L_det = 1.0·focal(α .25, γ 2) + 1.0·(1−IoU)` (IoU only on positives with a matched nodule box). AdamW 1e-3, wd 1e-4, cosine, batch 256, 20 epochs, seed 42, MPS. Augmentation: D4 flips/rot90 with box-offset adjustment (positives repeated ×10/epoch).
* Patch cache `data/processed/` (4.1 GB, git-ignored; rebuild `scripts/build_patch_cache.py`, ~18 min): train = all 1,218 positives + 60,000 seeded random negatives; val/test = ALL candidates (74,488 / 75,924). uint8 quantization of the [0,1] window (5.5 HU steps). Positive candidates carry the matched nodule box.
* Evaluation `src/evaluation/froc.py`: Python-3 port of the official script, validated by `scripts/validate_evaluation.py` (all counts match the official example output exactly; CPM identical to 6 d.p.). Default = shipped-script protocol (excluded findings radius 5 mm; ≤100 marks/scan); `legacy=True` reproduces the bundled example (older script: excluded radius 0.5 mm, no cap). CPM = mean sensitivity at 1/8,1/4,1/2,1,2,4,8 FP/scan.
* Result (best-val checkpoint = epoch 19): **val CPM 0.632, test CPM 0.675** (sens@1 FP/scan 0.661 / 0.695; max sens 0.958 / 0.962; random-score floor ≈ 0.005–0.007). Single seed, 118/105 nodules → differences of a few CPM points are within noise; add bootstrap CI before concluding anything from small differences. Test scored once (`scripts/evaluate.py`).
* Efficiency: 302,228 params; CPU latency 1.77 ± 0.34 ms (batch 1), 160.8 ms (batch 256); MPS 24.2 ms (batch 256). FLOPs not measured yet.
* Bug found by test: flips/rot90 pivot about (P−1)/2 but the patch centre is pixel P/2 → 1-px shift of image vs box; fixed with a roll after each op (tests added).
* Experiment folder: `experiments/baseline/EXP-001-baseline-2d/` (config.yaml, history.csv, metrics.json, training.log tracked; *.pt git-ignored). Predictions in `results/predictions/` (ignored). Never overwrite experiments: Trainer refuses an existing folder.
* Run: `python scripts/train.py configs/experiments/baseline.yaml baseline` then `python scripts/evaluate.py experiments/baseline/EXP-001-baseline-2d`.

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
