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

**Phase:** Phase 8 complete (attention: no measurable gain); awaiting researcher decision on the pruning base model before Phase 9

**Completed:** Phases 0–8

**In Progress:** None

**Next Task:** Phase 9 — Progressive pruning. DECISION NEEDED: which unpruned reference to prune. Evidence so far (val CPM, 3 seeds): CNN-5slice 0.799, +CBAM 0.800, +SE 0.785, +Transformer 0.774. Recommendation: prune the plain 5-slice CNN (simplest, best cost/performance; Transformer/attention gave no measured benefit) and keep E2/E3 as negative-result rows in the ablation table. Pruning study design: structured (channel) vs unstructured, schedule, fine-tune, report params/FLOPs/latency/sparsity/CPM (rules.md sections 21–22: sparsity != speed-up must be measured).

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
* Experiment folder: `experiments/baseline/EXP-001-baseline-2d/` (config.yaml, history.csv, metrics.json, froc_*.csv tracked; *.pt and training.log git-ignored). Predictions in `results/predictions/` (ignored). Never overwrite experiments: Trainer refuses an existing folder.
* Run: `python scripts/train.py configs/experiments/baseline.yaml baseline` then `python scripts/evaluate.py experiments/baseline/EXP-001-baseline-2d`.

---

# 2.5D Pipeline (Phase 5 results)

* Validation: `python scripts/validate_25d.py` → `data/metadata/25d_validation.json`, figures `results/figures/exp000_25d_*.png`, notebook `notebooks/04_25d_pipeline_validation.ipynb`.
* Cache patches bit-exactly equal an independent pad-then-crop re-extraction (37/37; reversed-order control differs 37/37). All 19 candidates within 2 slices of a scan end match the reference with correct edge replication (none are positives; 19 of them in train).
* All 1,557 positives have boxes inside the patch; 99.2% nodule cores brighter than surrounding ring; nodule centre within the ±2-slice window for 84% (rest are large nodules).
* Input-level 2D vs 2.5D: adjacent-slice correlation 0.98 (±1), 0.94 (±2). Small (<6 mm) nodule core intensity drops to ~59% of centre at ±2 mm, medium (6–10) to ~81%, large (≥10) ~96–97%. Suggests extra slices matter mostly for small nodules — a hypothesis for Phase 6, not a result.
* `centered_channels(total, n)` in `src/data/patch_store.py` selects the n centre channels (e.g. 3 of 5 → [1,2,3]).
* Progress bars: `src/utils/progress.py` (tqdm live bar in a terminal; plain-text `[████░░░░] 5/20 25% | elapsed ETA` lines when redirected). Used by training (epoch + batch bars), cache/index builders, evaluation, validation scripts. Training log with epoch bars: `experiments/<group>/<EXP>/training.log` (git-ignored); `tail -f` it during a run. tqdm added to the environment (conda-forge).
* Reproducibility check: the first epoch of a re-run with seed 42 gave exactly the same loss (0.6549) as EXP-001.

---

# 2.5D CNN baseline (Phase 6 results)

* Sweep: `scripts/run_sweep.py` (resumable; log `experiments/baseline/phase6_sweep.log`, 2h12m total) trained `EXP-002-slices{n}-s{seed}`; identical to EXP-001 except input slices (1/3/5 centre channels of the 5-slice cache) and seed. Runs: 5-slice seeds 42/43/44, 3-slice seed 42, 1-slice seeds 43/44 (+EXP-001 = 1-slice seed 42). `scripts/compare_experiments.py` → `results/tables/phase6_*.{csv,json}`, `results/figures/phase6_*.png`; notebook `03_baseline_analysis.ipynb` (Phase 6 section).
* Validation CPM: 1-slice 0.620 ± 0.012 (3 seeds), 3-slice 0.707 (1 seed), **5-slice 0.799 ± 0.019 (3 seeds)**. Test CPM: 0.668 ± 0.024 / 0.800 / 0.869 ± 0.013. Paired bootstrap ΔCPM val: 5−1 = +0.18 (95% CI 0.12–0.24), 3−1 = +0.09 (0.03–0.15), 5−3 = +0.09 (0.04–0.13); test agrees. Best epochs 13–19 of 20.
* By nodule size @1 FP/scan (val): <6 mm 0.58→0.71→0.84; 6–10 mm 0.52→0.62→0.75; ≥10 mm 0.85→0.90→0.88 (1/3/5 slices). Supports the Phase 5 hypothesis.
* Efficiency: 302,228 / 302,516 / 302,804 params; 105.0 / 107.4 / 109.7 MFLOPs per sample (MACs = half). Batch-1 CPU latency measurements were noisy (machine in use) — redo under quiet conditions before reporting.
* Decision (validation-based): **5-slice input is the default from Phase 7 on.** Untested: wider context (7 slices / stride 2) needs a new patch cache (`slice_stride`).
* Epoch time: ~40 s (1-slice) vs ~45–90 s (5-slice) on MPS (timing varies with other load).
* EXP-001 was re-evaluated once to add the FLOPs field (same checkpoint; CPM unchanged: val 0.632, test 0.675).

---

# CNN–Transformer (Phase 7 results) — Gate 3: not shown useful

* Code: `src/models/backbone/transformer.py` (`TransformerContext`, capacity-matched `ConvContext`, shared interface feature-map→feature-map), `detector.py` composes backbone → optional context → head (`model.context: none|transformer|conv_control`). Configs `cnn_transformer.yaml`, `cnn_extra_conv.yaml`. Runs: `scripts/run_phase7.py` (log `experiments/cnn_transformer/phase7_run.log`, ~4.5 h incl. waiting; resumable), compare: `scripts/compare_phase7.py` → `results/tables/phase7_*`, `results/figures/phase7_*`, notebook `05_model_analysis.ipynb`.
* Design: CNN map 4×4×128 → 16 tokens (dim 128) + learned pos-emb → 2 pre-norm layers (4 heads, MLP ratio 2, dropout 0.1) → map → unchanged head. Attention uses explicit matmuls (FLOPs countable; verified against analytic count and torch MHA).
* Results (3 seeds each, val CPM / test CPM): CNN-5slice 0.799±0.019 / 0.869±0.013; **CNN+Transformer 0.774±0.005 / 0.842±0.027**; conv control 0.791±0.004 / 0.839±0.015. Paired bootstrap ΔCPM val: T−CNN −0.024 (95% CI −0.061…+0.009), T−control −0.019 (−0.052…+0.011), control−CNN −0.005 (−0.032…+0.019); test: −0.025 (−0.059…+0.007), +0.003, −0.028 (−0.055…−0.002). → no evidence of benefit; not significantly worse either.
* Cost: params 302,804 → 570,068 (+88%), FLOPs 109.7 → 118.4 M (+8%), MPS batch-256 latency 25.1 → 31.5 ms, CPU 174 → 197 ms (control 190 ms). Control (598k params) is also no better → not just a capacity effect.
* Behaviour: Transformer starts slower (epoch-1 val CPM 0.24–0.42) but catches up by ~epoch 5; ends with the lowest train loss without better val CPM. Val CPM fluctuates 0.05–0.1 between epochs for all models → best-epoch selection is noisy.
* Caveats: coarse 4×4 token grid (16 tokens), same LR as CNN (not tuned for Transformer), 20 epochs, 118/105 eval nodules. Untested variants: finer token grid (earlier CNN stage), LR/epochs tuning, attention before last pooling.
* A test of the token-mixing unit test caught my own mistake: a constant shift is removed by LayerNorm; use random perturbations.

---

# Attention (Phase 8 results) — Gate 4: no measurable value

* Code: `src/models/attention/attention.py` (`SqueezeExcitation`, `CBAM` = channel+7×7 spatial; shared feature-map→feature-map interface; `keep` flag stores gates/maps), `SimpleCNN(attention=…, attention_stages=…)` inserts after each stage's convs before the pool; config `model.attention: {type: none|se|cbam, reduction: 4, stages: [0,1,2,3]}` (old configs with `attention: false` still build; old checkpoints load strictly). Configs `attention_se.yaml`, `attention_cbam.yaml`. Runs `scripts/run_phase8.py` (log `experiments/attention/phase8_run.log`, ~3.5 h; CBAM epochs ~2 min), compare `scripts/compare_phase8.py` → `results/tables/phase8_*`, `results/figures/phase8_*`, notebook `05_model_analysis.ipynb` (Phase 8 section).
* Results (3 seeds, val / test CPM): CNN 0.799±0.019 / 0.869±0.013; +SE 0.785±0.006 / 0.852±0.019; +CBAM 0.800±0.008 / 0.849±0.005. Paired ΔCPM vs CNN (val): SE −0.014 (CI −0.035…+0.009), CBAM +0.002 (−0.021…+0.029); test: −0.014, −0.019 (CIs include 0). CBAM−SE val +0.016 (−0.008…+0.038).
* Cost (measured idle machine, interleaved median of 3 rounds): params 302,804 → 313,984 (SE) / 314,380 (CBAM) (+3.7%); FLOPs 109.72 M → 109.75 / 110.83 M; activation memory 3.074 → 3.075 / 3.098 MB per sample (peak live 0.524 unchanged); latency CPU b1 2.03 → 2.05 / 2.53 ms, CPU b256 167 → 175 / 198 ms, MPS b256 24.5 → 31.5 / 46.3 ms. Transformer ref: 570,068 params, 118.4 MFLOPs, 3.32 MB act, MPS 30.9 ms, CPU b256 183 ms. FLOPs ≠ latency (launch-bound small kernels on MPS). These are the quiet-condition latencies that supersede the noisy Phase 6/7 values.
* Interpretability (CBAM s42, val positives): spatial attention stages 1–2 ≈ tissue-density map (not nodule-specific); stages 3–4 near-saturated, only 8–11% higher inside nodule box than outside; channel gates differ pos-vs-neg mainly at stage 4 (mean |Δ| 0.26) ≈ 0 at stages 1–2. Descriptive, not causal.
* Added `src/efficiency/memory.py` (hook-based activation/parameter memory, per sample) — first time memory is measured; computed for all models in `phase8_efficiency.csv`.
* Mistake caught during phase: a stray shell line aborted one command batch (configs/tests not yet written); verified by listing files before re-running — always check what a failed batch actually wrote.

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
