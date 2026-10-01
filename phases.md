# Project Phases

**File:** `phases.md`
**Project:** 2.5D CNN–Transformer Framework for Lung Nodule Detection
**Version:** 1.1
**Status:** Revised Project Roadmap (synced with `Project_Report.pdf`)

---

# 1. Purpose

This document defines the complete development and research roadmap for the project.

The project will be developed incrementally. Each phase has:

* a specific objective,
* required tasks,
* expected outputs,
* validation criteria,
* dependencies on previous phases.

No phase should be considered complete simply because its code has been written. A phase is complete only when its implementation, validation, documentation, and required outputs are finished.

---

# 2. Overall Project Flow

```text
PHASE 0
Project Initialization
        ↓
PHASE 1
Environment & Repository
        ↓
PHASE 2
Dataset Understanding & Validation
        ↓
PHASE 3
CT Preprocessing
        ↓
PHASE 4
2D Baseline
        ↓
PHASE 5
2.5D Data Pipeline
        ↓
PHASE 6
2.5D CNN Baseline
        ↓
PHASE 7
CNN–Transformer Architecture
        ↓
PHASE 8
Attention Module
        ↓
PHASE 9
Progressive Pruning
        ↓
PHASE 10
Complete Model
        ↓
PHASE 11
Ablation Study
        ↓
PHASE 12
Efficiency Evaluation
        ↓
PHASE 13
Error Analysis
        ↓
PHASE 14
Final Experiments
        ↓
PHASE 15
Research Analysis
        ↓
PHASE 16
Final Documentation
        ↓
PHASE 17
Reproducibility & Finalization
```

---

# 3. Phase 0 — Project Initialization

## Objective

Establish the project definition, research direction, documentation structure, and development rules before implementation begins.

## Tasks

* Create repository.
* Create project documentation.
* Define research objective.
* Define initial architecture.
* Define project rules.
* Define development phases.
* Create project memory file.
* Initialize Git.

## Required Files

```text
prd.md
architecture.md
rules.md
phases.md
design.md
memory.md
README.md
```

## Completion Criteria

```text
[ ] Research objective defined
[ ] Architecture direction defined
[ ] Repository structure defined
[ ] Development rules defined
[ ] Phase roadmap defined
[ ] AI/project memory system established
[ ] Git initialized
```

---

# 4. Phase 1 — Environment & Repository Setup

## Objective

Create a reproducible Python development environment suitable for the project.

## Tasks

* Create Conda environment.
* Install required dependencies.
* Configure PyTorch.
* Verify CPU execution.
* Verify Apple MPS where applicable.
* Configure VS Code.
* Configure Jupyter.
* Configure Git.
* Create directory structure.
* Create initial configuration system.

## Validation

Run environment checks for:

```text
Python
PyTorch
NumPy
SimpleITK
nibabel
pandas
scikit-learn
matplotlib
Jupyter
```

Verify:

```text
CPU → available
MPS → available where supported
```

## Deliverables

```text
environment.yml
requirements.txt
.gitignore
configs/base.yaml
```

## Completion Criteria

A minimal test script must successfully:

1. import required libraries,
2. create a PyTorch tensor,
3. execute a small model,
4. run on CPU,
5. run on MPS where supported.

---

# 5. Phase 2 — Dataset Understanding & Validation

## Objective

Understand and validate the local LUNA16 dataset before model development.

## Tasks

* Identify all CT scan files.
* Identify annotation files.
* Inspect metadata.
* Verify scan/annotation relationships.
* Inspect voxel spacing.
* Inspect slice dimensions.
* Inspect scan dimensions.
* Verify coordinate systems.
* Count available scans.
* Inspect nodule annotations.
* Identify missing/corrupted files.

## Dataset Analysis

Generate:

* scan count,
* annotation count,
* slice statistics,
* voxel-spacing statistics,
* scan-size statistics,
* nodule-size distribution,
* nodules-per-scan distribution.

## Important Rule

No model training should begin until the dataset has been validated.

## Deliverables

```text
data/metadata/
data/splits/
notebooks/01_data_exploration.ipynb
```

## Completion Criteria

```text
[ ] Dataset loads successfully
[ ] All relevant scans identified
[ ] Annotations validated
[ ] Coordinate system understood
[ ] Dataset statistics generated
[ ] Data integrity issues documented
[ ] Split strategy defined
```

---

# 6. Phase 3 — CT Preprocessing Pipeline

## Objective

Create a reproducible medical-image preprocessing pipeline.

## Tasks

Implement and validate:

1. CT loading.
2. HU conversion.
3. Intensity clipping/windowing.
4. Normalization.
5. Resampling where required.
6. Spatial coordinate handling.
7. Lung-region processing if used.
8. Slice extraction.
9. Patch/candidate generation.

## Important Requirement

Every transformation must preserve the correspondence between image data and nodule annotations.

## Deliverables

```text
src/preprocessing/
notebooks/02_preprocessing_validation.ipynb
configs/preprocessing.yaml
```

## Validation

Visually verify:

* CT slices,
* nodules,
* preprocessing output,
* coordinate alignment,
* patch extraction.

## Completion Criteria

The preprocessing pipeline must produce valid model-ready inputs from a raw LUNA16 scan.

---

# 7. Phase 4 — 2D Baseline

## Objective

Establish a simple baseline before introducing 2.5D and Transformer components.

## Purpose

The baseline provides a reference against which later improvements can be measured.

## Tasks

* Define a simple 2D representation.
* Implement baseline dataset.
* Implement baseline model.
* Implement baseline training.
* Implement baseline detection evaluation.
* Establish initial metrics.

## Evaluation

At minimum:

* sensitivity,
* false positives per scan,
* FROC,
* CPM where applicable.

## Deliverables

```text
src/models/
src/training/
src/evaluation/
experiments/baseline/
```

## Completion Criteria

A complete end-to-end baseline must run:

```text
Dataset
 ↓
Training
 ↓
Checkpoint
 ↓
Prediction
 ↓
Evaluation
```

---

# 8. Phase 5 — 2.5D Data Pipeline

## Objective

Introduce the project's primary CT representation.

## Tasks

Implement configurable adjacent-slice input.

Default (decision recorded in `memory.md`): 5 slices at 1 mm spacing.

```text
z-2
z-1
 z
z+1
z+2
```

`input_slices` and `slice_stride` remain configurable. In Phase 6, also compare 1 vs 3 vs 5 slices (and a wider stride) with all else fixed.

## Tasks

* Generate 2.5D inputs.
* Verify slice ordering.
* Verify boundary handling.
* Verify annotation alignment.
* Visualize 2.5D samples.
* Compare 2D and 2.5D inputs (including 1 vs 3 vs 5 slices).

## Deliverables

```text
src/preprocessing/slice_context.py
notebooks/
configs/
```

## Completion Criteria

The system must reliably generate valid 2.5D samples from LUNA16 scans.

---

# 9. Phase 6 — 2.5D CNN Baseline

## Objective

Determine the performance of the 2.5D representation before adding Transformer-based processing.

## Tasks

* Implement 2.5D CNN model.
* Train using the same dataset split.
* Maintain comparable training conditions.
* Evaluate detection performance.
* Measure computational characteristics.

## Compare

```text
2D baseline
        VS
2.5D CNN
```

## Metrics

* FROC,
* CPM,
* sensitivity,
* false positives per scan,
* parameters,
* FLOPs/MACs,
* memory,
* latency where practical.

## Completion Criteria

A reproducible 2.5D CNN baseline is available.

---

# 10. Phase 7 — CNN–Transformer Architecture

## Objective

Investigate Transformer-based contextual modeling on top of the 2.5D representation.

## Tasks

* Define CNN feature extractor.
* Define CNN-to-Transformer interface.
* Define token/feature representation.
* Implement Transformer module.
* Integrate Transformer with CNN features.
* Train the model.
* Evaluate against the 2.5D CNN baseline.

## Comparison

```text
2.5D CNN
    VS
2.5D CNN + Transformer
```

## Important

Do not assume the Transformer improves performance.

The experiment must determine its actual contribution.

## Deliverables

```text
src/models/backbone/
experiments/cnn_transformer/
```

---

# 11. Phase 8 — Attention Module

## Objective

Investigate whether an attention mechanism improves feature representation and detection performance.

## Tasks

* Define attention mechanism.
* Implement attention module.
* Integrate it into the architecture.
* Validate tensor dimensions.
* Train the model.
* Compare with the architecture without attention.

## Comparison

```text
CNN–Transformer
       VS
CNN–Transformer + Attention
```

## Required Analysis

Determine:

* performance change,
* parameter change,
* computational change,
* memory change.

## Completion Criteria

Attention contribution is experimentally measured.

---

# 12. Phase 9 — Progressive Pruning

## Objective

Investigate progressive model pruning as a method for reducing model complexity.

## Tasks

1. Establish unpruned reference model.
2. Define pruning strategy.
3. Define pruning schedule.
4. Apply pruning.
5. Recover/fine-tune.
6. Evaluate on the validation split.
7. Repeat for additional pruning levels until the stopping criterion is reached.
8. Evaluate the final (pruned) model once on the held-out test split.

Conceptually:

```text
Train
 ↓
Prune
 ↓
Fine-tune
 ↓
Evaluate
 ↓
Prune further
 ↓
Fine-tune
 ↓
Evaluate
```

## Measurements

Record:

* sparsity,
* parameters,
* model size,
* FLOPs/MACs,
* latency,
* memory,
* FROC,
* CPM,
* sensitivity.

## Important

Pruning must not be considered successful solely because sparsity increased.

## Completion Criteria

A complete pruning experiment with before/after measurements exists.

---

# 13. Phase 10 — Complete Model

## Objective

Integrate the experimentally validated components into the complete research model.

Conceptually:

```text
2.5D Input
     ↓
CNN
     ↓
Transformer
     ↓
Attention
     ↓
Multi-Scale Processing
     ↓
Detection Head
     ↓
Progressive Pruning
     ↓
Final (Pruned) Model
```

The exact final architecture depends on previous experimental results.

## Tasks

* Integrate components.
* Clean configuration.
* Remove unnecessary experimental code from the main path.
* Train final candidate model.
* Save complete configuration.
* Save checkpoint.
* Evaluate.

---

# 14. Phase 11 — Ablation Study

## Objective

Determine the contribution of individual research components.

## Minimum Experimental Structure

```text
E0 → 2D CNN
E1 → 2.5D CNN
E2 → 2.5D CNN + Transformer
E3 → E2 + Attention
E4 → E3 + Progressive Pruning (final model)
```

Additional experiments should be added where necessary.

## Required Table

A final ablation table should contain at least:

| Experiment | 2.5D | Transformer | Attention | Pruning | CPM | Sensitivity | Params | FLOPs |
| ---------- | ---: | ----------: | --------: | ------: | --: | ----------: | -----: | ----: |

Values will be filled only after experiments are completed.

## Completion Criteria

Each major component has a measurable comparison.

---

# 15. Phase 12 — Efficiency Evaluation

## Objective

Determine the computational characteristics of the proposed system.

## Measure

```text
Parameter Count
FLOPs / MACs
Model Size
Peak Memory
Inference Latency
Throughput
Sparsity
```

## Compare

At minimum:

```text
Baseline
2.5D CNN
CNN–Transformer
Attention Model
Pruned Model
Final Model
```

## Important

All measurements must use documented and comparable conditions.

---

# 16. Phase 13 — Error Analysis

## Objective

Understand where the model succeeds and fails.

## Analyze

* missed nodules,
* false positives,
* small nodules,
* larger nodules,
* difficult anatomical regions,
* low-contrast nodules,
* confusing structures,
* confidence distributions.

## Visual Analysis

Generate representative examples of:

```text
True Positive
False Positive
False Negative
Hard Case
```

## Purpose

Error analysis should help explain model behavior rather than simply provide additional figures.

---

# 17. Phase 14 — Final Experiments

## Objective

Run the final controlled experiments after the architecture and methodology have stabilized.

## Tasks

* Freeze experiment configuration.
* Verify dataset split.
* Verify preprocessing.
* Verify random seed.
* Train final models.
* Evaluate all required metrics.
* Run efficiency measurements.
* Save checkpoints.
* Save configurations.
* Preserve logs.

## Important

No experimental methodology should be changed after final evaluation without recording the change.

---

# 18. Phase 15 — Research Analysis

## Objective

Convert experimental results into scientifically defensible conclusions.

## Analyze

### Detection

* Did performance improve?
* At which false-positive rates?
* Does the improvement persist across relevant operating points?

### Architecture

* Did the Transformer contribute?
* Did attention contribute?

### Pruning

* How much complexity was removed?
* How much detection performance changed?
* Did actual latency change?

### Overall

* What component contributed most?
* What combinations were beneficial?
* What failed?
* What limitations remain?

---

# 19. Phase 16 — Final Documentation

## Objective

Prepare the complete academic and technical documentation.

## Documentation should include

* problem statement,
* motivation,
* related work,
* research gap,
* methodology,
* architecture,
* preprocessing,
* training methodology,
* pruning methodology,
* loss formulation,
* experiments,
* evaluation methodology,
* results,
* ablation study,
* efficiency analysis,
* error analysis,
* limitations,
* conclusion,
* future work.

## Outputs

Potential final artifacts:

```text
Final Report
Research Paper / Paper Draft
Presentation
Architecture Diagrams
Result Tables
Result Figures
Reproducibility Instructions
```

---

# 20. Phase 17 — Reproducibility & Finalization

## Objective

Verify that the completed project can be reconstructed from the repository.

## Final Checklist

```text
[ ] Environment documented
[ ] Dataset setup documented
[ ] Dataset split documented
[ ] Preprocessing documented
[ ] Model configuration saved
[ ] Training configuration saved
[ ] Checkpoints available
[ ] Evaluation scripts available
[ ] Metrics saved
[ ] Results saved
[ ] Ablation results saved
[ ] Efficiency results saved
[ ] Figures saved
[ ] Tables saved
[ ] Seeds documented
[ ] Dependencies documented
[ ] README completed
[ ] Final report completed
[ ] Known limitations documented
```

---

# 21. Phase Dependencies

The phases should generally follow this dependency chain:

```text
Phase 0
  ↓
Phase 1
  ↓
Phase 2
  ↓
Phase 3
  ↓
Phase 4
  ↓
Phase 5
  ↓
Phase 6
  ↓
Phase 7
  ↓
Phase 8
  ↓
Phase 9
  ↓
Phase 10
  ↓
Phase 11
  ↓
Phase 12
  ↓
Phase 13
  ↓
Phase 14
  ↓
Phase 15
  ↓
Phase 16
  ↓
Phase 17
```

Some analysis phases may overlap, but later research conclusions should not be finalized before the required experiments are complete.

---

# 22. Phase Completion Protocol

For every phase, use:

```text
PLAN
  ↓
IMPLEMENT
  ↓
TEST
  ↓
VALIDATE
  ↓
DOCUMENT
  ↓
UPDATE MEMORY
  ↓
MOVE TO NEXT PHASE
```

A phase should not be marked complete merely because the implementation runs once.

---

# 23. Phase Status Tracking

`memory.md` should maintain the current project state.

Example:

```text
Current Phase:
Phase 3 — CT Preprocessing

Completed:
Phase 0
Phase 1
Phase 2

In Progress:
Phase 3

Next:
Phase 4 — 2D Baseline

Blocked:
None

Important Decisions:
2.5D is the primary input representation.
CNN–Transformer is the primary architectural direction.
```

This is only an example. Actual status should be updated as the project progresses.

---

# 24. Research Decision Gates

Certain phases require a decision before proceeding.

## Gate 1 — After Dataset Validation

Determine whether the dataset pipeline is reliable.

```text
Valid → Continue
Invalid → Fix dataset pipeline
```

## Gate 2 — After 2D/2.5D Baselines

Determine whether the 2.5D representation is technically viable.

```text
Viable → Continue
Problems → Investigate / modify representation
```

## Gate 3 — After CNN–Transformer Experiment

Determine whether the selected Transformer integration is useful enough to continue.

## Gate 4 — After Attention Experiment

Determine whether the attention mechanism provides sufficient experimental value.

## Gate 5 — After Pruning

Determine whether pruning provides a useful performance/efficiency trade-off.

These gates do **not** mean that unsuccessful components must be removed from the research report. Negative results remain valid research findings.

---

# 25. Parallel Work That Can Be Performed

Some activities can proceed alongside model development.

```text
Literature Review
        │
        ├──────────────┐
        ▼              ▼
Dataset Work       Model Work
        │              │
        └──────┬───────┘
               ▼
          Experiments
               │
               ▼
          Analysis
```

Literature review should continue throughout the project because new relevant research may affect how results are interpreted.

---

# 26. Final Project Lifecycle

The complete lifecycle is:

```text
Research Question
      ↓
Literature Review
      ↓
System Design
      ↓
Dataset Validation
      ↓
Baseline
      ↓
Representation Study
      ↓
Architecture Study
      ↓
Attention Study
      ↓
Pruning Study
      ↓
Complete Model
      ↓
Ablation
      ↓
Efficiency
      ↓
Error Analysis
      ↓
Final Experiments
      ↓
Research Conclusions
      ↓
Documentation
      ↓
Reproducible Release
```

---

# 27. Final Phase Principle

The project should progress from **simple and measurable → increasingly complex → experimentally validated → optimized → analyzed**.

The final model must be the result of the research process, not an architecture whose superiority is assumed before experimentation.

The final research story should therefore be based on:

> **What was implemented → what was measured → what changed → what worked → what did not → why the results matter → what limitations remain.**
