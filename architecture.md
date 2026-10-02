# Architecture Document

**File:** `architecture.md`
**Project:** 2.5D CNN–Transformer Framework for Lung Nodule Detection
**Version:** 1.1
**Status:** Revised Architecture Specification (synced with `Project_Report.pdf`)

---

# 1. Purpose

This document defines the **technical architecture, software structure, module boundaries, technology stack, data flow, and implementation organization** of the lung nodule detection research project.

This document intentionally defines the architecture at a level that allows individual model components to be experimentally changed.

The exact CNN backbone, Transformer configuration, attention implementation, and pruning strategy should **not be hard-coded into the project architecture at this stage**. They are research variables and will be finalized through experimentation.

---

# 2. High-Level System Architecture

The system is organized into the following major stages:

```text
                         ┌─────────────────────┐
                         │     LUNA16 Data     │
                         │   CT + Annotations  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Validation   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ CT Preprocessing    │
                         │ HU / Normalization  │
                         │ Resampling / ROI    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   2.5D Formation    │
                         │ z-2 ... z+2 ...    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Candidate / Patch   │
                         │ Generation          │
                         └──────────┬──────────┘
                                    │
                                    ▼
              ┌────────────────────────────────────────┐
              │            Feature Extraction          │
              │                                        │
              │   ┌────────────┐    ┌──────────────┐  │
              │   │    CNN     │───►│ Transformer  │  │
              │   └────────────┘    └──────────────┘  │
              └───────────────────────┬────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────┐
                         │     Attention       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Multi-Scale Feature │
                         │ Processing / Fusion │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Detection Head      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Detection Outputs   │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │ Detection       │             │ Efficiency      │
          │ Evaluation      │             │ Evaluation      │
          └─────────────────┘             └─────────────────┘
```

---

# 3. Research Model Architecture

The conceptual model is:

```text
2.5D CT Input
      │
      ▼
┌──────────────────┐
│ CNN Feature      │
│ Extraction       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Transformer      │
│ Feature Modeling │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Attention        │
│ Mechanism        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Multi-Scale      │
│ Feature Fusion   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Detection Head   │
└────────┬─────────┘
         │
         ▼
   Nodule Outputs
```

The architecture should be implemented using modular components so that individual components can be enabled or disabled.

---

# 4. Architectural Principle

The project follows a **modular research architecture**.

Each major research component must have a clear interface.

```text
Data
  ↓
Preprocessing
  ↓
Representation
  ↓
Backbone
  ↓
Context Modeling
  ↓
Attention
  ↓
Feature Fusion
  ↓
Detection
  ↓
Loss
  ↓
Optimization
  ↓
Evaluation
```

This allows controlled experiments such as:

```text
Baseline
    ↓
+ 2.5D
    ↓
+ Transformer
    ↓
+ Attention
    ↓
+ Pruning
```

without rewriting the entire codebase.

---

# 5. Proposed Repository Structure

The project should use the following initial structure:

```text
lung-nodule-detection/
│
├── README.md
├── prd.md
├── architecture.md
├── rules.md
├── phases.md
├── design.md
├── memory.md
│
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
│
├── configs/
│   ├── base.yaml
│   ├── dataset.yaml
│   ├── preprocessing.yaml
│   ├── model.yaml
│   ├── training.yaml
│   ├── pruning.yaml
│   └── experiments/
│       ├── baseline.yaml
│       ├── cnn_transformer.yaml
│       ├── attention.yaml
│       └── pruning.yaml
│
├── data/
│   ├── raw/
│   ├── annotations/
│   ├── metadata/
│   ├── processed/
│   ├── candidates/
│   └── splits/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_baseline_analysis.ipynb
│   ├── 04_25d_pipeline_validation.ipynb
│   ├── 05_model_analysis.ipynb
│   └── 06_results_analysis.ipynb
│
├── src/
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── luna16.py
│   │   ├── annotations.py
│   │   ├── splits.py
│   │   └── validation.py
│   │
│   ├── preprocessing/
│   │   ├── hu.py
│   │   ├── normalization.py
│   │   ├── resampling.py
│   │   ├── lung_region.py
│   │   ├── slice_context.py
│   │   └── patches.py
│   │
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── cnn.py
│   │   │   └── transformer.py
│   │   │
│   │   ├── attention/
│   │   │   └── attention.py
│   │   │
│   │   ├── fusion/
│   │   │   └── multiscale.py
│   │   │
│   │   ├── heads/
│   │   │   └── detection.py
│   │   │
│   │   └── detector.py
│   │
│   ├── losses/
│   │   └── detection_loss.py
│   │
│   ├── pruning/
│   │   ├── scheduler.py
│   │   ├── strategy.py
│   │   ├── masks.py
│   │   └── recovery.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── train.py
│   │   ├── validation.py
│   │   ├── checkpointing.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── detection_metrics.py
│   │   ├── froc.py
│   │   ├── cpm.py
│   │   ├── error_analysis.py
│   │   └── evaluator.py
│   │
│   ├── efficiency/
│   │   ├── parameters.py
│   │   ├── flops.py
│   │   ├── latency.py
│   │   ├── memory.py
│   │   └── sparsity.py
│   │
│   ├── experiments/
│   │   ├── runner.py
│   │   ├── ablation.py
│   │   └── registry.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       ├── config.py
│       ├── visualization.py
│       └── io.py
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_pruning.py
│   └── test_evaluation.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── prune.py
│   └── benchmark.py
│
├── experiments/
│   ├── baseline/
│   ├── cnn_transformer/
│   ├── attention/
│   ├── pruning/
│   └── final/
│
├── checkpoints/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── tables/
│   ├── predictions/
│   └── efficiency/
│
└── logs/
```

---

# 6. Directory Responsibilities

## 6.1 `configs/`

Contains experiment configuration.

Configuration should control:

* dataset parameters,
* preprocessing,
* model architecture,
* training,
* loss weights,
* pruning,
* evaluation.

Model code should not contain experiment-specific constants wherever configuration is more appropriate.

---

## 6.2 `data/`

Contains dataset-related files.

```text
data/
├── raw/
├── annotations/
├── metadata/
├── processed/
├── candidates/
└── splits/
```

### `raw/`

Original CT data.

### `annotations/`

LUNA16 annotation files.

### `metadata/`

Dataset metadata and generated indexes.

### `processed/`

Preprocessed representations.

### `candidates/`

Generated candidate regions/patches.

### `splits/`

Reproducible train/validation/test split definitions.

Raw data should never be modified destructively.

---

# 7. Source Code Architecture

## 7.1 Data Layer

```text
src/data/
```

Responsible for:

* loading scans,
* reading annotations,
* constructing dataset objects,
* managing splits,
* validating dataset consistency.

The data layer should not contain model-specific logic.

---

# 8. Preprocessing Layer

```text
src/preprocessing/
```

Responsible for:

* HU conversion,
* intensity normalization,
* resampling,
* lung-region processing,
* 2.5D slice formation,
* patch/candidate extraction.

The preprocessing pipeline should be deterministic when configured with the same parameters.

---

# 9. Model Layer

```text
src/models/
```

The model is divided into independently replaceable components.

```text
models/
├── backbone/
├── attention/
├── fusion/
├── heads/
└── detector.py
```

---

## 9.1 Backbone

The backbone extracts feature representations from the 2.5D input.

```text
backbone/
├── cnn.py
└── transformer.py
```

The exact architecture remains configurable.

---

## 9.2 Attention

```text
attention/
└── attention.py
```

Contains the attention mechanism investigated by the project.

The implementation should expose a consistent interface so that attention can be enabled or disabled for ablation experiments.

---

## 9.3 Feature Fusion

```text
fusion/
└── multiscale.py
```

Responsible for combining feature maps from different spatial scales or stages.

The exact fusion architecture is a research variable.

---

## 9.4 Detection Head

```text
heads/
└── detection.py
```

Converts learned features into detection outputs.

The detection head should remain independent from preprocessing and dataset loading.

---

# 10. Loss Architecture

```text
src/losses/
```

The training objective is the detection objective itself:

```text
Total Loss = Detection Loss
```

Loss weights (`λ_cls`, `λ_loc`) must remain configurable so that the loss formulation can be changed without restructuring the code.

---

## 10.1 Loss Function and Training Objective

The proposed model is trained with a detection loss that combines a
classification/objectness term and a localization term. No additional
regularization or auxiliary loss term is used.

### 10.1.1 Detection Loss

```text
L_det = λ_cls L_focal + λ_loc L_loc
```

where:

* `L_focal` is the classification/objectness loss.
* `L_loc` is the localization loss.
* `λ_cls` and `λ_loc` control the contribution of each component.

### 10.1.2 Focal Loss

Because the candidate set for nodule detection is heavily dominated by
easy negative (non-nodule) samples, a focal loss formulation may be used
for the classification/objectness term.

```text
L_focal = -α_t (1-p_t)^γ log(p_t)
```

where:

```text
p_t = p,     if y = 1
p_t = 1-p,   if y = 0
```

`α_t` controls class weighting and `γ` controls the down-weighting of
easy examples.

### 10.1.3 Localization Loss

The localization component measures the discrepancy between the predicted
and ground-truth nodule location.

An IoU-based formulation may be used:

```text
L_loc = 1 - IoU(B, B*)
IoU(B, B*) = |B ∩ B*| / |B ∪ B*|
```

where `B` is the predicted bounding box and `B*` is the ground-truth box.

The exact localization formulation will be finalized during
implementation based on the selected detection head.

### 10.1.4 Total Training Loss

```text
L_total = L_det = λ_cls L_focal + λ_loc L_loc
```

### 10.1.5 Notation

| Symbol  | Meaning                                            |
| ------- | -------------------------------------------------- |
| L_total | Overall training loss                              |
| L_det   | Detection loss                                     |
| L_focal | Focal classification/objectness loss               |
| L_loc   | Localization loss                                  |
| λ_cls   | Weight of the classification/objectness loss       |
| λ_loc   | Weight of the localization loss                    |
| p       | Predicted probability of the positive/nodule class |
| y       | Ground-truth class label, y ∈ {0, 1}               |
| p_t     | Probability assigned to the ground-truth class     |
| α_t     | Class-balancing factor                             |
| γ       | Focal-loss focusing parameter                      |
| B, B*   | Predicted / ground-truth bounding box              |
| IoU     | Intersection over Union                            |

### 10.1.6 Loss Ablation

Model modifications are evaluated through the single-variable ablation
progression (see `prd.md`), all with `L_total = L_det`:

1. **Baseline:** 2D CNN, then 2.5D CNN.
2. **+ Transformer**, then **+ Attention**.
3. **+ Progressive pruning:** prune → fine-tune → evaluate cycles.

---

# 11. Pruning Architecture

```text
src/pruning/
```

The pruning system is separated from the model definition.

```text
pruning/
├── scheduler.py
├── strategy.py
├── masks.py
└── recovery.py
```

### `scheduler.py`

Determines when pruning operations occur.

### `strategy.py`

Defines how parameters are selected for pruning.

### `masks.py`

Manages pruning masks.

### `recovery.py`

Handles recovery/fine-tuning after pruning.

The detector should remain usable without the pruning module so that baseline experiments can be performed.

---

# 12. Training Architecture

```text
src/training/
```

Responsible for:

* training loops,
* validation,
* optimization,
* checkpointing,
* callbacks,
* logging.

The training engine should accept different model and loss configurations.

---

# 13. Evaluation Architecture

```text
src/evaluation/
```

Responsible for evaluating model predictions.

Core components:

```text
Detection Metrics
       ↓
FROC
       ↓
CPM
       ↓
Sensitivity / FP per scan
       ↓
Error Analysis
```

Evaluation code should be separated from training code so that the same evaluation pipeline can be applied to every experiment.

---

# 14. Efficiency Architecture

```text
src/efficiency/
```

Measures computational characteristics independently of detection performance.

The module should measure:

```text
Parameters
FLOPs / MACs
Model Size
Memory
Latency
Sparsity
Throughput
```

This separation is important because a model with fewer parameters is not necessarily faster on actual hardware.

---

# 15. Experiment Architecture

```text
src/experiments/
```

Responsible for reproducible experiments.

Each experiment should have:

```text
Configuration
     ↓
Dataset
     ↓
Model
     ↓
Loss
     ↓
Training
     ↓
Evaluation
     ↓
Efficiency
     ↓
Results
```

Experiments should be identifiable using a unique experiment name or ID.

---

# 16. Results Architecture

```text
results/
├── metrics/
├── figures/
├── tables/
├── predictions/
└── efficiency/
```

### `metrics/`

Numerical evaluation results.

### `figures/`

Plots and visualizations.

### `tables/`

Publication/report-ready tables.

### `predictions/`

Saved model predictions for analysis.

### `efficiency/`

Computational measurements.

Results from different experiments must not overwrite each other.

---

# 17. Experiment Flow

Every experiment should follow approximately:

```text
                    Experiment Config
                           │
                           ▼
                    Dataset Loader
                           │
                           ▼
                    Preprocessing
                           │
                           ▼
                       Model
                           │
                           ▼
                         Loss
                           │
                           ▼
                       Training
                           │
                           ▼
                     Checkpoint
                           │
                           ▼
                      Evaluation
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Detection Metrics         Efficiency Metrics
              │                         │
              └────────────┬────────────┘
                           ▼
                    Experiment Results
                           │
                           ▼
                    Analysis / Ablation
```

---

# 18. Configuration Architecture

The project should avoid embedding experimental parameters directly into Python files.

Example:

```yaml
experiment:
  name: baseline_2_5d

data:
  input_slices: 5
  slice_stride: 1
  patch_size: 64

model:
  backbone: ...
  transformer: ...
  attention: false

training:
  batch_size: 4
  epochs: 50
  learning_rate: 0.001

pruning:
  enabled: false
```

The exact values are placeholders and should be determined during implementation.

---

# 19. Technology Stack

## 19.1 Programming Language

**Python**

Primary language for:

* preprocessing,
* model development,
* training,
* evaluation,
* experimentation,
* visualization.

---

## 19.2 Deep Learning Framework

**PyTorch**

Used for:

* neural network implementation,
* automatic differentiation,
* training,
* model checkpointing,
* tensor operations.

---

## 19.3 Hardware Acceleration

The development environment is an Apple Silicon Mac.

Where supported and stable, PyTorch's **MPS backend** may be used for hardware acceleration.

CPU execution must remain possible for debugging and validation.

---

# 20. Core Libraries

The initial technology stack may include:

| Library      | Purpose                                            |
| ------------ | -------------------------------------------------- |
| Python       | Core programming                                   |
| PyTorch      | Deep learning                                      |
| NumPy        | Numerical operations                               |
| SciPy        | Scientific computation                             |
| pandas       | Metadata/table processing                          |
| scikit-learn | General ML utilities and metrics where appropriate |
| SimpleITK    | Medical image processing                           |
| nibabel      | Medical imaging formats where required             |
| PyYAML       | Configuration                                      |
| matplotlib   | Visualization                                      |
| tqdm         | Progress bars (training / pipelines)               |
| pytest       | Testing                                            |

Additional libraries should only be introduced when they provide a clear technical requirement.

---

# 21. Development Environment

Primary development tools:

```text
Operating System: macOS
Hardware: Apple Silicon Mac
IDE: VS Code
Environment: Conda
Python: 3.11
Notebook Support: Jupyter
Version Control: Git
```

Jupyter notebooks should primarily be used for:

* exploration,
* visualization,
* debugging,
* analysis,
* experimentation.

Core reusable functionality should reside inside `src/`.

---

# 22. Notebook vs Python Architecture

The project should follow:

```text
Reusable Code
     ↓
src/
     ↓
Scripts
     ↓
Experiments
     ↓
Notebooks
```

Not:

```text
Notebook
    ↓
Everything
```

Notebooks should call reusable functions from the project rather than containing duplicated implementations.

This prevents divergence between exploratory experiments and the actual research pipeline.

---

# 23. Model Dependency Architecture

The system should maintain the following dependency direction:

```text
Data
 ↓
Preprocessing
 ↓
Models
 ↓
Losses
 ↓
Training
 ↓
Evaluation
```

Cross-dependencies should be minimized.

For example:

* preprocessing should not import the detector,
* dataset code should not import training code,
* evaluation should not modify model architecture,
* visualization should not contain training logic.

---

# 24. Checkpoint Architecture

Checkpoints should contain sufficient information to reproduce an experiment.

Where practical, checkpoints should include:

```text
Model state
Optimizer state
Scheduler state
Epoch
Experiment configuration
Best validation metric
Random seed
Pruning state
```

Checkpoint naming should identify the experiment.

Example:

```text
experiments/
└── attention_exp_003/
    ├── config.yaml
    ├── best.pt
    ├── last.pt
    ├── metrics.json
    └── training.log
```

---

# 25. Final Model Architecture

The final architecture is expected to conceptually follow:

```text
                  ┌─────────────────────┐
                  │     CT Scan         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Preprocessing     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     2.5D Input      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   CNN Backbone      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Transformer Module  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Attention Module    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Multi-Scale Fusion  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Detection Head      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Nodule Predictions  │
                  └─────────────────────┘
```

Progressive pruning operates through the **training/optimization architecture**, rather than being treated as an ordinary sequential layer:

```text
                 Model
                   │
                   ▼
            Detection Loss
          (L_total = L_det)
                   │
                   ▼
              Optimization
                   │
                   ▼
   Progressive Pruning: Prune → Fine-tune → Evaluate
        (evaluated on the validation split)
                   │
                   ▼
        Stopping criterion reached?
        NO → prune further │ YES ↓
                   ▼
        Final (pruned) model
   evaluated on the held-out test split
```

---

# 26. Architecture Flexibility

The following components are intentionally **not finalized** in this document:

* exact CNN backbone,
* exact Transformer architecture,
* number of Transformer blocks,
* embedding dimension,
* attention mechanism variant,
* feature-fusion architecture,
* detection-head architecture,
* patch size,
* number of CT slices in the 2.5D representation,
* pruning ratio,
* pruning schedule,
* loss weights and the exact localization-loss formulation.

These are experimental decisions.

The architecture must therefore be implemented so these components can be changed without restructuring the entire repository.

---

# 27. Architecture Decision Principle

The final architecture should be selected based on experimental evidence.

The selection criteria include:

```text
Detection Performance
        +
Generalization
        +
Computational Cost
        +
Memory Usage
        +
Inference Latency
        +
Model Complexity
        +
Reproducibility
```

No component should be retained solely because it appears theoretically beneficial.

---

# 28. Architecture Summary

The project uses a modular architecture centered around:

```text
LUNA16
  ↓
Medical CT Preprocessing
  ↓
2.5D Representation
  ↓
CNN Feature Extraction
  ↓
Transformer Context Modeling
  ↓
Attention
  ↓
Multi-Scale Feature Processing
  ↓
Detection Head
  ↓
Detection Evaluation
```

with **Progressive Pruning** integrated into the training and optimization pipeline.

The repository structure is designed to support **baseline development → incremental architectural modifications → ablation → efficiency analysis → final model selection** without requiring major restructuring of the codebase.
