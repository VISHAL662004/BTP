# PRD — Project Requirements Document

**File:** `prd.md`
**Project:** 2.5D CNN–Transformer Framework for Lung Nodule Detection
**Dataset:** LUNA16
**Status:** Revised Project Definition (synced with `Project_Report.pdf`)
**Version:** 1.1

---

# 1. Project Overview

## 1.1 Project Title

**A 2.5D CNN–Transformer Framework for Efficient Lung Nodule Detection with Attention and Progressive Pruning**

---

## 1.2 Project Purpose

The project aims to develop and experimentally evaluate a deep-learning-based framework for **lung nodule detection from CT scans** using the **LUNA16 dataset**.

The proposed approach will investigate the use of **2.5D CT representations**, combining information from multiple adjacent CT slices with a **CNN–Transformer architecture**.

The framework will further investigate two major components:

1. **Attention mechanisms** for emphasizing features relevant to lung nodules.
2. **Progressive model pruning** for reducing model redundancy and exploring compression effects through a prune → fine-tune → evaluate cycle.

The project is research-oriented. The objective is not simply to build a detector, but to determine whether the proposed combination of techniques can provide a useful balance between **detection performance and computational efficiency**.

---

# 2. Problem Statement

Lung nodule detection from CT scans is challenging because nodules can vary significantly in:

* size,
* shape,
* appearance,
* density,
* location,
* contrast with surrounding tissue.

Small nodules can occupy only a small portion of a CT image, while surrounding anatomical structures can produce visually similar patterns.

Traditional 2D approaches can process CT slices efficiently but may lose important information along the axial direction.

Full 3D approaches can preserve volumetric information but can require substantially more computational and memory resources.

Therefore, this project investigates a **2.5D representation** that attempts to retain inter-slice contextual information while allowing the use of architectures based on 2D feature processing.

The research additionally investigates whether attention and progressive pruning can provide a useful balance between detection performance and efficiency.

In existing literature, CNN–Transformer architectures, attention techniques, and lightweight networks have been studied largely independently; their combination within a 2.5D detection pipeline has not been sufficiently investigated.

---

# 3. Target Research

The central research direction is:

> **Can a 2.5D CNN–Transformer lung nodule detection framework, enhanced with attention and progressive pruning, achieve a useful balance between detection performance and computational efficiency on LUNA16?**

The project will investigate the individual and combined contribution of these components rather than assuming beforehand that every component improves the detector.

---

# 4. Research Objectives

## 4.1 Primary Objective

Develop and evaluate a complete lung nodule detection framework using **2.5D CT input and CNN–Transformer feature extraction**.

## 4.2 Secondary Objectives

The project will:

1. Establish a reproducible lung nodule detection baseline.
2. Investigate 2D versus 2.5D CT representations.
3. Investigate the contribution of Transformer-based contextual modeling.
4. Investigate attention mechanisms for improved feature representation.
5. Develop and evaluate a progressive pruning strategy.
6. Measure detection performance using appropriate LUNA16 metrics (FROC, CPM, sensitivity, FPs per scan).
7. Measure computational characteristics of the models (parameters, FLOPs/MACs, memory, latency, sparsity, throughput).
8. Perform controlled single-variable ablation experiments.
9. Determine which components provide measurable improvements.
10. Analyze the trade-off between detection performance and computational cost.
11. Produce a reproducible research implementation and final experimental report.

---

## 4.3 Proposed Training Objective

The training objective is the detection objective itself:

```text
L_total = L_det = λ_cls L_focal + λ_loc L_loc
```

A focal loss may be used for the classification/objectness term (the
candidate set is dominated by easy negatives) and an IoU-based loss for
localization. The exact localization formulation is finalized based on
the selected detection head.

There is no additional regularization or auxiliary loss term.

The detailed mathematical formulation is documented in `architecture.md`.

---

# 5. Target System

The final system should provide a complete pipeline:

```text
LUNA16 CT Data
      ↓
Data Validation
      ↓
CT Preprocessing
      ↓
2.5D Representation
      ↓
Candidate / Patch Generation
      ↓
CNN Feature Extraction
      ↓
Transformer-Based Feature Modeling
      ↓
Attention
      ↓
Multi-Scale Feature Processing
      ↓
Detection
      ↓
Progressive Pruning
      ↓
Final (Pruned) Model
      ↓
Evaluation (held-out test split)
      ↓
Performance + Efficiency Analysis
```

The exact implementation of individual components should remain configurable until experimental evaluation determines the appropriate design.

---

# 6. Core Features

## 6.1 LUNA16 Dataset Support

The system must support the LUNA16 dataset (888 low-dose chest CT scans from LIDC-IDRI, with nodule annotations and candidates) and its official FROC/CPM evaluation protocol.

The implementation should:

* locate CT scans correctly,
* read annotations,
* associate nodules with the appropriate scans,
* validate dataset integrity,
* maintain scan-level identity,
* prevent data leakage between dataset splits.

---

## 6.2 CT Preprocessing

The system should provide configurable preprocessing for CT data.

Potential preprocessing operations include:

* Hounsfield Unit conversion,
* intensity clipping/windowing,
* normalization,
* spatial resampling where required,
* lung-region processing where required,
* slice extraction,
* candidate generation,
* patch extraction.

Preprocessing parameters must be configurable rather than hard-coded throughout the project.

---

# 7. 2.5D Representation

The primary input representation will be **2.5D**.

Instead of processing a single CT slice independently, the system will combine information from neighboring slices.

Conceptually:

```text
Slice z-1
   +
Slice z
   +
Slice z+1
   ↓
2.5D Input
```

The exact number of neighboring slices should remain configurable.

The project should allow experimentation with different slice contexts where computationally practical.

The purpose is to investigate whether additional axial context improves detection without requiring a complete 3D processing pipeline.

---

# 8. CNN–Transformer Architecture

The main architecture will investigate the combination of:

### CNN

Used for extracting local spatial and structural features from the CT representation.

### Transformer

Used for modeling contextual relationships between extracted features.

The project should not assume a particular CNN backbone, Transformer architecture, depth, or parameter size at the beginning.

These should be selected based on:

* experimental performance,
* computational requirements,
* memory requirements,
* reproducibility,
* suitability for the dataset.

---

# 9. Attention Mechanism

The framework will investigate an attention mechanism capable of emphasizing informative features associated with lung nodules.

The attention component may operate across:

* channels,
* spatial regions,
* feature representations,
* or combinations of these.

The exact attention design will be documented separately in `architecture.md` and experimentally evaluated.

Attention is investigated through ablation; its value is not assumed. Attention alone should not be treated as the primary novelty claim.

---

# 10. Progressive Pruning

The project will investigate **progressive pruning** rather than relying exclusively on a single pruning operation.

Conceptually:

```text
Train
 ↓
Evaluate
 ↓
Prune
 ↓
Recover / Fine-tune
 ↓
Evaluate
 ↓
Prune further
 ↓
Recover / Fine-tune
 ↓
Final model
```

The objective is to investigate whether model redundancy can be reduced while maintaining acceptable detection performance.

The project must measure both:

### Detection impact

* sensitivity,
* FROC/CPM,
* false positives,
* other relevant detection metrics.

### Model impact

* parameter count,
* model size,
* computational cost,
* inference latency,
* memory usage.

The project must not claim that pruning provides practical inference acceleration merely because the number of zero-valued parameters increases.

---

# 11. Detection Requirements

The system should detect pulmonary nodules and provide suitable detection outputs.

Depending on the selected detection formulation, outputs may include:

* nodule confidence,
* candidate coordinates,
* bounding region,
* estimated nodule size,
* detection score.

The final output representation must be consistent with the chosen evaluation methodology.

---

# 12. Evaluation Requirements

Evaluation must go beyond classification accuracy.

The primary evaluation should include detection-oriented metrics appropriate for LUNA16, particularly:

* **FROC**
* **CPM / Competition Performance Metric**
* sensitivity at specified false-positive rates
* false positives per scan

Additional metrics may include:

* precision,
* recall,
* ROC-AUC where applicable,
* PR-AUC where applicable.

Classification metrics must not be presented as substitutes for proper detection evaluation.

---

# 13. Efficiency Evaluation

Since model efficiency is an important research dimension, the project should measure:

| Metric                | Purpose                        |
| --------------------- | ------------------------------ |
| Parameters            | Model complexity               |
| FLOPs / MACs          | Computational complexity       |
| Model size            | Storage requirement            |
| Peak memory           | Hardware requirement           |
| Inference latency     | Runtime performance            |
| Throughput            | Processing capability          |
| Sparsity              | Effect of pruning              |
| Detection performance | Accuracy/performance trade-off |

Efficiency measurements should be performed consistently across comparable experiments.

---

# 14. Experimental Design

The project must use controlled experiments.

The expected progression is:

```text
Experiment 0
2D CNN Baseline
        ↓
Experiment 1
2.5D CNN
        ↓
Experiment 2
2.5D CNN + Transformer
        ↓
Experiment 3
+ Attention
        ↓
Experiment 4
+ Progressive Pruning
        ↓
Final (Pruned) Model
```

This progression is a research plan rather than a guaranteed final result.

If an experiment demonstrates that a component does not improve the system, that result must be recorded rather than hidden.

---

# 15. Ablation Study Requirements

Ablation studies must determine the contribution of individual components.

At minimum, experiments should investigate:

1. 2D versus 2.5D input.
2. CNN versus CNN–Transformer architecture.
3. Without versus with attention.
4. Without versus with progressive pruning.
5. Different pruning levels where practical.

The final report should clearly distinguish:

> **what the complete system achieves**

from

> **what each individual component contributes.**

---

# 16. Dataset Splitting and Leakage Prevention

Dataset splitting must be performed at an appropriate **scan/patient level** rather than randomly splitting individual slices or patches.

The same CT scan must not unintentionally appear in multiple dataset partitions.

The project must document:

* training set,
* validation set,
* test set,
* split methodology,
* random seed,
* preprocessing applied to each partition.

Any deviation from the planned split must be recorded.

---

# 17. Reproducibility Requirements

The project must be reproducible as far as practically possible.

The implementation should record:

* Python version,
* framework versions,
* library versions,
* random seeds,
* dataset version/source,
* preprocessing parameters,
* model configuration,
* training configuration,
* optimizer,
* learning-rate schedule,
* batch size,
* number of epochs,
* checkpoint information,
* evaluation configuration.

Experiments should use configuration files or clearly defined configuration objects rather than scattered hard-coded values.

---

# 18. Hardware Target

The primary development environment is:

```text
Machine:
MacBook Pro M3
RAM:
8 GB
Operating Environment:
macOS
Development:
VS Code
Python:
3.11 (Conda environment)
Framework:
PyTorch
```

The system should therefore be designed with memory-conscious experimentation in mind.

The hardware limitation is an implementation constraint, not itself a research contribution.

---

# 19. Expected Research Output

The completed project should produce:

### Software

* reproducible preprocessing pipeline,
* dataset loader,
* 2.5D data pipeline,
* baseline detector,
* CNN–Transformer detector,
* attention module,
* pruning implementation,
* training pipeline,
* evaluation pipeline,
* efficiency-analysis tools,
* experiment tracking.

### Research Results

* baseline results,
* complete-model results,
* ablation results,
* pruning results,
* efficiency measurements,
* error analysis,
* qualitative detection examples.

### Documentation

* methodology,
* architecture,
* implementation details,
* experimental setup,
* results,
* analysis,
* limitations,
* reproducibility information.

---

# 20. Research Contribution Target

The project does not assume that any single component adds value. CNN–Transformer models, attention networks, and lightweight architectures have each been studied separately for lung nodule classification/detection; their combination in a 2.5D detection model with progressive compression is relatively understudied.

The project will investigate the combined framework — **a 2.5D CNN–Transformer detector with attention and progressive pruning** — and the individual contribution of each component through single-variable ablation on LUNA16.

The strength of the contribution must ultimately be determined from:

* existing literature,
* implementation,
* controlled experiments,
* ablation studies,
* statistical/quantitative results,
* efficiency measurements.

No novelty claim should be finalized before the relevant literature and experimental evidence have been reviewed.

---

# 21. Success Criteria

The project will be considered technically successful if it produces:

* a working LUNA16 preprocessing pipeline,
* a reproducible baseline,
* a working 2.5D detection system,
* a working CNN–Transformer model,
* an implemented attention mechanism,
* an implemented progressive pruning procedure,
* complete ablation experiments,
* proper LUNA16 detection evaluation,
* computational-efficiency measurements,
* reproducible experiment configurations,
* final research documentation.

A component should only be considered beneficial if experimental evidence demonstrates an improvement or a meaningful trade-off.

---

# 22. Non-Goals

The initial project will **not** attempt to:

* build a clinical diagnostic system,
* replace radiologists,
* provide medical diagnosis,
* claim clinical deployment readiness,
* guarantee real-world patient outcomes,
* optimize exclusively for classification accuracy,
* assume that pruning automatically produces hardware speedup,
* assume that Transformer architectures are inherently superior,
* assume that attention improves detection before experimentation,
* claim novelty solely from combining existing modules.

The system is a **research prototype for lung nodule detection**.

---

# 23. Final Project Definition

The project can be summarized as:

> **Design, implement, and experimentally evaluate a 2.5D CNN–Transformer framework for lung nodule detection on LUNA16, incorporating attention and progressive pruning, with particular emphasis on detection performance, ablation-based validation, computational efficiency, and reproducibility.**

This definition should serve as the **high-level source of truth** for the subsequent architecture, rules, phases, design, implementation, and experiment documentation.
