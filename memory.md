# Project Memory

**File:** `memory.md`
**Project:** 2.5D CNN–Transformer Framework for Lung Nodule Detection
**Version:** 1.1
**Status:** Initial (documentation synced with `Project_Report.pdf`)

---

# Purpose

This file is the **persistent project-state memory** for the AI working on this project.

It should contain only concise information needed to understand the **current state of the project** and continue work correctly across sessions.

The AI should update this file whenever a significant project decision, implementation milestone, experiment result, issue, or next step changes the project state.

---

# Current State

**Phase:** Not started

**Completed:** None

**In Progress:** None

**Next Task:** Begin Phase 1 — Environment & Repository Setup

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
