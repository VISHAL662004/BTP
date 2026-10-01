# Design System

**File:** `design.md`
**Project:** 2.5D CNN–Transformer Framework for Lung Nodule Detection
**Version:** 1.1
**Status:** Revised Design Specification (synced with `Project_Report.pdf`)

---

# 1. Purpose

This document defines the visual and presentation standards for the project.

It covers:

* documentation appearance,
* typography,
* colors,
* diagrams,
* plots,
* tables,
* experiment-result presentation,
* metrics,
* model architecture visualization,
* notebook presentation,
* research-report consistency.

The goal is to maintain a **clean, technical, research-oriented visual language** throughout the project.

The design should prioritize:

> **Clarity → Readability → Scientific accuracy → Consistency**

Visual styling must never distort or exaggerate experimental results.

---

# 2. Design Philosophy

The project is a technical research project involving:

* medical imaging,
* deep learning,
* lung nodule detection,
* CNNs,
* Transformers,
* attention,
* pruning,
* experimental evaluation.

Therefore, the visual design should feel:

* scientific,
* precise,
* modern,
* minimal,
* professional,
* data-oriented.

Avoid excessive decorative elements.

The design should make it easy to understand:

```text
What is the input?
        ↓
What happens to it?
        ↓
What is the model doing?
        ↓
What is being measured?
        ↓
What did the experiment show?
```

---

# 3. Color System

The project should use a restrained color palette.

## 3.1 Primary Colors

| Purpose          | Color     | Hex       |
| ---------------- | --------- | --------- |
| Primary          | Deep Blue | `#2563EB` |
| Secondary        | Teal      | `#0F766E` |
| Dark Text        | Slate     | `#1E293B` |
| Secondary Text   | Gray      | `#64748B` |
| Background       | White     | `#FFFFFF` |
| Light Background | Slate 50  | `#F8FAFC` |
| Border           | Slate 200 | `#E2E8F0` |

These colors should be used consistently across diagrams, documentation, dashboards, and presentation materials.

---

# 4. Semantic Result Colors

Result colors should communicate meaning consistently.

| Meaning                | Color | Hex       |
| ---------------------- | ----- | --------- |
| Improvement / positive | Green | `#16A34A` |
| Warning / trade-off    | Amber | `#D97706` |
| Error / failure        | Red   | `#DC2626` |
| Neutral                | Gray  | `#64748B` |
| Baseline/reference     | Blue  | `#2563EB` |

These colors should **not** be used to imply a conclusion that is unsupported by the numbers.

For example, if Model A has a higher CPM than Model B, the higher value may be highlighted as a numerical difference. The visualization must still provide the actual values.

---

# 5. Color-Blind Accessibility

Color must not be the only mechanism used to distinguish results.

For example, instead of:

```text
Green = Model A
Red = Model B
```

also use:

* labels,
* markers,
* line styles,
* patterns,
* direct annotations.

Important information must remain understandable in grayscale.

---

# 6. Typography

## 6.1 Primary Font

Use:

**Inter**

for general project interfaces, dashboards, diagrams, and presentation materials where available.

Fallback:

```text
Arial
Helvetica
sans-serif
```

---

## 6.2 Code Font

Use:

**JetBrains Mono**

for:

* code,
* file paths,
* commands,
* configuration values,
* experiment IDs.

Fallback:

```text
Menlo
Monaco
Consolas
monospace
```

---

## 6.3 Research Document Font

For formal reports or papers, typography may follow the required academic template.

If no external template is specified:

```text
Body:
Inter / Arial

Headings:
Inter / Arial

Code:
JetBrains Mono
```

If a university or publication requires another font, the required template takes priority over this document.

---

# 7. Typography Hierarchy

Use a clear hierarchy.

```text
H1
Project / Major Section

H2
Major Topic

H3
Subtopic

Body
Normal explanatory text

Caption
Figure/table explanation

Code
Implementation/configuration
```

Suggested sizes for digital documentation:

| Element |     Size |
| ------- | -------: |
| H1      | 30–36 px |
| H2      | 24–28 px |
| H3      | 18–22 px |
| Body    | 15–17 px |
| Caption | 12–14 px |
| Code    | 13–15 px |

For academic documents, use the publication/university formatting requirements instead.

---

# 8. Markdown Documentation Style

All project Markdown files should use:

* clear headings,
* short paragraphs,
* tables where useful,
* bullet lists,
* code blocks,
* diagrams,
* checklists.

Avoid extremely long unbroken paragraphs.

Preferred:

```markdown
## Dataset Validation

The dataset must be validated before model development.

### Required Checks

- Scan availability
- Annotation availability
- Coordinate consistency
- Voxel spacing
```

Avoid:

```markdown
## Dataset Validation

The dataset must be validated and then...
```

with multiple unrelated concepts in one paragraph.

---

# 9. Documentation Callouts

Use consistent callouts for important information.

### Important

```text
IMPORTANT
This preprocessing step must preserve annotation coordinates.
```

### Research Decision

```text
RESEARCH DECISION
The current experiment uses five adjacent CT slices (z-2 … z+2, 1 mm apart).
```

### Warning

```text
WARNING
Do not split individual slices from the same scan across train and test sets.
```

### Experimental

```text
EXPERIMENTAL
The exact Transformer configuration has not yet been finalized.
```

These labels should be used sparingly.

---

# 10. Architecture Diagram Style

Architecture diagrams should be simple and directional.

Preferred structure:

```text
┌──────────────────────┐
│      LUNA16 CT       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Preprocessing      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    2.5D Formation    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    CNN Features      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Transformer Features │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      Attention       │
└──────────────────────┘
```

---

# 11. Architecture Diagram Rules

Every architecture diagram should:

* flow primarily from top to bottom or left to right,
* use consistent box sizes,
* use consistent terminology,
* use arrows to indicate data flow,
* avoid unnecessary decorative elements,
* clearly distinguish data, processing, and output.

Do not use ambiguous arrows.

For example:

```text
→
```

should represent a meaningful flow.

---

# 12. Research Component Diagram

Research-specific components should be visually distinguishable from ordinary processing components.

Conceptually:

```text
Input
  ↓
Backbone
  ↓
Transformer
  ↓
Attention
  ↓
Detection Head
```

Training-specific mechanisms:

```text
Detection Loss
(L_total = L_det)
       ↓
 Optimization
       ↓
 Progressive Pruning
       ↓
 Fine-tuning
```

This distinction prevents pruning from being incorrectly interpreted as ordinary inference layers.

---

# 13. Model Visualization

When showing the final architecture, include two views when appropriate.

## Inference View

```text
Input
 ↓
Preprocessing
 ↓
2.5D Representation
 ↓
CNN
 ↓
Transformer
 ↓
Attention
 ↓
Feature Fusion
 ↓
Detection Head
 ↓
Prediction
```

## Training View

```text
Prediction
    ↓
Detection Loss (L_total = L_det)
    ↓
Optimization
    ↓
Progressive Pruning
    ↓
Recovery / Fine-tuning
```

This avoids mixing training-only operations with inference operations.

---

# 14. Medical Image Visualization

CT images should be displayed using appropriate grayscale/windowing conventions.

When showing CT examples, clearly indicate:

* slice number where relevant,
* windowing information where relevant,
* coordinate information where relevant,
* nodule annotation,
* predicted detection where applicable.

Annotations should be visually distinguishable from the underlying CT image.

---

# 15. CT Comparison Figures

When comparing preprocessing stages, use the same anatomical region.

Example:

```text
Raw CT
   │
   ├── HU Conversion
   │
   ├── Windowing
   │
   ├── Normalization
   │
   └── Final Input
```

Do not compare unrelated slices and present them as a preprocessing comparison.

---

# 16. Prediction Visualization

Detection results should distinguish:

### Ground Truth

Use:

**Green outline/marker**

### Model Prediction

Use:

**Blue outline/marker**

### False Positive

Use:

**Red outline/marker**

### Missed Detection

Use:

**Orange marker or annotation**

The exact colors may be adjusted for accessibility, but the semantic mapping must remain consistent.

A legend should be included whenever the meaning is not immediately obvious.

---

# 17. Metric Presentation

Metrics should always include:

* metric name,
* value,
* dataset split,
* experiment ID where relevant.

Example:

```text
Validation CPM: 0.812
Experiment: EXP-004-attention
```

Avoid presenting:

```text
CPM = 81.2%
```

if the metric is actually represented on a 0–1 scale unless the conversion is explicitly defined.

---

# 18. Result Tables

Result tables should be compact and easy to compare.

Example:

| Model           | CPM ↑ | Sensitivity ↑ | FP/Scan ↓ | Params ↓ | FLOPs ↓ |
| --------------- | ----: | ------------: | --------: | -------: | ------: |
| Baseline        |     — |             — |         — |        — |       — |
| 2.5D            |     — |             — |         — |        — |       — |
| CNN–Transformer |     — |             — |         — |        — |       — |
| + Attention     |     — |             — |         — |        — |       — |
| + Pruning       |     — |             — |         — |        — |       — |

Arrows indicate the desired direction of the metric, not a claim about which model currently performs best.

---

# 19. Decimal Formatting

Use consistent precision.

Recommended:

```text
CPM:
3 decimal places

Sensitivity:
3 decimal places

FLOPs:
2–3 significant decimal places

Parameters:
K / M where appropriate

Latency:
milliseconds with consistent precision
```

Do not mix:

```text
0.81
81.2%
0.8123
```

for the same metric without explicitly explaining the representation.

---

# 20. Result Highlighting

Use restrained highlighting.

Example:

| Model    |       CPM |   Params |
| -------- | --------: | -------: |
| Baseline |     0.721 |     4.2M |
| Proposed | **0.758** | **2.1M** |

Bold may be used to highlight the value being discussed.

However, do not automatically bold every maximum/minimum if doing so could imply an overall ranking that the metric alone does not establish.

---

# 21. Ablation Visualization

Ablation results should show the incremental effect of components.

Example:

```text
2D CNN
   ↓
2.5D CNN
   ↓
+ Transformer
   ↓
+ Attention
   ↓
+ Pruning
```

A corresponding table should provide the exact numerical values.

The visualization should not replace the numerical results.

---

# 22. Performance Curves

For FROC results, use:

* false positives per scan on the X-axis,
* sensitivity on the Y-axis.

Axes must include units.

Example:

```text
Sensitivity
1.0 |                         ●
    |                    ●
    |               ●
    |          ●
    |     ●
0.0 |____________________________
       0.125 0.25 0.5 1 2 4 8
           False Positives / Scan
```

The exact plotting implementation may vary.

---

# 23. Training Curves

Training visualizations should commonly include:

* training loss,
* validation loss,
* relevant detection metric,
* learning rate.

Where appropriate:

```text
Epoch → X-axis
Metric → Y-axis
```

Training and validation curves should use consistent labels.

---

# 24. Loss Visualization

The training objective is `L_total = L_det`, where
`L_det = λ_cls L_focal + λ_loc L_loc`.

If the individual terms are plotted:

```text
Total Loss
Focal (classification/objectness) Loss
Localization Loss
```

should be distinguishable, and the values of `λ_cls` and `λ_loc` must be displayed or documented.

---

# 25. Pruning Visualization

Pruning results should visualize the relationship between:

```text
Sparsity
    ↕
Detection Performance
```

and where possible:

```text
Sparsity
    ↕
Latency
```

The visualization should make it possible to identify the trade-off rather than presenting pruning as inherently beneficial.

---

# 26. Efficiency Visualization

Useful visualizations include:

### Parameters vs CPM

```text
Parameter Count
        ↕
       CPM
```

### FLOPs vs CPM

```text
FLOPs
  ↕
CPM
```

### Latency vs Sensitivity

```text
Latency
  ↕
Sensitivity
```

These plots should use actual measured values.

---

# 27. Experiment Naming in Figures

Every saved figure should have a meaningful filename.

Example:

```text
results/figures/
├── exp001_training_loss.png
├── exp002_25d_vs_2d.png
├── exp003_transformer_ablation.png
├── exp004_attention_ablation.png
└── exp005_pruning_tradeoff.png
```

Avoid:

```text
figure1.png
newplot.png
final_final.png
```

---

# 28. File Naming Convention

Use:

```text
lowercase_snake_case
```

Examples:

```text
training_config.yaml
detection_metrics.csv
pruning_results.csv
pruning_ablation.png
```

Experiment identifiers should remain explicit where useful.

---

# 29. Notebook Design

Jupyter notebooks should follow a consistent structure:

```text
1. Objective
2. Configuration
3. Imports
4. Dataset
5. Preprocessing
6. Visualization
7. Experiment
8. Results
9. Analysis
10. Conclusion
```

Each notebook should clearly state what experiment or investigation it performs.

---

# 30. Notebook Rules

Notebooks should not contain hidden state dependencies.

A notebook should ideally execute from:

```text
Restart Kernel
      ↓
Run All
      ↓
Successful Result
```

Cells should be logically ordered.

Temporary debugging code should be removed before the notebook is considered final.

---

# 31. Code Presentation

Code shown in documentation should:

* be concise,
* use syntax highlighting,
* show only relevant sections,
* include file paths where useful.

Example:

```python
from src.models.detector import Detector

model = Detector(config)
```

Do not place hundreds of lines of implementation inside Markdown documentation when a file reference is sufficient.

---

# 32. Configuration Presentation

Configuration examples should use YAML where appropriate:

```yaml
model:
  attention:
    enabled: true

training:
  batch_size: 4
```

Configuration values should be clearly separated from explanatory text.

---

# 33. Research Status Indicators

Use a consistent status system:

| Status         | Meaning                                   |
| -------------- | ----------------------------------------- |
| `PLANNED`      | Not implemented                           |
| `IN_PROGRESS`  | Currently being developed                 |
| `IMPLEMENTED`  | Code exists                               |
| `VALIDATED`    | Tested successfully                       |
| `EXPERIMENTAL` | Being experimentally evaluated            |
| `COMPLETED`    | Final required work finished              |
| `BLOCKED`      | Cannot proceed due to an unresolved issue |

Do not mark a component `COMPLETED` merely because the code exists.

---

# 34. Dashboard / Summary Design

If a project dashboard is created, it should prioritize:

```text
Project Status
       ↓
Current Phase
       ↓
Latest Experiment
       ↓
Detection Results
       ↓
Efficiency Results
       ↓
Known Issues
       ↓
Next Task
```

Avoid turning the dashboard into a decorative interface.

---

# 35. Final Research Presentation

The final presentation should follow approximately:

```text
Problem
   ↓
Motivation
   ↓
Research Gap
   ↓
Proposed Approach
   ↓
Architecture
   ↓
Experimental Setup
   ↓
Ablation
   ↓
Detection Results
   ↓
Efficiency Results
   ↓
Error Analysis
   ↓
Limitations
   ↓
Conclusion
```

The presentation should show evidence before conclusions.

---

# 36. Research Result Integrity

Visual design must never:

* hide poor results,
* crop plots misleadingly,
* manipulate axes,
* exaggerate differences,
* omit relevant baselines,
* use color to imply unsupported conclusions.

The underlying numerical result must always remain available.

---

# 37. Final Design Principle

The visual identity of the project should communicate:

> **Medical imaging + deep learning + rigorous experimentation + computational efficiency.**

Every visual element should serve one of three purposes:

```text
Explain
Compare
Validate
```

If an element does not improve understanding, it should generally be removed.
