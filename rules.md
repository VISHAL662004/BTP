# Rules — Project Development & Research Rules

**File:** `rules.md`
**Project:** 2.5D CNN–Transformer Framework for Lung Nodule Detection
**Version:** 1.1
**Status:** Revised Project Rules (synced with `Project_Report.pdf`)

---

# 1. Purpose

This document defines the rules that must be followed throughout the development, experimentation, research, documentation, and evaluation of the project.

These rules exist to ensure that the project remains:

* scientifically valid,
* reproducible,
* modular,
* computationally practical,
* properly evaluated,
* transparent about limitations,
* and suitable for academic research.

These rules apply to both **human development decisions and AI-assisted development**.

---

# 2. Core Development Principles

The project must follow these principles:

1. **Research before implementation.**
2. **Evidence before claims.**
3. **Experiments before conclusions.**
4. **Reproducibility over convenience.**
5. **Modularity over hard-coding.**
6. **Controlled experiments over arbitrary comparisons.**
7. **Detection metrics over inappropriate classification-only metrics.**
8. **No unsupported novelty claims.**
9. **No silent changes to experimental methodology.**
10. **Every important research decision must be documented.**

---

# 3. AI Development Rules

AI tools may assist with:

* code generation,
* debugging,
* documentation,
* experiment planning,
* literature analysis,
* mathematical formulation,
* visualization,
* refactoring,
* error diagnosis,
* test generation.

However, AI-generated content must not automatically be considered correct.

Every generated implementation must be:

1. inspected,
2. tested,
3. validated against the project requirements,
4. checked against the relevant paper/source where applicable,
5. evaluated experimentally when it affects research results.

---

# 4. AI Must Not Invent Information

The AI must never invent:

* experimental results,
* dataset statistics,
* model performance,
* literature findings,
* citations,
* papers,
* equations attributed to a paper,
* implementation details supposedly taken from a paper,
* hardware benchmarks,
* clinical conclusions.

If information is unknown, the AI must explicitly state that it is unknown or requires verification.

---

# 5. Source and Literature Rules

When implementing or discussing a method based on a research paper:

* preserve the paper's actual terminology,
* distinguish the paper's method from our proposed modification,
* do not claim that our implementation is identical unless verified,
* do not copy proprietary/non-public implementation details without a legitimate source,
* clearly distinguish existing methods from proposed methods.

When external research is required, use reliable primary sources whenever possible.

Preferred sources include:

* original research papers,
* official dataset documentation,
* official framework documentation,
* official GitHub repositories associated with the research,
* peer-reviewed publications.

---

# 6. Novelty Rules

Novelty must never be assumed.

The following are **not automatically novel**:

* using a Transformer,
* using CNNs,
* using 2.5D input,
* using attention,
* using pruning,
* using multi-scale features,
* using LUNA16,
* combining known modules.

The project must distinguish between:

```text
Existing Technique
        ↓
Our Adaptation
        ↓
Our Combination
        ↓
Experimental Evidence
        ↓
Potential Research Contribution
```

Any final novelty claim must be supported by literature review and experimental evidence.

---

# 7. Architecture Rules

The architecture must remain modular.

Components should be independently replaceable wherever practical.

For example:

```text
CNN
 ↓
Transformer
 ↓
Attention
 ↓
Fusion
 ↓
Detection Head
```

should not be implemented as one monolithic class.

Instead, individual components should have clear interfaces.

---

# 8. Do Not Over-Commit to Architecture

At the beginning of the project, the following must remain configurable:

* CNN backbone,
* Transformer architecture,
* number of Transformer blocks,
* feature dimensions,
* attention mechanism,
* number of 2.5D slices,
* patch size,
* fusion mechanism,
* detection head,
* pruning strategy,
* pruning ratio,
* pruning schedule,
* loss weights.

Do not describe an experimentally unverified configuration as the final architecture.

---

# 9. Baseline First Rule

A complete research system must not be built before a working baseline exists.

The development sequence should generally follow:

```text
Dataset
 ↓
Preprocessing
 ↓
Baseline
 ↓
2.5D
 ↓
CNN + Transformer
 ↓
Attention
 ↓
Pruning
 ↓
Final (Pruned) Model
```

Each stage must be independently testable.

---

# 10. One Major Change at a Time

When establishing the contribution of a component, avoid simultaneously changing multiple major variables.

For example, if evaluating attention:

```text
Baseline
vs
Baseline + Attention
```

is preferred over:

```text
Baseline
vs
New Backbone + Attention + New Loss + New Training Schedule
```

This allows the effect of individual components to be identified.

---

# 11. Dataset Rules

The LUNA16 dataset must be treated as the authoritative dataset for this project.

The project must:

* validate available files,
* verify scan/annotation relationships,
* preserve scan identity,
* record preprocessing decisions,
* maintain reproducible splits.

Original dataset files must not be destructively modified.

---

# 12. Data Leakage Rules

Data leakage is strictly prohibited.

The same scan must not unintentionally contribute data to multiple dataset partitions.

In particular, the project must not:

```text
Randomly split slices
```

or:

```text
Randomly split patches from the same scan
```

if that results in information from the same scan appearing in both training and validation/test sets.

Splitting must occur at an appropriate scan/patient level.

---

# 13. Preprocessing Rules

Preprocessing must be:

* deterministic where appropriate,
* configurable,
* documented,
* reproducible.

The preprocessing pipeline must clearly distinguish:

### Training preprocessing

May include appropriate random augmentation.

### Validation preprocessing

Must not use random training augmentation.

### Test preprocessing

Must reproduce the defined evaluation pipeline.

No test information may influence training preprocessing decisions.

---

# 14. 2.5D Input Rules

The project will use 2.5D representations as the primary research direction.

The number of neighboring slices must be configurable.

For example:

```text
z-1
 z
z+1
```

may form one representation.

However, this must remain an experimental parameter rather than an immutable assumption.

Slice ordering must be preserved correctly.

Adjacent slices must correspond to the correct spatial order in the original CT volume.

---

# 15. Medical Image Handling Rules

CT data must be handled with awareness of medical-image metadata.

Where required, preserve or correctly interpret:

* voxel spacing,
* image orientation,
* slice ordering,
* origin,
* coordinate systems,
* physical dimensions.

Pixel/voxel coordinates must not be confused with physical coordinates.

When converting between coordinate systems, the transformation must be explicitly documented and tested.

---

# 16. Augmentation Rules

Augmentation must be medically and technically reasonable.

Do not introduce transformations that could create unrealistic CT anatomy or invalidate nodule annotations.

Every augmentation must preserve the correspondence between:

```text
Image
+
Nodule annotation
```

Augmentations must be applied consistently to both when required.

---

# 17. Training Rules

Every training experiment must record:

* experiment ID,
* configuration,
* random seed,
* dataset split,
* model configuration,
* optimizer,
* learning rate,
* scheduler,
* batch size,
* number of epochs,
* loss configuration,
* pruning configuration,
* checkpoint information.

Training results must not be manually edited.

---

# 18. Random Seed Rules

Experiments must use explicit random seeds whenever reproducibility permits.

Seeds should be recorded for:

* Python,
* NumPy,
* PyTorch,
* relevant data-loader operations.

If complete determinism is not possible because of hardware or backend behavior, the limitation must be documented.

---

# 19. Loss Function Rules

The training objective is the detection objective itself:

```text
L_total = L_det = λ_cls L_focal + λ_loc L_loc
```

There is no additional regularization or auxiliary loss term.

* Detection loss is the sole optimization objective.
* Loss weights (`λ_cls`, `λ_loc`) must be configurable.
* The classification/objectness term may use focal loss and the localization term may use an IoU-based loss; the exact localization formulation is finalized based on the selected detection head.
* If any additional loss term is ever introduced, it must be documented, kept independently measurable, made configurable, and evaluated against a baseline model through a controlled experiment, and the change must be recorded as a methodology change.

---

# 20. Experimental Component Rules

Attention and progressive pruning are **research components**, not assumed improvements.

Each must:

* have a clearly defined formulation,
* be configurable and support being disabled for ablation,
* be compared against the same architecture without it,
* not be claimed beneficial until experiments demonstrate it.

---

# 21. Pruning Rules

Pruning must be treated as an experimental optimization/compression process. Progressive pruning follows a prune → fine-tune → evaluate cycle; pruning decisions are evaluated on the validation split and only the final (pruned) model is evaluated on the held-out test split.

The project must record:

* pruning type,
* target sparsity,
* pruning schedule,
* pruning frequency,
* layers affected,
* recovery/fine-tuning procedure,
* performance before pruning,
* performance after pruning.

Do not assume:

```text
More pruning = Better model
```

or:

```text
More sparsity = Faster inference
```

These must be measured.

---

# 22. Structured vs Unstructured Pruning

The project must clearly distinguish:

### Unstructured pruning

Individual weights are removed.

### Structured pruning

Entire channels, filters, heads, blocks, or other structured units are removed.

A sparse model may have fewer effective weights without obtaining proportional hardware acceleration.

Therefore, practical speedup must be measured separately.

---

# 23. Efficiency Measurement Rules

Efficiency must be measured rather than inferred.

At minimum, where practical:

```text
Parameter Count
FLOPs / MACs
Model Size
Memory Usage
Inference Latency
Sparsity
```

Latency measurements must specify:

* hardware,
* device,
* batch size,
* input size,
* warm-up procedure,
* number of measurements.

Comparisons must use the same conditions.

---

# 24. Evaluation Rules

The project is a **detection project**.

Therefore, classification accuracy alone must not be treated as the primary success metric.

Evaluation should prioritize:

* FROC,
* CPM,
* sensitivity,
* false positives per scan.

Additional metrics may be reported where appropriate.

---

# 25. FROC Rules

FROC evaluation must use a clearly documented protocol.

The project must record:

* false-positive rates,
* sensitivity at corresponding operating points,
* matching criteria,
* nodule/candidate matching procedure,
* scan-level aggregation,
* CPM calculation methodology.

The implementation must be validated against the relevant LUNA16 evaluation methodology.

---

# 26. Test Set Rules

The test set must remain isolated from model development decisions.

The test set must not be repeatedly used to:

* select hyperparameters,
* select architectures,
* determine pruning levels,
* tune loss weights,
* choose checkpoints.

If the official LUNA16 evaluation protocol is used, it must be documented explicitly.

---

# 27. Hyperparameter Rules

Hyperparameters must not be changed silently.

Examples:

* learning rate,
* batch size,
* weight decay,
* dropout,
* Transformer depth,
* feature dimension,
* patch size,
* slice count,
* pruning percentage,
* loss weights.

Every meaningful change should be associated with an experiment/configuration.

---

# 28. Experiment Naming Rules

Experiments should have unique, descriptive names.

Example:

```text
EXP-001-baseline-2d
EXP-002-baseline-25d
EXP-003-cnn-transformer
EXP-004-attention
EXP-005-progressive-pruning
```

Never overwrite previous experimental results.

---

# 29. Results Rules

Raw results must be preserved.

Do not manually replace experimental numbers in result files.

Each result should be traceable to:

```text
Experiment ID
      ↓
Configuration
      ↓
Checkpoint
      ↓
Evaluation
      ↓
Result
```

---

# 30. Visualization Rules

Visualizations should clearly identify:

* experiment,
* metric,
* dataset partition,
* units,
* relevant configuration.

Plots must not be manipulated to make weak results appear stronger.

Axes must not be misleading.

---

# 31. Error Handling Rules

The system must fail clearly when critical assumptions are violated.

Examples:

```text
Missing CT scan
Missing annotation
Invalid metadata
Invalid voxel spacing
Shape mismatch
Coordinate mismatch
NaN loss
Inf loss
Invalid configuration
Missing checkpoint
Unsupported device
```

Errors should provide useful diagnostic information.

Avoid broad exception handling such as:

```python
try:
    ...
except:
    pass
```

unless there is a documented reason.

---

# 32. Validation Rules in Code

Important assumptions should be validated programmatically.

Examples:

```text
Input tensor shape
Expected channel count
Slice ordering
Patch dimensions
Annotation coordinates
Class labels
Configuration values
Checkpoint compatibility
```

Fail early when possible.

---

# 33. Numerical Stability

Training code should monitor for:

* NaN loss,
* infinite loss,
* exploding gradients,
* invalid predictions,
* invalid probabilities,
* empty batches,
* unexpected tensor ranges.

If numerical instability occurs, investigate the cause rather than simply suppressing the error.

---

# 34. Memory Rules

The development machine has limited RAM.

Therefore:

* do not load the entire LUNA16 dataset into RAM unnecessarily,
* use lazy/on-demand loading where practical,
* avoid unnecessary copies of large CT volumes,
* release unused tensors,
* monitor memory usage,
* use smaller batches when required,
* avoid unnecessarily large intermediate feature maps.

Memory-saving techniques should not silently change the experimental methodology.

---

# 35. Device Rules

The project should support:

```text
CPU
```

as the fallback execution device.

Where stable and supported:

```text
Apple MPS
```

may be used for acceleration.

Device selection should be configurable.

Code should not assume CUDA is available.

---

# 36. Dependency Rules

Only introduce a library when:

1. it solves a real project requirement,
2. it is reasonably maintained,
3. its license is appropriate,
4. its functionality cannot be reasonably implemented using existing dependencies,
5. adding it does not unnecessarily complicate reproducibility.

Avoid dependency bloat.

---

# 37. Library Rules

Core libraries should include only what is necessary.

Expected categories:

```text
Deep Learning
    PyTorch

Medical Imaging
    SimpleITK
    nibabel

Numerical
    NumPy
    SciPy

Data
    pandas

Machine Learning Utilities
    scikit-learn

Configuration
    PyYAML

Visualization
    matplotlib

Testing
    pytest
```

Additional libraries require justification.

---

# 38. Version Rules

Package versions should be recorded.

At minimum record:

* Python,
* PyTorch,
* torchvision where used,
* NumPy,
* SimpleITK,
* nibabel,
* scikit-learn,
* matplotlib,
* other critical dependencies.

Environment configuration should be reproducible through `environment.yml` and/or `requirements.txt`.

---

# 39. Git Rules

Git should be used for source-code version control.

Commit messages should describe meaningful changes.

Do not commit:

* large raw CT datasets,
* generated caches,
* temporary files,
* secrets,
* unnecessary checkpoints,
* local environment files.

Use `.gitignore` appropriately.

---

# 40. Sensitive Information Rules

Never place in the repository:

* passwords,
* API keys,
* access tokens,
* private credentials,
* personal secrets.

If external services are introduced later, credentials must be stored outside source code.

---

# 41. Documentation Rules

Every major research component must have documentation explaining:

```text
What it is
Why it exists
How it works
How it is implemented
How it is configured
How it is evaluated
```

Documentation must distinguish between:

```text
Implemented
```

and:

```text
Planned
```

and:

```text
Experimental
```

---

# 42. Project Memory Rules

`memory.md` will act as the project's persistent AI/project state file.

It should record concise information such as:

```text
Completed
Current Work
Pending Work
Decisions
Known Issues
Important Experimental Results
Next Action
```

The AI must update it when a significant project state changes.

Do not use it as a replacement for detailed documentation.

---

# 43. AI Context Rules

When AI is asked to modify or continue the project, it should first use:

```text
prd.md
architecture.md
rules.md
phases.md
memory.md
```

as the project's primary context.

If these documents conflict, the most recently confirmed project decision should be treated as authoritative and the conflict should be documented.

AI must not silently change an established research decision.

---

# 44. AI Coding Rules

Before generating significant code, AI should determine:

1. Which phase the project is currently in.
2. Which component is being implemented.
3. Which interfaces already exist.
4. Which configuration controls the component.
5. Which tests are required.
6. Whether the implementation affects existing experiments.

Generated code should integrate with the existing repository rather than creating an unrelated parallel implementation.

---

# 45. AI Research Boundaries

AI may:

* propose hypotheses,
* identify possible approaches,
* explain methods,
* compare architectures,
* suggest experiments,
* help formulate equations,
* analyze results.

AI must not:

* fabricate evidence,
* fabricate citations,
* manufacture experimental results,
* declare novelty without evidence,
* hide negative results,
* alter results to match expectations,
* present speculation as established fact.

---

# 46. AI Medical Boundaries

This project is an academic machine-learning research project.

The system must not be described as:

* a clinical diagnostic device,
* a replacement for a radiologist,
* a clinically validated medical system,
* a patient-specific diagnostic tool.

Model predictions are research outputs and must not be represented as medical diagnoses.

---

# 47. AI Decision-Making Boundary

AI may recommend technical experiments, but final research decisions must remain explicit and human-controlled.

For example:

```text
AI:
"Experiment X may be useful."

Researcher:
"Run Experiment X."

System:
"Experiment X executed and recorded."
```

AI must not silently decide to:

* change the dataset split,
* change evaluation methodology,
* remove unsuccessful experiments,
* alter reported results,
* redefine the research objective.

---

# 48. No Result Manipulation

The project must preserve negative results.

If:

```text
Baseline > Proposed Model
```

the result must be recorded.

If:

```text
Pruning decreases performance
```

the result must be recorded.

If:

```text
Attention provides no measurable improvement
```

the result must be recorded.

Negative findings are part of the research process.

---

# 49. No Cherry-Picking

Do not report only favorable experiments.

When comparing models, report the relevant configurations and evaluation conditions.

If a model is excluded from the final comparison, document why.

---

# 50. Reproducibility Checklist

Before considering an experiment complete, verify:

```text
[ ] Experiment ID assigned
[ ] Configuration saved
[ ] Dataset split recorded
[ ] Random seed recorded
[ ] Model configuration recorded
[ ] Training configuration recorded
[ ] Checkpoint saved
[ ] Evaluation completed
[ ] Metrics saved
[ ] Efficiency measured where required
[ ] Result visualization generated where required
[ ] Experiment documented
[ ] memory.md updated if significant
```

---

# 51. Definition of Done

A feature is not considered complete merely because the code runs.

A research component is complete when:

```text
Implementation
      ↓
Unit / sanity test
      ↓
Integration test
      ↓
Training experiment
      ↓
Evaluation
      ↓
Result recorded
      ↓
Documentation updated
```

---

# 52. Final Rule

The most important project rule is:

> **Do not optimize the project for obtaining a predetermined result. Optimize it for obtaining a valid, reproducible, and scientifically defensible result.**

All architecture decisions, experiments, pruning strategies, attention mechanisms, and final conclusions must follow this principle.
