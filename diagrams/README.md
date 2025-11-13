# AdaptBMA Architecture Diagrams

This folder contains Mermaid diagrams that visualize the AdaptBMA (Adaptive Bag-based Multi-instance Aggregation) model architecture and training pipeline.

## Diagram Files

### 1. [High-Level System Architecture](01_high_level_architecture.mmd)
**Overview of the complete system**
- Data input from QLD1 (source) and QLD2 (target) domains
- Preprocessing and patch extraction
- AdaptBMA model components
- Loss functions (Classification, Adversarial, MMD, Orthogonal)
- Training and validation pipeline
- Model checkpointing

**Key Components:**
- Feature Extractor (ViT-R50)
- Attention Aggregator
- Classifier Head
- Domain Discriminator with GRL
- Four loss components

---

### 2. [Detailed Model Architecture](02_detailed_model_architecture.mmd)
**Layer-by-layer architecture breakdown**
- Image to patch extraction (12 patches per image)
- Feature extraction through ViT-R50
- MIL attention aggregation mechanism
- Task head (classification branch)
- Domain head (discriminator branch with GRL)

**Dimensions:**
- Input: 4032×3024 → 12×224×224 patches
- Features: 768-dim (ViT) → 512-dim (aggregation)
- Task output: 3 classes
- Domain output: 1 binary prediction

---

### 3. [Training Flow](03_training_flow.mmd)
**Step-by-step training loop**
- Dual-domain batch loading
- Forward passes for both domains
- Ramp-up coefficient computation
- Multi-component loss calculation
- Gradient clipping and optimization
- Validation and checkpointing logic

**Flow:**
1. Load source and target batches
2. Forward through model
3. Compute all losses with ramp-up
4. Backward pass with gradient clipping
5. Validate both domains
6. Save best model based on target F1

---

### 4. [Loss Components](04_loss_components.mmd)
**Detailed breakdown of each loss function**

**L_cls (Classification Loss):**
- CrossEntropy for both domains
- Combined loss for joint training

**L_adv (Adversarial Loss):**
- DANN with Gradient Reversal Layer
- Binary cross-entropy with label smoothing
- Domain labels: 0 (source), 1 (target)

**L_mmd (Maximum Mean Discrepancy):**
- Class-conditional variant
- Multi-kernel RBF: σ ∈ {0.5, 1.0, 2.0, 4.0}
- Distribution alignment per class

**L_orth (Orthogonal Regularization):**
- Weight matrix orthogonality
- Normalized Frobenius norm
- Decouples task and domain features

---

### 5. [Training Strategy](05_training_strategy.mmd)
**Three-phase training approach**

**Phase 1 - Warmup (Epochs 0-2):**
- Gradual ramp-up of λ values: 0.0 → 0.4
- Focus on classification
- Gentle adaptation introduction

**Phase 2 - Adaptation (Epochs 3-5):**
- Full ramp-up: λ → 1.0
- Active domain confusion
- Distribution alignment
- Feature orthogonality

**Phase 3 - Fine-tuning (Epochs 6+):**
- Constant λ values
- Convergence
- Target improvement, source preservation

**Validation:**
- Every epoch: both domains
- Pile-level aggregation
- Early stopping on target F1

---

### 6. [Data Pipeline](06_data_pipeline.mmd)
**Data flow from CSV to training**
- CSV loading for both domains
- Pile-level train/val splitting
- Dataset creation
- DataLoader initialization
- Synchronized batch iteration
- Cycling shorter loader

**Data Stats:**
- QLD1: 60 images, 12 piles, 3 classes
- QLD2: 48 images, 12 piles, 3 classes
- Train/Val split: 70/30

---

### 7. [Gradient Reversal Layer](07_gradient_reversal.mmd)
**GRL mechanism explained**

**Forward Pass:**
- Identity function: output = input
- Features pass through unchanged

**Backward Pass:**
- Gradient reversal: grad_input = -λ × grad_output
- Reversed gradients flow to feature extractor

**Effect:**
- Discriminator learns to classify domains
- Feature extractor learns domain-invariant features
- Adversarial minimax game

---

### 8. [Evaluation Pipeline](08_evaluation_pipeline.mmd)
**From bag predictions to pile metrics**

**Bag-Level:**
- Process each image (bag) independently
- Get 3-class probability predictions
- Multiple bags per pile

**Pile-Level:**
- Group predictions by pile_id
- Mean pooling of probabilities
- Argmax for final prediction

**Metrics:**
- Pile-level accuracy
- Weighted F1-score (primary metric)
- Per-class F1-scores
- Confusion matrix

---

## Viewing on GitHub

All `.mmd` files can be viewed directly on GitHub with automatic Mermaid rendering. Simply click on any file above to see the rendered diagram.

## Model Name

**AdaptBMA** - Adaptive Bag-based Multi-instance Aggregation

Full name: *Adaptive Bag-based Multi-instance Aggregation for Cross-Domain Biological Assessment*

## Quick Stats

| Metric | Value |
|--------|-------|
| Total Parameters | 99,204,421 |
| Trainable Parameters | 1,314,309 |
| Loss Components | 4 |
| Adaptation Methods | 3 (DANN + MMD + Orth) |
| Domains | 2 (QLD1, QLD2) |
| Training Level | Bag (image) |
| Evaluation Level | Pile (aggregated) |

## Key Equations

**Total Loss:**
```
L_total = L_cls + λ_adv·L_adv + λ_mmd·L_mmd + λ_orth·L_orth
```

**Ramp-up Schedule:**
```
λ(e) = min(1.0, e / rampup_epochs)
```

**MMD (Multi-kernel):**
```
MMD²(X,Y) = Σ_σ [E[k_σ(x,x')] + E[k_σ(y,y')] - 2E[k_σ(x,y)]]
```

**Orthogonal Loss:**
```
L_orth = ||W_cls · W_dom^T||²_F / (||W_cls||_F · ||W_dom||_F)
```

## Related Documentation

- [Complete Architecture Documentation](../ARCHITECTURE_DIAGRAMS.md)
- [Domain Adaptation Implementation](../DOMAIN_ADAPTATION_README.md)
- [Testing Summary](../TESTING_SUMMARY.md)
- [Main README](../README.md)

---

**Last Updated:** November 2025
**Model Version:** AdaptBMA v1.0
