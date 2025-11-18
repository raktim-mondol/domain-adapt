# Domain Adaptation Modes Guide

This guide explains the three domain adaptation modes available in the AdaptBMA system: **Supervised**, **Unsupervised**, and **Semi-Supervised**.

## Overview

The domain adaptation system supports three different training modes, each designed for different scenarios based on the availability of target domain labels:

| Mode | Source Labels | Target Labels | Use Case |
|------|--------------|---------------|----------|
| **Supervised** | ✓ All | ✓ All | Both domains fully labeled |
| **Unsupervised** | ✓ All | ✗ None | Only source domain labeled |
| **Semi-Supervised** | ✓ All | ✓ Partial (10-30%) | Target domain partially labeled |

---

## 1. Supervised Domain Adaptation

### Description
Both source and target domains have complete labels. This is the traditional domain adaptation setting where you can leverage labels from both domains during training.

### When to Use
- You have labels for both QLD1 (source) and QLD2 (target) datasets
- You want maximum performance when labels are available
- Benchmarking and comparing with other methods

### Configuration
```python
# In classification_model/configs/config.py
USE_DOMAIN_ADAPTATION = True
DA_MODE = 'supervised'
USE_CLASS_COND_MMD = True  # Can use class-conditional MMD
```

### How It Works
1. **Classification Loss**: Computed on both source and target samples
   ```
   L_cls = L_cls_source + L_cls_target
   ```

2. **Domain Adversarial Loss (DANN)**: Forces domain confusion
   - Uses both source and target data
   - GRL (Gradient Reversal Layer) prevents discriminator from distinguishing domains

3. **MMD Loss**: Class-conditional alignment
   - Aligns feature distributions per class
   - Ensures class 1 in source aligns with class 1 in target, etc.

4. **Orthogonal Regularization**: Decouples task/domain features

### Training Command
```bash
cd classification_model
python scripts/train_domain_adaptation.py
```

### Expected Performance
- **Best overall performance** (utilizes all available labels)
- Source F1: 85-95%
- Target F1: 80-90%

---

## 2. Unsupervised Domain Adaptation

### Description
Only the source domain has labels; the target domain is completely unlabeled. This is the most challenging but realistic scenario in many real-world applications.

### When to Use
- QLD2 (target) data is unlabeled or labels are expensive to obtain
- You want to adapt a model trained on QLD1 to work on QLD2 without labeling QLD2
- Real-world deployment where new domain data arrives unlabeled

### Configuration
```python
# In classification_model/configs/config.py
USE_DOMAIN_ADAPTATION = True
DA_MODE = 'unsupervised'
USE_CLASS_COND_MMD = False  # MUST be False (no target labels for class conditioning)
```

### How It Works
1. **Classification Loss**: Computed ONLY on source samples
   ```
   L_cls = L_cls_source
   ```
   - No classification loss from target (no labels available)

2. **Domain Adversarial Loss (DANN)**: Key technique for UDA
   - Uses both source and target data (labels not needed)
   - Learns domain-invariant features

3. **MMD Loss**: Non-class-conditional alignment
   - Aligns overall feature distributions (cannot split by class)
   - Uses `mmd_loss()` instead of `class_conditional_mmd_loss()`

4. **Orthogonal Regularization**: Still applicable (doesn't need target labels)

### Technical Compatibility

| Technique | Compatible? | Notes |
|-----------|------------|-------|
| DANN | ✓ YES | Designed specifically for UDA |
| MMD | ✓ YES | Use non-class-conditional variant |
| Orthogonal | ✓ YES | Doesn't require target labels |

### Training Command
```bash
cd classification_model
python scripts/test_all_da_modes.py  # Test UDA mode
```

### Expected Performance
- **Lower than supervised** (no target labels to guide learning)
- Source F1: 85-95%
- Target F1: 60-75% (estimated, depends on domain shift)

### Limitations
- Cannot directly measure target performance during training (no labels)
- Relies heavily on domain invariance assumptions
- May struggle with severe domain shift

---

## 3. Semi-Supervised Domain Adaptation

### Description
Source domain is fully labeled, but only a small fraction (10-30%) of the target domain is labeled. This balances performance and annotation cost.

### When to Use
- You can label a small subset of QLD2 data
- You want better performance than unsupervised but lower annotation cost than supervised
- Active learning scenarios where you select which samples to label

### Configuration
```python
# In classification_model/configs/config.py
USE_DOMAIN_ADAPTATION = True
DA_MODE = 'semi_supervised'
SSDA_LABELED_RATIO = 0.2  # 20% of target data is labeled
# OR
SSDA_LABELED_SAMPLES_PER_CLASS = 5  # 5 labeled piles per class

SSDA_RANDOM_SEED = 42  # For reproducible labeled sample selection
```

### How It Works
1. **Classification Loss**: Source + labeled target subset
   ```
   L_cls = L_cls_source + L_cls_target_labeled
   ```
   - Only labeled target samples contribute to classification loss

2. **Domain Adversarial Loss (DANN)**: Uses ALL target data
   - Both labeled and unlabeled target samples
   - Maximizes domain confusion

3. **MMD Loss**: Non-class-conditional (includes unlabeled)
   ```python
   # Combines labeled + unlabeled target features
   all_target_features = cat([labeled_features, unlabeled_features])
   L_mmd = mmd_loss(source_features, all_target_features)
   ```
   - Cannot use class-conditional (unlabeled samples have no class)

4. **Orthogonal Regularization**: Still applicable

### Data Split Example
For 9 target piles with `SSDA_LABELED_RATIO = 0.33`:
```
Total piles: 9
Labeled piles: 3 (33%)
Unlabeled piles: 6 (67%)

Per-class labeled counts:
  Class 1: 1 pile
  Class 2: 1 pile
  Class 3: 1 pile
```

### Training Setup
```python
# Split target piles into labeled and unlabeled
from src.utils.semi_supervised_utils import split_semi_supervised_piles

target_labeled, target_unlabeled = split_semi_supervised_piles(
    target_piles, target_df,
    labeled_ratio=0.2,  # 20% labeled
    random_seed=42
)

# Create separate datasets
target_train_labeled = create_bag_dataset_from_piles(
    df, target_labeled, image_dir, is_labeled=True
)
target_train_unlabeled = create_bag_dataset_from_piles(
    df, target_unlabeled, image_dir, is_labeled=False
)

# Train with both
train_model_domain_adaptation(
    ...,
    target_train_loader=target_labeled_loader,
    target_train_loader_unlabeled=target_unlabeled_loader
)
```

### Training Command
```bash
cd classification_model
python scripts/test_all_da_modes.py  # Test SSDA mode
```

### Expected Performance
- **Between supervised and unsupervised**
- Source F1: 85-95%
- Target F1: 70-85% (depends on labeled ratio)

### Performance vs. Labeled Ratio

| Labeled Ratio | Target F1 (est.) | Annotation Cost |
|---------------|------------------|-----------------|
| 10% | 65-75% | Very Low |
| 20% | 70-80% | Low |
| 30% | 75-85% | Medium |
| 50% | 80-88% | Medium-High |
| 100% (Supervised) | 85-90% | High |

---

## Quick Comparison

### Loss Functions by Mode

| Loss Component | Supervised | Unsupervised | Semi-Supervised |
|----------------|-----------|--------------|-----------------|
| **Classification (Source)** | ✓ | ✓ | ✓ |
| **Classification (Target)** | ✓ All | ✗ None | ✓ Labeled only |
| **DANN (Source)** | ✓ | ✓ | ✓ |
| **DANN (Target)** | ✓ | ✓ | ✓ All (labeled + unlabeled) |
| **MMD** | Class-conditional | Non-conditional | Non-conditional |
| **Orthogonal** | ✓ | ✓ | ✓ |

### Configuration Summary

```python
# Supervised
DA_MODE = 'supervised'
USE_CLASS_COND_MMD = True

# Unsupervised
DA_MODE = 'unsupervised'
USE_CLASS_COND_MMD = False  # REQUIRED

# Semi-Supervised
DA_MODE = 'semi_supervised'
USE_CLASS_COND_MMD = False  # Automatically handled
SSDA_LABELED_RATIO = 0.2  # 20% labeled
```

---

## Implementation Details

### Dataset Handling

**Unlabeled samples use label = -1 as sentinel value:**
```python
# In dataset.py
if self.is_labeled:
    label = image_data['pile_label']
else:
    label = -1  # Sentinel for unlabeled
```

**Creating datasets:**
```python
# Labeled
dataset = create_bag_dataset_from_piles(
    df, pile_ids, image_dir, is_labeled=True
)

# Unlabeled
dataset = create_bag_dataset_from_piles(
    df, pile_ids, image_dir, is_labeled=False
)
```

### Training Loop Adaptations

**Mode detection:**
```python
da_mode = config.DA_MODE  # 'supervised', 'unsupervised', or 'semi_supervised'
```

**Classification loss computation:**
```python
if da_mode == 'unsupervised':
    loss_cls = loss_cls_source  # No target loss

elif da_mode == 'semi_supervised':
    # Only compute loss on labeled target samples
    labeled_mask = target_labels >= 0
    loss_cls_target = criterion(logits[labeled_mask], labels[labeled_mask])
    loss_cls = loss_cls_source + loss_cls_target

else:  # supervised
    loss_cls = loss_cls_source + loss_cls_target
```

**MMD loss selection:**
```python
if da_mode == 'supervised' and config.USE_CLASS_COND_MMD:
    # Class-conditional MMD
    loss_mmd = class_conditional_mmd_loss(
        source_features, target_features,
        source_labels, target_labels, ...
    )
else:
    # Non-class-conditional MMD
    loss_mmd = mmd_loss(source_features, target_features, ...)
```

### Validation Handling

**Unsupervised validation:**
- Target metrics (accuracy, F1) are 0.0 (no ground truth)
- Only source metrics are meaningful
- Still track predictions for analysis

**Semi-supervised validation:**
- Metrics computed only on labeled target samples
- Unlabeled samples filtered out before metric computation

---

## Best Practices

### Choosing the Right Mode

1. **Use Supervised** if:
   - Both domains are fully labeled
   - You need maximum performance
   - Annotation cost is not a concern

2. **Use Unsupervised** if:
   - Target domain has no labels
   - Annotation is too expensive
   - You're deploying to a new domain without labels

3. **Use Semi-Supervised** if:
   - You can afford to label 10-30% of target data
   - You want a balance between performance and cost
   - You're using active learning

### Hyperparameter Tuning

**For Unsupervised:**
- Increase `LAMBDA_ADV` (1.5-2.0) - stronger domain confusion needed
- Increase `LAMBDA_MMD` (0.7-1.0) - more distribution alignment
- Extend `RAMPUP_EPOCHS` (7-10) - gradual adaptation helps

**For Semi-Supervised:**
- Standard hyperparameters work well
- Consider increasing labeled ratio if target performance is poor
- Balance between labeled and unlabeled losses

### Monitoring Training

**Supervised:**
- Watch both source and target F1
- Target F1 should steadily improve

**Unsupervised:**
- Only source F1 is available
- Monitor domain discriminator accuracy (should approach 50%)
- Track MMD loss (should decrease)

**Semi-Supervised:**
- Track labeled target F1 (on small labeled subset)
- Monitor source F1 for stability
- Check MMD and adversarial losses

---

## Testing

### Run All Mode Tests
```bash
cd classification_model
python scripts/test_all_da_modes.py
```

This will test all three modes with dummy data and report:
```
SUPERVISED      : ✓ PASS
UNSUPERVISED    : ✓ PASS
SEMI_SUPERVISED : ✓ PASS
```

### Manual Testing

**Test Supervised:**
```python
config.DA_MODE = 'supervised'
config.USE_CLASS_COND_MMD = True
# Both source and target datasets with is_labeled=True
```

**Test Unsupervised:**
```python
config.DA_MODE = 'unsupervised'
config.USE_CLASS_COND_MMD = False
# Source: is_labeled=True, Target: is_labeled=False
```

**Test Semi-Supervised:**
```python
config.DA_MODE = 'semi_supervised'
config.SSDA_LABELED_RATIO = 0.2

# Split target data
labeled_piles, unlabeled_piles = split_semi_supervised_piles(...)
# Create separate labeled and unlabeled loaders
```

---

## Troubleshooting

### Common Issues

**Issue: MMD loss is 0 in unsupervised mode**
- **Cause**: Using `USE_CLASS_COND_MMD = True` with unlabeled target
- **Fix**: Set `USE_CLASS_COND_MMD = False` for unsupervised mode

**Issue: Semi-supervised training crashes**
- **Cause**: No labeled target samples in some batches
- **Fix**: Code handles this gracefully; check labeled ratio is sufficient

**Issue: Target accuracy is 0 in unsupervised mode**
- **Status**: Expected! Target is unlabeled, so accuracy cannot be computed
- **Monitor**: Source accuracy, MMD loss, discriminator accuracy

**Issue: Semi-supervised performance not better than unsupervised**
- **Cause**: Labeled ratio too low or labeled samples not representative
- **Fix**: Increase `SSDA_LABELED_RATIO` to 0.25-0.3

---

## References

### Key Files

- `configs/config.py` - Configuration for DA modes
- `src/data/dataset.py` - Dataset with unlabeled support
- `src/utils/domain_adaptation.py` - Training loop for all modes
- `src/utils/semi_supervised_utils.py` - Semi-supervised utilities
- `src/losses/mmd.py` - MMD loss (class-conditional and standard)
- `scripts/test_all_da_modes.py` - Test script for all modes

### Related Documentation

- `DOMAIN_ADAPTATION_README.md` - General domain adaptation guide
- `CLAUDE.md` - Project overview and conventions
- `ARCHITECTURE_DIAGRAMS.md` - System architecture

---

## Summary

The AdaptBMA system now supports three domain adaptation modes:

1. **Supervised** - Both domains labeled (best performance)
2. **Unsupervised** - Only source labeled (most practical)
3. **Semi-Supervised** - Source + partial target labels (balanced)

All three modes use the same underlying techniques (DANN, MMD, Orthogonal) with appropriate adaptations:

- **DANN**: Works with all modes (doesn't need labels)
- **MMD**: Class-conditional for supervised, non-conditional for others
- **Orthogonal**: Works with all modes

Choose the mode based on your label availability and performance requirements.
