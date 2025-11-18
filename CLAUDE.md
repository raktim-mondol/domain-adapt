# CLAUDE.md

This file provides guidance to Claude Code and AI assistants when working with this repository.

## Project Overview

**Domain Adaptation for BMA MIL Classifier** - A research project implementing domain adaptation techniques to improve cross-domain performance of a coal mining spoil classification system using Multiple Instance Learning (MIL).

**Dataset**: Coal Mining SpoilType Classification
**BMA**: SpoilType (classification target in coal mining context)

### Repository Purpose

Transfer learning between QLD1 (source domain) and QLD2 (target domain) for 3-class BMA (SpoilType) classification using:
- **DANN**: Domain Adversarial Neural Networks with Gradient Reversal
- **MMD**: Maximum Mean Discrepancy for distribution alignment
- **Orthogonal Regularization**: Feature decoupling between task and domain

### Model Name: **AdaptBMA**
*Adaptive Bag-based Multi-instance Aggregation for Coal Mining Analysis*

## Repository Structure

```
domain-adapt/
├── classification_model/          # Main implementation directory
│   ├── src/
│   │   ├── models/                # Neural network architectures
│   │   │   ├── bma_mil_model.py           # Core MIL classifier
│   │   │   └── domain_discriminator.py    # DANN components (GRL + Discriminator)
│   │   ├── losses/                # Loss functions for domain adaptation
│   │   │   ├── mmd.py                     # Multi-kernel MMD implementation
│   │   │   └── orthogonal.py              # Orthogonal regularization
│   │   ├── data/                  # Dataset and data loading
│   │   │   ├── dataset.py                 # Bag-level dataset (image-level)
│   │   │   ├── pile_dataset.py            # Pile-level dataset
│   │   │   └── patch_extractor.py         # Patch extraction from images
│   │   ├── utils/                 # Training and evaluation utilities
│   │   │   ├── training.py                # Standard bag-level training
│   │   │   ├── pile_training.py           # Pile-level training
│   │   │   ├── domain_adaptation.py       # Domain adaptation training loop
│   │   │   ├── evaluation.py              # Metrics and evaluation
│   │   │   ├── pooling.py                 # Pile-level pooling methods
│   │   │   ├── early_stopping.py          # Early stopping logic
│   │   │   └── logging_utils.py           # Logging utilities
│   │   ├── augmentation.py        # Data augmentation pipeline
│   │   └── feature_extractor.py   # ViT-R50 wrapper
│   ├── scripts/
│   │   ├── train.py                       # Standard training script
│   │   ├── train_domain_adaptation.py     # Domain adaptation training
│   │   ├── test_domain_adaptation.py      # Component tests
│   │   ├── test_da_training_dummy.py      # End-to-end test with dummy data
│   │   └── generate_dummy_data.py         # Generate synthetic test data
│   ├── configs/
│   │   ├── config.py                      # Main configuration file
│   │   └── config_dummy.py                # Test configuration
│   ├── test/                      # Unit and integration tests
│   ├── data/                      # Data directory (CSV files)
│   ├── models/                    # Saved model checkpoints
│   ├── results/                   # Training plots and results
│   ├── logs/                      # Training logs
│   ├── CLAUDE.md                  # Detailed guide for classification_model
│   └── README.md                  # Classification model documentation
├── diagrams/                      # Mermaid diagrams for architecture
│   ├── *.mmd                      # Individual diagram files
│   └── README.md                  # Diagram index
├── scripts/                       # Repository-level scripts
│   └── install_pkgs.sh            # Package installation
├── DOMAIN_ADAPTATION_README.md    # Domain adaptation implementation guide
├── ARCHITECTURE_DIAGRAMS.md       # Complete architecture diagrams
├── TESTING_SUMMARY.md             # Comprehensive testing results
├── domain_adaptation_plan.md      # Original implementation plan
├── requirements.txt               # Python dependencies
└── README.md                      # Repository overview

```

## Key Components

### 1. Core MIL Model (`classification_model/src/models/bma_mil_model.py`)

**Architecture Flow**:
```
Input Image (4032×3024)
  ↓ Patch Extraction (12 patches)
Raw Patches [12, 3, 224, 224]
  ↓ Feature Extractor (ViT-R50)
Patch Features [12, 768]
  ↓ Attention Aggregator
Bag Feature z [512]
  ↓ Classifier Head
Class Logits [3]
```

**Key Methods**:
- `forward()`: Returns both logits and bag features (z)
- `get_classifier_weights()`: Extract weights for orthogonal loss
- Supports both bag-level and pile-level training modes

### 2. Domain Adaptation Components

#### Gradient Reversal Layer (`classification_model/src/models/domain_discriminator.py`)
- **Forward**: Identity (y = x)
- **Backward**: Reverses gradients (dy/dx = -λ * grad_output)
- **Lambda scheduling**: 0 → 1 over first 5 epochs (configurable)

#### Domain Discriminator
- **Architecture**: Linear(512→256) → ReLU → Dropout → Linear(256→1)
- **Stability**: Spectral normalization on all linear layers
- **Output**: Single domain prediction logit

#### MMD Loss (`classification_model/src/losses/mmd.py`)
- **Kernels**: Multi-kernel RBF with bandwidths [0.5, 1.0, 2.0, 4.0]
- **Mode**: Class-conditional (aligns per-class distributions)
- **Formula**: MMD²(X,Y) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

#### Orthogonal Regularization (`classification_model/src/losses/orthogonal.py`)
- **Purpose**: Decouple task features from domain features
- **Formula**: L_orth = ||W_cls · W_dom^T||_F² / (||W_cls||_F · ||W_dom||_F)
- **Effect**: Forces classifier and discriminator to learn independent features

### 3. Training Pipeline (`classification_model/src/utils/domain_adaptation.py`)

**Dual-Domain Training Loop**:
1. Load batches from both source (QLD1) and target (QLD2) loaders
2. Forward pass through shared model for both domains
3. Compute combined loss: `L_total = L_cls + λ_adv·L_adv + λ_mmd·L_mmd + λ_orth·L_orth`
4. Backward pass with gradient clipping
5. Optimizer step
6. Validation on both domains (pile-level aggregation)
7. Early stopping based on target domain F1

**Loss Components**:
- **L_cls**: CrossEntropy for both domains (supervised)
- **L_adv**: Binary CrossEntropy for domain classification (with GRL)
- **L_mmd**: Multi-kernel class-conditional MMD
- **L_orth**: Orthogonal regularization between heads

## Development Workflows

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the provided script
bash scripts/install_pkgs.sh
```

**Required Libraries**:
- torch >= 1.9.0 (PyTorch)
- torchvision >= 0.10.0
- timm >= 0.6.0 (ViT models)
- pandas, numpy, scikit-learn
- matplotlib, Pillow, opencv-python
- tqdm

### Configuration (`classification_model/configs/config.py`)

**Critical Settings to Review**:

```python
# Enable Domain Adaptation
USE_DOMAIN_ADAPTATION = True  # Set to False for standard training

# Data Paths (UPDATE THESE)
QLD1_DATA_PATH = 'data/qld1_data.csv'      # Source domain
QLD2_DATA_PATH = 'data/qld2_data.csv'      # Target domain
QLD1_IMAGE_DIR = 'data/qld1_images/'
QLD2_IMAGE_DIR = 'data/qld2_images/'

# Training Mode
TRAINING_LEVEL = 'bag'  # 'bag' (image-level) or 'pile' (pile-level)

# Domain Adaptation Hyperparameters
LAMBDA_ADV = 1.0       # Adversarial loss weight
LAMBDA_MMD = 0.5       # MMD loss weight
LAMBDA_ORTH = 0.01     # Orthogonal loss weight
GRL_COEFF = 1.0        # Gradient reversal coefficient

# Ramp-up Schedules
RAMPUP_EPOCHS = 5              # Epochs for gradual ramp-up
RAMPUP_LAMBDA_ADV = True       # Enable adversarial ramp-up
RAMPUP_LAMBDA_MMD = True       # Enable MMD ramp-up
RAMPUP_GRL_COEFF = True        # Enable GRL ramp-up

# MMD Configuration
MMD_BANDWIDTHS = [0.5, 1.0, 2.0, 4.0]  # Multi-kernel bandwidths
USE_CLASS_COND_MMD = True               # Class-conditional variant

# Stability Settings
USE_SPECTRAL_NORM = True           # Spectral normalization
DOMAIN_LABEL_SMOOTHING = 0.05      # Label smoothing (0.0-0.5)
USE_GRADIENT_CLIPPING = True       # Enable gradient clipping
GRADIENT_CLIP_MAX_NORM = 5.0       # Gradient clip threshold

# Feature Extractor
TRAINABLE_FEATURE_LAYERS = 2   # 0=frozen, -1=all, N=last N layers

# Optimizer
USE_ADAMW = True               # AdamW optimizer
LEARNING_RATE = 1e-4

# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
```

### Training Commands

#### Standard Training (No Domain Adaptation)
```bash
cd classification_model
python scripts/train.py
```

#### Domain Adaptation Training
```bash
cd classification_model
python scripts/train_domain_adaptation.py
```

#### Component Testing
```bash
cd classification_model
# Test all domain adaptation components
python scripts/test_domain_adaptation.py

# End-to-end test with dummy data (fast)
python scripts/test_da_training_dummy.py

# Generate synthetic test data
python scripts/generate_dummy_data.py
```

### Testing Suite

```bash
cd classification_model

# Unit tests
python test/test_unit.py

# End-to-end integration test
python test/test_end_to_end.py

# Test augmentation pipeline
python test/test_augmentation.py

# Test pooling methods
python test/test_pooling_methods.py

# Test scheduler functionality
python test/test_scheduler.py

# GPU training verification
python test/check_gpu_training.py
```

## Key Conventions and Best Practices

### Data Format

**CSV Structure** (both QLD1 and QLD2):
```csv
Sl,pile,image_path,BMA_label
1,pile_001,image_001.jpg,1
2,pile_001,image_002.jpg,1
...
```
- `pile`: Pile identifier (groups images together)
- `image_path`: Filename only (not full path)
- `BMA_label`: SpoilType label - Integer 1, 2, or 3 (converted to 0-indexed internally)

### Data Splitting

**CRITICAL**: Always split at **pile level** to prevent data leakage:
1. Split piles into train/val/test sets
2. All images from a pile go to the same split
3. Assertions verify no pile overlap between splits

### Training Levels

**Bag-level (Default)**:
- Each image is a training sample (bag = 12 patches)
- Faster training, lower memory
- Validation uses pile-level aggregation

**Pile-level**:
- Each pile is a training sample (pile = multiple images)
- Slower training, higher memory
- Consistent train/val methodology

### Code Modification Guidelines

1. **Never modify existing test files** unless fixing bugs
2. **Always update configuration** via `configs/config.py`, not hardcoded values
3. **Preserve data splitting logic** - pile-level splits are critical
4. **Maintain backward compatibility** with standard (non-DA) training
5. **Add comprehensive docstrings** for new functions
6. **Include type hints** where applicable

### File Naming Conventions

- Models: `best_bma_mil_model.pth` or `best_da_model.pth`
- Logs: Timestamped automatically
- Results: `*_results.txt`, `*_history.png`
- Configs: `config.py` (main), `config_*.py` (variants)

## Common Tasks

### Switch Between Training Modes

**Enable Domain Adaptation**:
```python
# In configs/config.py
USE_DOMAIN_ADAPTATION = True
```

**Disable Domain Adaptation (Standard Training)**:
```python
# In configs/config.py
USE_DOMAIN_ADAPTATION = False
```

### Adjust Domain Adaptation Strength

**Increase Adversarial Adaptation**:
```python
LAMBDA_ADV = 2.0  # Stronger domain confusion
```

**Increase Distribution Alignment**:
```python
LAMBDA_MMD = 1.0  # Stronger feature alignment
```

**Reduce Orthogonal Constraint**:
```python
LAMBDA_ORTH = 0.001  # More feature sharing between heads
```

### Change Pooling Method (Pile-level Aggregation)

```python
# In configs/config.py
POOLING_METHOD = 'mean'      # Average probabilities (default)
# POOLING_METHOD = 'max'     # Maximum probability per class
# POOLING_METHOD = 'attention'  # Learned attention weights
# POOLING_METHOD = 'majority'   # Vote counting
```

### Ablation Studies

**Test individual techniques by disabling others:**

```python
# Baseline (No Adaptation)
USE_DOMAIN_ADAPTATION = False

# DANN Only
LAMBDA_ADV = 1.0
LAMBDA_MMD = 0.0
LAMBDA_ORTH = 0.0

# MMD Only
LAMBDA_ADV = 0.0
LAMBDA_MMD = 0.5
LAMBDA_ORTH = 0.0

# DANN + MMD
LAMBDA_ADV = 1.0
LAMBDA_MMD = 0.5
LAMBDA_ORTH = 0.0

# Full (DANN + MMD + Orth)
LAMBDA_ADV = 1.0
LAMBDA_MMD = 0.5
LAMBDA_ORTH = 0.01
```

### Resume Training

```python
# In configs/config.py
RESUME_TRAINING = True
CHECKPOINT_PATH = 'models/best_bma_mil_model.pth'
```

### Adjust Data Augmentation

```python
# In configs/config.py
INCLUDE_ORIGINAL_AND_AUGMENTED = True
NUM_AUGMENTATION_VERSIONS = 3

# Enable specific augmentation types
ENABLE_GEOMETRIC_AUG = True   # Rotation, flipping
ENABLE_COLOR_AUG = False      # Color jittering
ENABLE_NOISE_AUG = False      # Gaussian noise
```

### Fine-tune Feature Extractor

```python
# Freeze all (feature extraction only)
TRAINABLE_FEATURE_LAYERS = 0

# Train last 2 layers/blocks
TRAINABLE_FEATURE_LAYERS = 2

# Train entire feature extractor
TRAINABLE_FEATURE_LAYERS = -1
```

## Output Files

### Models
- `classification_model/models/best_bma_mil_model.pth`: Best standard model
- `classification_model/models/best_da_model.pth`: Best domain adaptation model
- `classification_model/models/best_da_dummy_model.pth`: Test model from dummy data

### Results
- `classification_model/results/training_history.png`: Training/validation curves
- `classification_model/results/da_training_history.png`: DA training curves
- `classification_model/results/test_results.txt`: Final test metrics
- `classification_model/results/kfold_results.txt`: K-fold CV results

### Logs
- `classification_model/logs/training_*.log`: Timestamped training logs

## Git Workflow

### Branch Strategy

**Current Development Branch**: `claude/claude-md-mi3twcfr5j7wv39u-013QgU6o7tvZdQ26rSE99Qpg`

**Always**:
1. Develop on the designated Claude branch
2. Commit with clear, descriptive messages
3. Push to the designated branch (NOT main/master)
4. Never force push without explicit permission

### Commit Guidelines

```bash
# Good commit message examples
git commit -m "Add MMD loss with multi-kernel RBF implementation"
git commit -m "Fix tensor type handling in domain adaptation training loop"
git commit -m "Update config with domain adaptation hyperparameters"

# Stage and commit
git add <files>
git commit -m "Descriptive message about what and why"

# Push to remote
git push -u origin claude/claude-md-mi3twcfr5j7wv39u-013QgU6o7tvZdQ26rSE99Qpg
```

### Pull Request Process

When creating a PR:
1. Ensure all tests pass
2. Update documentation if needed
3. Provide clear summary of changes
4. Include test plan checklist

## Troubleshooting

### Common Issues

**Issue: MMD always returns 0**
- **Cause**: Source and target features identical, or bandwidth too small
- **Fix**: Check data loading; verify dual domains; adjust MMD_BANDWIDTHS

**Issue: Out of memory**
- **Cause**: Batch size too large or pile sizes too big
- **Fix**: Reduce BATCH_SIZE (try 4 or 2); reduce MAX_IMAGES_PER_PILE

**Issue: Domain discriminator accuracy stuck at 50%**
- **Status**: Expected! This means domain confusion is working
- **Explanation**: GRL successfully prevents domain classification

**Issue: Source performance drops significantly**
- **Cause**: Adaptation too aggressive
- **Fix**: Reduce LAMBDA_ADV and LAMBDA_MMD; increase LAMBDA_ORTH; extend RAMPUP_EPOCHS

**Issue: Target performance doesn't improve**
- **Cause**: Insufficient adaptation or wrong hyperparameters
- **Fix**: Increase LAMBDA_MMD; enable USE_CLASS_COND_MMD; check data paths

**Issue: Training unstable (loss spikes)**
- **Cause**: Gradient explosion or discriminator instability
- **Fix**: Reduce GRADIENT_CLIP_MAX_NORM (try 2.0); increase DOMAIN_LABEL_SMOOTHING; extend RAMPUP_EPOCHS

### Validation

**Always verify**:
1. Component tests pass: `python scripts/test_domain_adaptation.py`
2. Data paths are correct in config.py
3. CSV files have correct format (pile, image_path, BMA_label)
4. Image files exist and are readable
5. GPU/CPU device matches configuration

## Performance Expectations

### Model Size
- Total parameters: ~99.2M
- Trainable parameters: ~1.3M (frozen backbone)
- Checkpoint size: ~407 MB

### Training Time (Approximate)
- **CPU**: 2-3 min/epoch (batch_size=2, 20 batches)
- **GPU**: 20-30 sec/epoch (batch_size=8, standard settings)

### Memory Usage
- **CPU**: ~4 GB RAM
- **GPU**: ~6-8 GB VRAM (batch_size=8)

## Testing Status

**All Systems Verified** ✓ (See `TESTING_SUMMARY.md` for details)
- Component tests: PASS
- Integration tests: PASS
- End-to-end training: PASS
- Dummy data generation: PASS
- All loss components: PASS
- Ramp-up schedules: PASS

**Status**: READY FOR PRODUCTION USE

## Important Notes

1. **No feature pre-extraction**: Raw patches fed directly to model (end-to-end)
2. **Pile-level splitting**: Always performed before dataset creation
3. **Bag-level features**: All adaptation losses operate on bag features (z)
4. **Pile-level evaluation**: Mean pooling aggregates bags to piles
5. **Attention pooling**: Only trainable method; requires gradients
6. **Class imbalance**: Handled via weighted loss (optional)
7. **Domain labels**: 0 for source (QLD1), 1 for target (QLD2)

## Documentation References

### Detailed Guides (in repository root)
- `DOMAIN_ADAPTATION_README.md`: Comprehensive DA implementation guide
- `ARCHITECTURE_DIAGRAMS.md`: Complete architecture with diagrams
- `TESTING_SUMMARY.md`: Full testing results and verification
- `domain_adaptation_plan.md`: Original implementation specification

### Classification Model Docs (in classification_model/)
- `CLAUDE.md`: Detailed guide for classification model only
- `README.md`: Model documentation and usage

### Additional Resources
- `diagrams/`: Mermaid diagrams for all architecture components
- Individual `.mmd` files for GitHub rendering

## Quick Reference

### Essential Commands
```bash
# Setup
pip install -r requirements.txt

# Test components
cd classification_model && python scripts/test_domain_adaptation.py

# Train with domain adaptation
cd classification_model && python scripts/train_domain_adaptation.py

# Standard training (no DA)
cd classification_model && python scripts/train.py

# Generate test data
cd classification_model && python scripts/generate_dummy_data.py
```

### Key Configuration Variables
```python
USE_DOMAIN_ADAPTATION = True    # Enable/disable DA
TRAINING_LEVEL = 'bag'          # 'bag' or 'pile'
LAMBDA_ADV = 1.0                # Adversarial weight
LAMBDA_MMD = 0.5                # MMD weight
LAMBDA_ORTH = 0.01              # Orthogonal weight
RAMPUP_EPOCHS = 5               # Gradual ramp-up
```

### Key Files to Modify
1. `classification_model/configs/config.py`: All hyperparameters
2. `classification_model/scripts/train_domain_adaptation.py`: Training script
3. `classification_model/src/utils/domain_adaptation.py`: Training loop
4. `classification_model/src/models/domain_discriminator.py`: DANN components

### Key Files to NOT Modify (Unless Fixing Bugs)
1. Test files in `classification_model/test/`
2. Core model architecture in `src/models/bma_mil_model.py`
3. Loss implementations in `src/losses/`
4. Data loading in `src/data/`

## For AI Assistants

### Code Navigation Tips
1. Start with `classification_model/configs/config.py` to understand settings
2. Check `classification_model/CLAUDE.md` for detailed model guide
3. Review `DOMAIN_ADAPTATION_README.md` for DA implementation details
4. Examine `TESTING_SUMMARY.md` for verification of all components

### When Making Changes
1. Always read relevant configuration first
2. Check if tests exist for the component
3. Update configuration via config.py, not hardcoded
4. Preserve pile-level splitting logic
5. Maintain backward compatibility
6. Add/update docstrings
7. Run tests after changes

### Code Reading Priority
For understanding the codebase:
1. `classification_model/configs/config.py` (configuration)
2. `classification_model/src/models/bma_mil_model.py` (core model)
3. `classification_model/src/utils/domain_adaptation.py` (training loop)
4. `classification_model/src/models/domain_discriminator.py` (DANN)
5. `classification_model/src/losses/mmd.py` (MMD loss)
6. `classification_model/src/data/dataset.py` (data loading)

### Common Modification Patterns
- **Hyperparameter tuning**: Edit `configs/config.py` only
- **New loss function**: Add to `src/losses/`, update training loop
- **New model component**: Add to `src/models/`, integrate in training
- **New pooling method**: Add to `src/utils/pooling.py`
- **New augmentation**: Add to `src/augmentation.py`

## Version History

This repository implements the AdaptBMA domain adaptation system with:
- ✓ DANN with Gradient Reversal Layer
- ✓ Multi-kernel class-conditional MMD
- ✓ Orthogonal regularization
- ✓ Dual-domain training pipeline
- ✓ Comprehensive testing suite
- ✓ Production-ready codebase

**Last Updated**: 2024 (Implementation complete)

## Contact and Support

For questions or issues:
1. Check this CLAUDE.md file
2. Review detailed documentation in repository root
3. Review `classification_model/CLAUDE.md` for model-specific guidance
4. Run component tests to verify setup
5. Check training logs for errors

---

**Status**: PRODUCTION READY ✓

**Model**: AdaptBMA (Adaptive Bag-based Multi-instance Aggregation)

**Purpose**: Cross-domain BMA (SpoilType) classification for coal mining with domain adaptation
