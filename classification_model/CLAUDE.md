# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BMA (Bone Marrow Aspirate) classification system using Multi-Level Multiple Instance Learning (MIL). The system performs 3-class BMA classification with hierarchical architecture: Patch → Image → Pile level aggregation.

**Key Innovation**: Supports both bag-level (image) and pile-level training with flexible pooling methods for aggregation.

## Architecture

### Multi-Level MIL Pipeline
- **Input**: 4032×3024 images organized into piles
- **Patch Extraction**: 12 patches per image (3×4 grid, 1008×1008 → 224×224)
- **Feature Extraction**: ViT-R50 (`vit_base_r50_s16_224.orig_in21k`) produces 768-dim features
- **Aggregation**: Attention-based MIL aggregation
- **Classification**: 3-class BMA prediction

### Training Modes

The system supports two distinct training paradigms, configured via `TRAINING_LEVEL` in config.py:

1. **Bag-level Training** (`TRAINING_LEVEL = 'bag'`)
   - Each image is a training sample (one bag = one image with 12 patches)
   - Loss computed per bag using pile label
   - Validation aggregates bags to pile-level metrics
   - Faster training, lower memory usage

2. **Pile-level Training** (`TRAINING_LEVEL = 'pile'`)
   - Each pile is a training sample (one pile = multiple images)
   - Loss computed per pile using aggregated predictions
   - Consistent train/val methodology
   - Slower training, higher memory usage

## Essential Commands

### Environment Setup
```bash
# No virtual environment specified - use system Python or create one
pip install -r requirements.txt
```

### Training
```bash
# Main training script (auto-detects training mode from config)
python scripts/train.py

# Test GPU availability
python verify_gpu_training.py
python check_gpu_training.py
```

### Testing
```bash
# Unit tests
python test_unit.py

# End-to-end integration test
python test_end_to_end.py

# Test augmentation pipeline
python test_augmentation.py

# Test progress display
python test_progress_display.py

# Test scheduler functionality
python test_scheduler.py

# Test pooling methods
python test_pooling_methods.py
```

## Configuration (`configs/config.py`)

### Critical Settings

```python
# Data paths (MUST UPDATE)
IMAGE_DIR = r'D:\SCANDY\Data\BWM_Data'  # Update to your image directory
DATA_PATH = 'data/BWM_label_data.csv'

# Training mode (KEY SETTING)
TRAINING_LEVEL = 'bag'  # 'bag' or 'pile'

# Data split mode
SPLIT_MODE = 'kfold'  # 'standard' (70/10/20) or 'kfold'
NUM_FOLDS = 3  # For k-fold cross-validation

# Feature extractor fine-tuning
TRAINABLE_FEATURE_LAYERS = 2  # 0=frozen, -1=all trainable, N=last N layers

# Pooling method (pile-level aggregation)
POOLING_METHOD = 'mean'  # 'mean', 'max', 'attention', 'majority'

# Optimizer and scheduler
USE_ADAMW = True  # AdamW optimizer with better weight decay
USE_LR_SCHEDULER = True  # Enable learning rate scheduling
LR_SCHEDULER_TYPE = 'reduce_on_plateau'  # or 'cosine_annealing'

# Data augmentation
INCLUDE_ORIGINAL_AND_AUGMENTED = True  # Include both original and augmented patches
NUM_AUGMENTATION_VERSIONS = 3  # Number of augmented versions per patch
ENABLE_GEOMETRIC_AUG = True
ENABLE_COLOR_AUG = False
ENABLE_NOISE_AUG = False

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

# Resume training
RESUME_TRAINING = True
CHECKPOINT_PATH = 'models/best_bma_mil_model.pth'
```

## Key Implementation Details

### Data Splitting Strategy
- **Pile-level splitting FIRST**: Ensures no data leakage between train/val/test
- Images from train piles → training set
- Images from val piles → validation set
- Images from test piles → test set
- Assertions verify no pile overlap between splits

### Pooling Methods for Pile-level Aggregation

Four methods implemented to aggregate bag predictions to pile level:

1. **Mean Pooling**: Average probabilities across bags
2. **Max Pooling**: Maximum probability per class across bags
3. **Attention Pooling**: Learned attention weights (trainable)
4. **Majority Voting**: Vote counting from bag predictions

During evaluation, all 4 methods are displayed side-by-side for comparison.

### Feature Extractor Flexibility

```python
TRAINABLE_FEATURE_LAYERS:
  0: Fully frozen (feature extraction only)
  -1: Fully trainable (end-to-end training)
  N: Last N layers/blocks trainable
```

For ViT-R50: 12 blocks total, can train last N blocks.

### Augmentation Strategy

Controlled by `INCLUDE_ORIGINAL_AND_AUGMENTED` and `NUM_AUGMENTATION_VERSIONS`:
- `NUM_AUGMENTATION_VERSIONS=1, INCLUDE_ORIGINAL=False`: 12 patches (no aug)
- `NUM_AUGMENTATION_VERSIONS=2, INCLUDE_ORIGINAL=True`: 36 patches (12 orig + 24 aug)
- `NUM_AUGMENTATION_VERSIONS=3, INCLUDE_ORIGINAL=False`: 36 patches (all aug)

Validation always uses original patches only (no augmentation).

## File Structure

### Core Model Files
- `src/models/bma_mil_model.py`: End-to-end MIL model with integrated feature extractor
- `src/feature_extractor.py`: ViT-R50 wrapper for feature extraction

### Data Loading
- `src/data/dataset.py`: Bag-level dataset (returns individual images)
- `src/data/pile_dataset.py`: Pile-level dataset (returns entire piles)
- `src/data/patch_extractor.py`: Extracts 12 patches from 4032×3024 images

### Training & Evaluation
- `src/utils/training.py`: Bag-level training loop
- `src/utils/pile_training.py`: Pile-level training loop
- `src/utils/evaluation.py`: Evaluation metrics and pile-level aggregation
- `src/utils/pooling.py`: All 4 pooling method implementations

### Utilities
- `src/augmentation.py`: Data augmentation pipeline (CLAHE, geometric, color, noise)
- `src/utils/early_stopping.py`: Early stopping implementation
- `src/utils/logging_utils.py`: Logging and result saving

## Common Tasks

### Switch Training Modes

**To Pile-level Training:**
```python
# In configs/config.py
TRAINING_LEVEL = 'pile'
```

**To Bag-level Training:**
```python
# In configs/config.py
TRAINING_LEVEL = 'bag'
```

### Change Pooling Method

```python
# In configs/config.py
POOLING_METHOD = 'attention'  # or 'mean', 'max', 'majority'
```

### Adjust Data Augmentation

```python
# In configs/config.py
INCLUDE_ORIGINAL_AND_AUGMENTED = True
NUM_AUGMENTATION_VERSIONS = 3
ENABLE_GEOMETRIC_AUG = True
ENABLE_COLOR_AUG = False
ENABLE_NOISE_AUG = False
```

### Change Split Mode

**Standard 70/10/20 Split:**
```python
SPLIT_MODE = 'standard'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
```

**K-Fold Cross-Validation:**
```python
SPLIT_MODE = 'kfold'
NUM_FOLDS = 5
```

### Adjust Feature Extractor Training

```python
# Freeze feature extractor (faster training, less memory)
TRAINABLE_FEATURE_LAYERS = 0

# Train only last 2 layers/blocks
TRAINABLE_FEATURE_LAYERS = 2

# Train entire feature extractor (slower, more memory)
TRAINABLE_FEATURE_LAYERS = -1
```

## Data Format

**CSV Structure** (`BWM_label_data.csv`):
```
Sl, pile, image_path, BMA_label
```
- `pile`: Pile identifier (groups images together)
- `image_path`: Filename only (not full path)
- `BMA_label`: Integer label (1, 2, or 3) - converted to 0-indexed internally

## Model Architecture Details

```
Input Image (4032×3024)
    ↓
PatchExtractor: 12 patches (3×4 grid)
    ↓
Raw Patches [12, 3, 224, 224]
    ↓
FeatureExtractor (ViT-R50): trainable/frozen
    ↓
Features [12, 768]
    ↓
MIL Aggregator (Attention-based)
    ↓
Aggregated Features [IMAGE_HIDDEN_DIM=512]
    ↓
Classifier Head
    ↓
Logits [NUM_CLASSES=3]
```

For pile-level training, an additional aggregation step pools multiple image predictions:
```
Bag Predictions [N_images, 3] → Pooling Method → Pile Prediction [3]
```

## Performance Optimization

### Memory Management
- Bag-level training uses less memory per batch
- Pile-level training can hit memory limits with large piles
- Adjust `MAX_IMAGES_PER_PILE` to limit pile size
- Reduce `BATCH_SIZE` if OOM errors occur

### Training Speed
- Bag-level: Faster (more samples, simpler forward pass)
- Pile-level: Slower (fewer samples, complex aggregation)
- Frozen feature extractor: Faster training
- More augmentation versions: Slower data loading

### GPU Utilization
- Verify GPU usage with `verify_gpu_training.py`
- `DEVICE` auto-detects CUDA availability
- Can force CPU with `DEVICE = 'cpu'` in config

## Output Files

### Models
- `models/best_bma_mil_model.pth`: Best model checkpoint (automatically saved)

### Results
- `results/training_history.png`: Training/validation curves
- `results/test_results.txt`: Final test set evaluation
- `results/kfold_results.txt`: K-fold cross-validation results

### Logs
- `logs/`: Training logs with timestamps

## Important Notes

1. **No feature pre-extraction**: Raw patches fed directly to model (end-to-end training)
2. **Pile-level splitting**: Always performed BEFORE dataset creation to prevent data leakage
3. **Class imbalance**: Handled via weighted loss when `USE_WEIGHTED_LOSS = True`
4. **Attention pooling**: Only trainable method; requires gradients during training
5. **Resume training**: Automatically loads checkpoint if `RESUME_TRAINING = True`
6. **Evaluation**: Always shows all 4 pooling methods side-by-side for comparison

## Documentation

Reference guides in repository root:
- `ARCHITECTURE_REFACTORED.md`: Complete architecture explanation
- `QUICK_SWITCH_GUIDE.md`: Switching between bag/pile training modes
- `POOLING_QUICK_REFERENCE.md`: Pooling methods guide
- `AUGMENTATION_STRATEGIES_GUIDE.md`: Data augmentation details
- `PILE_LEVEL_TRAINING_GUIDE.md`: Pile-level training methodology
- `QUICK_SCHEDULER_GUIDE.md`: Learning rate scheduler configuration
