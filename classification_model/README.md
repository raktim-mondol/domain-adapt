# BMA Multi-Level Multiple Instance Learning (MIL) Classifier

A deep learning framework for BMA (Bone Marrow Aspirate) classification using hierarchical Multiple Instance Learning with Vision Transformer features.

## Architecture

**Three-Level Hierarchical Architecture:**
1. **Patch Level**: Extract 12 patches (3×4 grid) from 4032×3024 images
2. **Image Level**: Aggregate patch features using attention mechanism
3. **Pile Level**: Aggregate image features for final classification (4 BMA classes)

**Feature Extraction**: ViT-R50 (vit_base_r50_s16_224.orig_in21k) - 768-dimensional features

## Project Structure

```
pile_level_classification_windsurf/
├── src/                          # Source code
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   └── bma_mil_model.py     # BMA MIL classifier architecture
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── patch_extractor.py   # Patch extraction from images
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── training.py          # Training utilities
│   │   ├── evaluation.py        # Evaluation metrics
│   │   ├── logging_utils.py     # Logging and result saving
│   │   └── early_stopping.py    # Early stopping implementation
│   ├── __init__.py
│   ├── feature_extractor.py     # ViT-R50 feature extraction
│   └── augmentation.py          # Data augmentation pipeline
├── configs/                      # Configuration files
│   ├── __init__.py
│   └── config.py                # Main configuration
├── scripts/                      # Training and utility scripts
│   └── train.py                 # Main training script
├── tests/                        # Test suite
│   ├── test_unit.py             # Unit tests
│   ├── test_end_to_end.py       # Integration tests
│   └── test_augmentation.py     # Augmentation visualization
├── docs/                         # Documentation
├── data/                         # Data directory (create this)
│   ├── BWM_label_data.csv       # Labels file
│   └── images/                  # Image directory
├── models/                       # Saved models (auto-created)
├── results/                      # Training results (auto-created)
├── logs/                         # Log files (auto-created)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Data

1. Place your CSV file at `data/BWM_label_data.csv`
2. Place images in `data/images/` directory
3. Update `configs/config.py` if paths differ

## Usage

### Training

```bash
python scripts/train.py
```

### Configuration

Edit `configs/config.py` to customize:

- **Model Architecture**: Feature dimensions, hidden layers
- **Training Parameters**: Epochs, batch size, learning rate
- **Data Augmentation**: Enable/disable geometric, color, noise augmentations
- **Early Stopping**: Patience, minimum delta
- **Class Imbalance**: Weighted loss

### Data Augmentation

**Training Pipeline:**
- Histogram normalization (CLAHE)
- Geometric transforms (rotation, zoom, shear, flip)
- Color augmentations (brightness, contrast, saturation, hue)
- Noise and blur

**Validation/Test Pipeline:**
- Histogram normalization only

Configure in `configs/config.py`:
```python
ENABLE_GEOMETRIC_AUG = True
ENABLE_COLOR_AUG = False
ENABLE_NOISE_AUG = False
```

## Testing

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Individual Test Suites
```bash
# Unit tests
python tests/test_unit.py

# End-to-end tests
python tests/test_end_to_end.py

# Augmentation visualization
python tests/test_augmentation.py
```

## Features

### ✓ Multi-Level MIL Architecture
- Handles variable number of images per pile
- Attention mechanisms at image and pile levels
- Hierarchical feature aggregation

### ✓ Advanced Data Augmentation
- Medical image-specific histogram normalization
- Geometric transformations for robustness
- Optional color and noise augmentations

### ✓ Class Imbalance Handling
- Automatic class weight computation
- Weighted loss function
- Stratified data splitting

### ✓ Training Utilities
- Early stopping with patience
- Comprehensive logging
- Model checkpointing
- Training visualization

### ✓ Evaluation Metrics
- Accuracy and F1 scores (overall and per-class)
- Confusion matrices
- Attention weight visualization

## Model Performance

The model outputs:
- **Pile-level predictions**: 4 BMA classes
- **Attention weights**: Image importance scores
- **Per-class metrics**: F1 scores for each BMA class

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full list

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bma_mil_classifier,
  title={BMA Multi-Level MIL Classifier},
  author={Research Team},
  year={2024},
  url={https://github.com/yourusername/bma-mil-classifier}
}
```

## License

[Add your license here]

## Contact

[Add contact information]

## Acknowledgments

- ViT-R50 model from timm library
- Multiple Instance Learning framework
- PyTorch deep learning framework
