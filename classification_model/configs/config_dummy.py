"""
Configuration file for Domain Adaptation Testing with Dummy Data
"""

import torch


class Config:
    # Data parameters (DUMMY DATA)
    DATA_PATH = 'data/qld1_data.csv'
    IMAGE_DIR = 'data/qld1_images'
    NUM_CLASSES = 3

    # Image processing
    ORIGINAL_SIZE = (4032, 3024)
    PATCH_SIZE = 1008
    TARGET_SIZE = 224
    NUM_PATCHES_PER_IMAGE = 12
    MAX_IMAGES_PER_PILE = 100000

    # Model architecture
    FEATURE_DIM = 768  # ViT-R50 feature dimension
    IMAGE_HIDDEN_DIM = 512
    PILE_HIDDEN_DIM = 256

    # Training parameters (REDUCED FOR TESTING)
    BATCH_SIZE = 2  # Small batch for testing
    NUM_EPOCHS = 3  # Just 3 epochs for quick test
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.3

    # Optimizer and Scheduler
    USE_ADAMW = True
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_TYPE = 'reduce_on_plateau'

    # ReduceLROnPlateau parameters
    LR_SCHEDULER_MODE = 'min'
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_MIN_LR = 1e-7
    LR_SCHEDULER_THRESHOLD = 1e-4

    # Training level
    TRAINING_LEVEL = 'bag'

    # Pile-level aggregation method
    POOLING_METHOD = 'mean'

    # Class imbalance handling
    USE_WEIGHTED_LOSS = True

    # Data split
    SPLIT_MODE = 'standard'
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    RANDOM_STATE = 42

    # Cross-validation
    NUM_FOLDS = 3

    # Feature extractor
    FEATURE_EXTRACTOR_MODEL = 'vit_base_r50_s16_224.orig_in21k'
    TRAINABLE_FEATURE_LAYERS = 0  # Frozen for quick testing

    # Paths
    BEST_MODEL_PATH = 'models/best_da_dummy_model.pth'
    TRAINING_PLOT_PATH = 'results/da_dummy_training_history.png'
    LOG_DIR = 'logs'

    # Device
    DEVICE = 'cpu'  # Use CPU for testing

    # Checkpointing / Resume
    RESUME_TRAINING = False
    CHECKPOINT_PATH = BEST_MODEL_PATH

    # Early stopping
    USE_EARLY_STOPPING = False  # Disabled for quick test
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Evaluation
    EVAL_ON_PILE_LEVEL = True

    # Logging
    ENABLE_LOGGING = True
    LOG_LEVEL = 'INFO'

    # Data Augmentation Parameters
    HISTOGRAM_METHOD = 'clahe'

    # Augmentation strategy (MINIMAL FOR TESTING)
    INCLUDE_ORIGINAL_AND_AUGMENTED = False
    NUM_AUGMENTATION_VERSIONS = 1  # No augmentation for quick test

    # Enable/disable augmentation types
    ENABLE_GEOMETRIC_AUG = False
    ENABLE_COLOR_AUG = False
    ENABLE_NOISE_AUG = False

    # Geometric augmentation parameters
    ROTATION_RANGE = 15
    ZOOM_RANGE = (0.9, 2.5)
    HORIZONTAL_FLIP = True

    # Color augmentation parameters
    BRIGHTNESS_RANGE = (0.8, 1.2)
    CONTRAST_RANGE = (0.8, 1.2)
    SATURATION_RANGE = (0.8, 1.2)
    HUE_RANGE = (-0.1, 0.1)
    COLOR_PROB = 0.5

    # Noise and blur parameters
    NOISE_STD = 0.01
    BLUR_SIGMA = (0.1, 2.0)
    NOISE_PROB = 0.3

    # ==================== Domain Adaptation Parameters ====================

    # Enable domain adaptation
    USE_DOMAIN_ADAPTATION = True  # ENABLED FOR TESTING

    # Domain data paths (DUMMY DATA)
    QLD1_DATA_PATH = 'data/qld1_data.csv'  # Source domain data
    QLD2_DATA_PATH = 'data/qld2_data.csv'  # Target domain data
    QLD1_IMAGE_DIR = 'data/qld1_images'    # Source domain images
    QLD2_IMAGE_DIR = 'data/qld2_images'    # Target domain images

    # Domain adaptation loss weights
    LAMBDA_ADV = 1.0          # Weight for adversarial domain confusion loss (DANN)
    LAMBDA_MMD = 0.5          # Weight for MMD distribution alignment loss
    LAMBDA_ORTH = 0.01        # Weight for orthogonal regularization loss

    # Gradient Reversal Layer (GRL) coefficient
    GRL_COEFF = 1.0           # Gradient reversal scaling factor

    # MMD parameters
    MMD_BANDWIDTHS = [0.5, 1.0, 2.0, 4.0]  # RBF kernel bandwidths for multi-kernel MMD
    USE_CLASS_COND_MMD = True               # Use class-conditional MMD

    # Orthogonal regularization
    USE_PROTOTYPE_LOSS = False              # Include prototype alignment loss (optional)

    # Ramp-up schedules
    RAMPUP_EPOCHS = 2         # Faster ramp-up for testing
    RAMPUP_LAMBDA_ADV = True  # Ramp up LAMBDA_ADV from 0 to final value
    RAMPUP_LAMBDA_MMD = True  # Ramp up LAMBDA_MMD from 0 to final value
    RAMPUP_GRL_COEFF = True   # Ramp up GRL_COEFF from 0 to final value

    # Domain discriminator parameters
    USE_SPECTRAL_NORM = True  # Use spectral normalization in domain discriminator
    DOMAIN_DROPOUT = 0.3      # Dropout rate for domain discriminator

    # Domain label smoothing (for stability)
    DOMAIN_LABEL_SMOOTHING = 0.05  # Smoothing factor for domain labels

    # Gradient clipping (for stability)
    USE_GRADIENT_CLIPPING = True
    GRADIENT_CLIP_MAX_NORM = 5.0

    # Validation and early stopping (for domain adaptation)
    DA_EARLY_STOPPING_METRIC = 'target_f1'  # Metric to use

    # ======================================================================
