"""
Configuration file for BMA MIL Classifier
"""

import torch


class Config:
    # Data parameters
    DATA_PATH = 'data/CVM_label_data.csv'
    # IMAGE_DIR can be:
    # - Relative path: 'data/images'
    # - Absolute Windows path: r'C:\Users\YourName\Pictures\pile_images'
    # - Absolute Windows path: 'C:/Users/YourName/Pictures/pile_images'
    # All images referenced in the CSV should be in this single folder
    IMAGE_DIR = r'D:\SCANDY\Data\CVM_Data'  # Update this to your image directory (supports Windows paths)
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

    # Training parameters
    BATCH_SIZE = 6
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.3
    
    # Optimizer and Scheduler
    USE_ADAMW = True  # Use AdamW optimizer (improved Adam with better weight decay)
    USE_LR_SCHEDULER = True  # Enable learning rate scheduler
    LR_SCHEDULER_TYPE = 'reduce_on_plateau'  # 'reduce_on_plateau' or 'cosine_annealing'
    
    # ReduceLROnPlateau parameters
    LR_SCHEDULER_MODE = 'min'  # 'min' for loss, 'max' for accuracy
    LR_SCHEDULER_FACTOR = 0.5  # Reduce LR by this factor
    LR_SCHEDULER_PATIENCE = 5  # Number of epochs with no improvement before reducing LR
    LR_SCHEDULER_MIN_LR = 1e-7  # Minimum learning rate
    LR_SCHEDULER_THRESHOLD = 1e-4  # Threshold for measuring improvement
    
    # Training level
    TRAINING_LEVEL = 'bag'  # 'bag' or 'pile'
    # 'bag': Train on individual images (bags), validate on piles (aggregated)
    # 'pile': Train on entire piles, validate on piles (both pile-level)
    
    # Pile-level aggregation method (only used when TRAINING_LEVEL = 'pile')
    POOLING_METHOD = 'mean'  # 'mean', 'max', 'attention', or 'majority'
    # NOTE: This setting is IGNORED for bag-level training
    # For bag-level training: all 3 methods (mean, max, majority) are automatically evaluated during testing
    # For pile-level training: this determines which method is used during training and validation
    
    # Class imbalance handling
    USE_WEIGHTED_LOSS = True

    # Data split (pile-level)
    SPLIT_MODE = 'kfold'  # 'standard' or 'kfold'
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    RANDOM_STATE = 42
    
    # Cross-validation (pile-level)
    NUM_FOLDS = 3  # Only used when SPLIT_MODE = 'kfold'

    # Feature extractor (integrated into model)
    FEATURE_EXTRACTOR_MODEL = 'vit_base_r50_s16_224.orig_in21k'
    
    # Feature extractor training mode:
    # TRAINABLE_FEATURE_LAYERS:
    #   - 0: Fully frozen (no gradients)
    #   - -1: Fully trainable (all layers)
    #   - N (1-12 for ViT): Last N blocks/layers trainable, rest frozen
    TRAINABLE_FEATURE_LAYERS = 2  # -1=all trainable, 0=frozen, N=last N layers trainable

    # Paths
    BEST_MODEL_PATH = 'models/best_bma_mil_model.pth'
    TRAINING_PLOT_PATH = 'results/training_history.png'
    LOG_DIR = 'logs'

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    #DEVICE = 'cpu'

    # Checkpointing / Resume
    RESUME_TRAINING = True
    CHECKPOINT_PATH = BEST_MODEL_PATH
    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Evaluation
    EVAL_ON_PILE_LEVEL = True  # Aggregate bag predictions to pile level during validation
    
    # Logging
    ENABLE_LOGGING = True
    LOG_LEVEL = 'INFO'
    
    # Data Augmentation Parameters
    HISTOGRAM_METHOD = 'clahe'

    # Augmentation strategy
    INCLUDE_ORIGINAL_AND_AUGMENTED = True  # If True: includes original patches
                                            # If False: only augmented patches
    NUM_AUGMENTATION_VERSIONS = 3           # Number of augmented versions per patch
                                            # 1: 12 patches (standard)
                                            # 2: 24 patches (12 original + 12 augmented) if INCLUDE_ORIGINAL_AND_AUGMENTED=True
                                            # 3: 36 patches (12 original + 36 augmented) if INCLUDE_ORIGINAL_AND_AUGMENTED=True
                                            #    OR 36 patches (only augmented) if INCLUDE_ORIGINAL_AND_AUGMENTED=False

    # Enable/disable augmentation types
    ENABLE_GEOMETRIC_AUG = True
    ENABLE_COLOR_AUG = False
    ENABLE_NOISE_AUG = False

    # Geometric augmentation parameters
    ROTATION_RANGE = 15
    ZOOM_RANGE = (0.9, 2.5)
    #SHEAR_RANGE = 10
    HORIZONTAL_FLIP = True
    #VERTICAL_FLIP = True
    #GEOMETRIC_PROB = 0.5

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
    USE_DOMAIN_ADAPTATION = False  # Set to True to enable domain adaptation

    # Domain data paths (QLD1 = source, QLD2 = target)
    QLD1_DATA_PATH = 'data/qld1_data.csv'  # Source domain data
    QLD2_DATA_PATH = 'data/qld2_data.csv'  # Target domain data
    QLD1_IMAGE_DIR = 'data/qld1_images'    # Source domain images
    QLD2_IMAGE_DIR = 'data/qld2_images'    # Target domain images

    # Domain adaptation loss weights
    LAMBDA_ADV = 1.0          # Weight for adversarial domain confusion loss (DANN)
    LAMBDA_MMD = 0.5          # Weight for MMD distribution alignment loss
    LAMBDA_ORTH = 0.01        # Weight for orthogonal regularization loss

    # Gradient Reversal Layer (GRL) coefficient
    GRL_COEFF = 1.0           # Gradient reversal scaling factor (tied to LAMBDA_ADV)

    # MMD parameters
    MMD_BANDWIDTHS = [0.5, 1.0, 2.0, 4.0]  # RBF kernel bandwidths for multi-kernel MMD
    USE_CLASS_COND_MMD = True               # Use class-conditional MMD (recommended when target is labeled)

    # Orthogonal regularization
    USE_PROTOTYPE_LOSS = False              # Include prototype alignment loss (optional)

    # Ramp-up schedules (gradual increase over first N epochs)
    RAMPUP_EPOCHS = 5         # Number of epochs for ramping up lambda values
    RAMPUP_LAMBDA_ADV = True  # Ramp up LAMBDA_ADV from 0 to final value
    RAMPUP_LAMBDA_MMD = True  # Ramp up LAMBDA_MMD from 0 to final value
    RAMPUP_GRL_COEFF = True   # Ramp up GRL_COEFF from 0 to final value

    # Domain discriminator parameters
    USE_SPECTRAL_NORM = True  # Use spectral normalization in domain discriminator
    DOMAIN_DROPOUT = 0.3      # Dropout rate for domain discriminator

    # Domain label smoothing (for stability)
    DOMAIN_LABEL_SMOOTHING = 0.05  # Smoothing factor for domain labels (0.0 to 0.5)

    # Gradient clipping (for stability)
    USE_GRADIENT_CLIPPING = True
    GRADIENT_CLIP_MAX_NORM = 5.0

    # Validation and early stopping (for domain adaptation)
    DA_EARLY_STOPPING_METRIC = 'target_f1'  # Metric to use: 'target_f1', 'target_acc', 'source_f1'

    # ======================================================================
