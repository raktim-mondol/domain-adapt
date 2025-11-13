"""
Domain Adaptation Training Script
Trains BMA MIL Classifier with domain adaptation between QLD1 (source) and QLD2 (target)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models import BMA_MIL_Classifier, DomainAdaptationModel
from src.data.dataset import create_bag_dataset_from_piles
from src.feature_extractor import FeatureExtractor
from src.augmentation import get_augmentation_pipeline
from src.utils import (
    train_model_domain_adaptation,
    compute_class_weights,
    setup_logging
)
from configs.config import Config


def load_domain_data(data_path, image_dir, domain_name):
    """
    Load and prepare data for a domain

    Args:
        data_path: Path to CSV file
        image_dir: Path to image directory
        domain_name: Name of domain (for logging)

    Returns:
        df: Processed DataFrame
    """
    print(f"\nLoading {domain_name} domain data...")
    df = pd.read_csv(data_path)

    # Clean data
    df = df[df['BMA_label'] != 'BMA_label']
    df['BMA_label'] = df['BMA_label'].astype(int)

    # Filter out class 4 if present
    if 4 in df['BMA_label'].unique():
        df = df[df['BMA_label'] != 4]
        print(f"  Filtered out class 4 from {domain_name}")

    print(f"  {domain_name} - Images: {len(df)}, Piles: {df['pile'].nunique()}")
    print(f"  Classes: {sorted(df['BMA_label'].unique())}")

    # Check class distribution
    class_dist = df.groupby('pile')['BMA_label'].first().value_counts().sort_index()
    print(f"  Pile-level class distribution:")
    for cls, count in class_dist.items():
        print(f"    Class {cls}: {count} piles")

    return df


def split_domain_data(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    Split domain data into train/val/test sets at pile level

    Args:
        df: DataFrame with pile and label columns
        train_ratio: Ratio for training
        val_ratio: Ratio for validation
        test_ratio: Ratio for testing
        random_state: Random seed

    Returns:
        train_piles, val_piles, test_piles: Lists of pile IDs
    """
    from sklearn.model_selection import train_test_split

    # Get unique piles and labels
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
    unique_piles = pile_labels['pile'].values
    labels = pile_labels['BMA_label'].values - 1  # 0-indexed

    # Split
    try:
        # First split: train vs (val+test)
        train_piles, temp_piles, _, temp_labels = train_test_split(
            unique_piles, labels,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=labels
        )

        # Second split: val vs test
        val_ratio_adj = val_ratio / (val_ratio + test_ratio)
        val_piles, test_piles, _, _ = train_test_split(
            temp_piles, temp_labels,
            test_size=(1 - val_ratio_adj),
            random_state=random_state,
            stratify=temp_labels
        )

        print("  Stratified split successful")

    except ValueError as e:
        print(f"  [WARNING] Stratified split failed: {e}")
        print("  Using random split instead...")

        # Fallback to random split
        train_piles, temp_piles = train_test_split(
            unique_piles,
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )

        val_ratio_adj = val_ratio / (val_ratio + test_ratio)
        val_piles, test_piles = train_test_split(
            temp_piles,
            test_size=(1 - val_ratio_adj),
            random_state=random_state
        )

    print(f"  Split: {len(train_piles)} train, {len(val_piles)} val, {len(test_piles)} test piles")

    return list(train_piles), list(val_piles), list(test_piles)


def create_domain_dataloaders(df, train_piles, val_piles, image_dir,
                              augmentation=None, batch_size=1):
    """
    Create dataloaders for a domain

    Args:
        df: DataFrame
        train_piles: Training pile IDs
        val_piles: Validation pile IDs
        image_dir: Image directory
        augmentation: Augmentation pipeline
        batch_size: Batch size

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = create_bag_dataset_from_piles(
        df, train_piles, image_dir,
        augmentation=augmentation,
        is_training=True,
        max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
        include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
        num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
    )

    val_dataset = create_bag_dataset_from_piles(
        df, val_piles, image_dir,
        augmentation=None,
        is_training=False,
        max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
        include_original_and_augmented=False,
        num_augmentation_versions=1
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def plot_domain_adaptation_history(history, save_path='results/da_training_history.png'):
    """
    Plot training history for domain adaptation

    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training loss
    axes[0, 0].plot(history['train_loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # Plot 2: Source domain accuracy
    axes[0, 1].plot(history['source_val_acc'], label='Source Acc', color='blue')
    axes[0, 1].plot(history['source_val_f1'], label='Source F1', color='cyan')
    axes[0, 1].set_title('Source Domain Performance')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Target domain accuracy
    axes[1, 0].plot(history['target_val_acc'], label='Target Acc', color='red')
    axes[1, 0].plot(history['target_val_f1'], label='Target F1', color='orange')
    axes[1, 0].set_title('Target Domain Performance')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Comparison
    axes[1, 1].plot(history['source_val_f1'], label='Source F1', color='blue')
    axes[1, 1].plot(history['target_val_f1'], label='Target F1', color='red')
    axes[1, 1].set_title('F1-Score Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"\n[PLOT] Training history saved to {save_path}")


def main():
    """
    Main domain adaptation training pipeline
    """
    print(f"\n{'='*80}")
    print("BMA MIL Classifier - Domain Adaptation Training")
    print(f"{'='*80}")
    print("Architecture: DANN + MMD + Orthogonal Regularization")
    print(f"Source Domain: QLD1")
    print(f"Target Domain: QLD2")
    print(f"Device: {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")

    # Setup logging
    logger = setup_logging(log_dir=Config.LOG_DIR, mode='domain_adaptation')

    # Check if domain adaptation is enabled
    if not Config.USE_DOMAIN_ADAPTATION:
        print("[WARNING] Domain adaptation is disabled in config!")
        print("Set Config.USE_DOMAIN_ADAPTATION = True to enable.")
        return

    # Load data for both domains
    source_df = load_domain_data(Config.QLD1_DATA_PATH, Config.QLD1_IMAGE_DIR, 'QLD1 (Source)')
    target_df = load_domain_data(Config.QLD2_DATA_PATH, Config.QLD2_IMAGE_DIR, 'QLD2 (Target)')

    # Split both domains
    print(f"\n{'='*80}")
    print("Splitting Domains")
    print(f"{'='*80}")

    print("\nSource Domain (QLD1) Split:")
    source_train_piles, source_val_piles, source_test_piles = split_domain_data(
        source_df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
        random_state=Config.RANDOM_STATE
    )

    print("\nTarget Domain (QLD2) Split:")
    target_train_piles, target_val_piles, target_test_piles = split_domain_data(
        target_df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
        random_state=Config.RANDOM_STATE
    )

    # Setup augmentation
    augmentation = get_augmentation_pipeline(is_training=True, target_size=224)

    # Create dataloaders
    print(f"\n{'='*80}")
    print("Creating DataLoaders")
    print(f"{'='*80}")

    print("\nSource Domain (QLD1):")
    source_train_loader, source_val_loader = create_domain_dataloaders(
        source_df, source_train_piles, source_val_piles,
        Config.QLD1_IMAGE_DIR, augmentation, batch_size=Config.BATCH_SIZE
    )
    print(f"  Train: {len(source_train_loader.dataset)} bags")
    print(f"  Val:   {len(source_val_loader.dataset)} bags")

    print("\nTarget Domain (QLD2):")
    target_train_loader, target_val_loader = create_domain_dataloaders(
        target_df, target_train_piles, target_val_piles,
        Config.QLD2_IMAGE_DIR, augmentation, batch_size=Config.BATCH_SIZE
    )
    print(f"  Train: {len(target_train_loader.dataset)} bags")
    print(f"  Val:   {len(target_val_loader.dataset)} bags")

    # Initialize feature extractor
    print(f"\n{'='*80}")
    print("Initializing Model")
    print(f"{'='*80}")
    print(f"\nFeature Extractor: {Config.FEATURE_EXTRACTOR_MODEL}")

    feature_extractor = FeatureExtractor(
        device='cpu',
        trainable_layers=0
    )

    # Create base MIL model
    base_model = BMA_MIL_Classifier(
        feature_extractor=feature_extractor.model,
        feature_dim=Config.FEATURE_DIM,
        hidden_dim=Config.IMAGE_HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT_RATE,
        trainable_layers=Config.TRAINABLE_FEATURE_LAYERS
    )

    # Wrap with domain adaptation
    model = DomainAdaptationModel(
        base_model=base_model,
        feature_dim=Config.IMAGE_HIDDEN_DIM,  # Use hidden_dim from aggregator
        grl_lambda=Config.GRL_COEFF,
        use_spectral_norm=Config.USE_SPECTRAL_NORM,
        dropout=Config.DOMAIN_DROPOUT
    )

    model = model.to(Config.DEVICE)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print(f"\nDomain Adaptation Settings:")
    print(f"  LAMBDA_ADV:  {Config.LAMBDA_ADV}")
    print(f"  LAMBDA_MMD:  {Config.LAMBDA_MMD}")
    print(f"  LAMBDA_ORTH: {Config.LAMBDA_ORTH}")
    print(f"  GRL_COEFF:   {Config.GRL_COEFF}")
    print(f"  MMD Bandwidths: {Config.MMD_BANDWIDTHS}")
    print(f"  Class-Conditional MMD: {Config.USE_CLASS_COND_MMD}")
    print(f"  Ramp-up Epochs: {Config.RAMPUP_EPOCHS}")

    # Compute class weights (from source domain)
    class_weights = None
    if Config.USE_WEIGHTED_LOSS:
        source_train_df = source_df[source_df['pile'].isin(source_train_piles)]
        class_weights = compute_class_weights(
            source_train_df, Config.NUM_CLASSES, Config.DEVICE
        )

    # Train
    print(f"\n{'='*80}")
    print("Starting Domain Adaptation Training")
    print(f"{'='*80}\n")

    history = train_model_domain_adaptation(
        model=model,
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_train_loader=target_train_loader,
        target_val_loader=target_val_loader,
        num_epochs=Config.NUM_EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        config=Config,
        class_weights=class_weights,
        fold=None,
        resume_state=None
    )

    # Plot results
    plot_domain_adaptation_history(history)

    # Print final results
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\nFinal Results:")
    print(f"  Source Domain - Acc: {history['source_val_acc'][-1]:.4f}, F1: {history['source_val_f1'][-1]:.4f}")
    print(f"  Target Domain - Acc: {history['target_val_acc'][-1]:.4f}, F1: {history['target_val_f1'][-1]:.4f}")
    print(f"\nBest Target F1: {max(history['target_val_f1']):.4f} (Epoch {np.argmax(history['target_val_f1'])+1})")
    print(f"{'='*80}\n")

    if logger and logger.hasHandlers():
        logger.info(f"Training complete - Best Target F1: {max(history['target_val_f1']):.4f}")


if __name__ == '__main__':
    main()
