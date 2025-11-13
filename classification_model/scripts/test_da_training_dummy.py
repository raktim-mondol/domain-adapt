"""
Quick test of domain adaptation training with dummy data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Use dummy config
import configs.config_dummy as config_module
Config = config_module.Config

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.models import BMA_MIL_Classifier, DomainAdaptationModel
from src.data.dataset import create_bag_dataset_from_piles
from src.feature_extractor import FeatureExtractor
from src.augmentation import get_augmentation_pipeline
from src.utils import train_model_domain_adaptation, compute_class_weights

print(f"\n{'='*80}")
print("Quick Domain Adaptation Training Test with Dummy Data")
print(f"{'='*80}")
print(f"Device: {Config.DEVICE}")
print(f"Epochs: {Config.NUM_EPOCHS}")
print(f"Batch Size: {Config.BATCH_SIZE}")
print(f"{'='*80}\n")

# Load dummy data
print("Loading dummy data...")
source_df = pd.read_csv(Config.QLD1_DATA_PATH)
target_df = pd.read_csv(Config.QLD2_DATA_PATH)

# Clean data
source_df = source_df[source_df['BMA_label'] != 'BMA_label']
source_df['BMA_label'] = source_df['BMA_label'].astype(int)
target_df = target_df[target_df['BMA_label'] != 'BMA_label']
target_df['BMA_label'] = target_df['BMA_label'].astype(int)

print(f"  Source (QLD1): {len(source_df)} images, {source_df['pile'].nunique()} piles")
print(f"  Target (QLD2): {len(target_df)} images, {target_df['pile'].nunique()} piles")

# Simple split
from sklearn.model_selection import train_test_split

# Source domain split
source_piles = source_df['pile'].unique()
source_train_piles, source_val_piles = train_test_split(
    source_piles, test_size=0.3, random_state=42
)

# Target domain split
target_piles = target_df['pile'].unique()
target_train_piles, target_val_piles = train_test_split(
    target_piles, test_size=0.3, random_state=42
)

print(f"\nSplits:")
print(f"  Source: {len(source_train_piles)} train, {len(source_val_piles)} val piles")
print(f"  Target: {len(target_train_piles)} train, {len(target_val_piles)} val piles")

# Create datasets
print("\nCreating datasets...")
augmentation = get_augmentation_pipeline(is_training=False, target_size=224)  # No aug for speed

source_train_dataset = create_bag_dataset_from_piles(
    source_df, list(source_train_piles), Config.QLD1_IMAGE_DIR,
    augmentation=None, is_training=True, max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
    include_original_and_augmented=False, num_augmentation_versions=1
)

source_val_dataset = create_bag_dataset_from_piles(
    source_df, list(source_val_piles), Config.QLD1_IMAGE_DIR,
    augmentation=None, is_training=False, max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
    include_original_and_augmented=False, num_augmentation_versions=1
)

target_train_dataset = create_bag_dataset_from_piles(
    target_df, list(target_train_piles), Config.QLD2_IMAGE_DIR,
    augmentation=None, is_training=True, max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
    include_original_and_augmented=False, num_augmentation_versions=1
)

target_val_dataset = create_bag_dataset_from_piles(
    target_df, list(target_val_piles), Config.QLD2_IMAGE_DIR,
    augmentation=None, is_training=False, max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
    include_original_and_augmented=False, num_augmentation_versions=1
)

print(f"  Source train: {len(source_train_dataset)} bags")
print(f"  Source val:   {len(source_val_dataset)} bags")
print(f"  Target train: {len(target_train_dataset)} bags")
print(f"  Target val:   {len(target_val_dataset)} bags")

# Create dataloaders
source_train_loader = DataLoader(source_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
source_val_loader = DataLoader(source_val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
target_train_loader = DataLoader(target_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
target_val_loader = DataLoader(target_val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# Create model
print("\nCreating model...")
feature_extractor = FeatureExtractor(device='cpu', trainable_layers=0)

base_model = BMA_MIL_Classifier(
    feature_extractor=feature_extractor.model,
    feature_dim=Config.FEATURE_DIM,
    hidden_dim=Config.IMAGE_HIDDEN_DIM,
    num_classes=Config.NUM_CLASSES,
    dropout=Config.DROPOUT_RATE,
    trainable_layers=Config.TRAINABLE_FEATURE_LAYERS
)

model = DomainAdaptationModel(
    base_model=base_model,
    feature_dim=Config.IMAGE_HIDDEN_DIM,
    grl_lambda=Config.GRL_COEFF,
    use_spectral_norm=Config.USE_SPECTRAL_NORM,
    dropout=Config.DOMAIN_DROPOUT
)

model = model.to(Config.DEVICE)

print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Compute class weights
source_train_df = source_df[source_df['pile'].isin(source_train_piles)]
class_weights = compute_class_weights(source_train_df, Config.NUM_CLASSES, Config.DEVICE)

# Train
print(f"\n{'='*80}")
print("Starting Training (Quick Test)")
print(f"{'='*80}\n")

try:
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

    print(f"\n{'='*80}")
    print("Training Test Complete!")
    print(f"{'='*80}")
    print(f"\nFinal Results:")
    print(f"  Source Domain - Acc: {history['source_val_acc'][-1]:.4f}, F1: {history['source_val_f1'][-1]:.4f}")
    print(f"  Target Domain - Acc: {history['target_val_acc'][-1]:.4f}, F1: {history['target_val_f1'][-1]:.4f}")
    print(f"\nBest Target F1: {max(history['target_val_f1']):.4f}")
    print(f"{'='*80}\n")

    print("✓ Domain adaptation training pipeline working correctly!")
    print("✓ All components integrated successfully!")
    print("✓ Ready for real data training!")

except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
