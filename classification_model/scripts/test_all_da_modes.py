"""
Test Script for All Domain Adaptation Modes
Tests supervised, unsupervised, and semi-supervised DA with dummy data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.models import BMA_MIL_Classifier, DomainAdaptationModel
from src.data.dataset import create_bag_dataset_from_piles
from src.feature_extractor import FeatureExtractor
from src.utils import train_model_domain_adaptation, train_one_epoch_domain_adaptation
from src.utils.semi_supervised_utils import (
    split_semi_supervised_piles,
    create_semi_supervised_info,
    print_semi_supervised_split_info
)
from configs.config import Config


def generate_dummy_csv(num_piles=6, images_per_pile=3, num_classes=3, filename='dummy_data.csv'):
    """Generate dummy CSV for testing"""
    data = []
    for pile_idx in range(num_piles):
        pile_id = f"pile_{pile_idx+1:03d}"
        class_label = (pile_idx % num_classes) + 1  # 1-indexed
        for img_idx in range(images_per_pile):
            data.append({
                'Sl': len(data) + 1,
                'pile': pile_id,
                'image_path': f"{pile_id}_img_{img_idx+1}.jpg",
                'BMA_label': class_label
            })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {len(df)} images, {num_piles} piles")
    return df


def create_dummy_images(df, image_dir):
    """Create dummy images for testing"""
    os.makedirs(image_dir, exist_ok=True)

    from PIL import Image
    for img_path in df['image_path'].unique():
        full_path = os.path.join(image_dir, img_path)
        if not os.path.exists(full_path):
            # Create 4032x3024 dummy image
            img = Image.new('RGB', (4032, 3024), color=(100, 150, 200))
            img.save(full_path)

    print(f"Created {len(df['image_path'].unique())} dummy images in {image_dir}")


def test_supervised_da(config):
    """Test supervised domain adaptation"""
    print("\n" + "="*80)
    print("TEST 1: SUPERVISED DOMAIN ADAPTATION")
    print("="*80)

    # Generate data
    qld1_df = generate_dummy_csv(num_piles=6, images_per_pile=2, filename='data/test_qld1.csv')
    qld2_df = generate_dummy_csv(num_piles=6, images_per_pile=2, filename='data/test_qld2.csv')

    create_dummy_images(qld1_df, 'data/test_qld1_images')
    create_dummy_images(qld2_df, 'data/test_qld2_images')

    # Split data
    qld1_piles = qld1_df['pile'].unique().tolist()
    qld2_piles = qld2_df['pile'].unique().tolist()

    qld1_train = qld1_piles[:4]
    qld1_val = qld1_piles[4:]
    qld2_train = qld2_piles[:4]
    qld2_val = qld2_piles[4:]

    # Create datasets (both labeled)
    source_train = create_bag_dataset_from_piles(
        qld1_df, qld1_train, 'data/test_qld1_images',
        is_training=True, is_labeled=True
    )
    source_val = create_bag_dataset_from_piles(
        qld1_df, qld1_val, 'data/test_qld1_images',
        is_training=False, is_labeled=True
    )
    target_train = create_bag_dataset_from_piles(
        qld2_df, qld2_train, 'data/test_qld2_images',
        is_training=True, is_labeled=True
    )
    target_val = create_bag_dataset_from_piles(
        qld2_df, qld2_val, 'data/test_qld2_images',
        is_training=False, is_labeled=True
    )

    # Create dataloaders
    source_train_loader = DataLoader(source_train, batch_size=2, shuffle=True)
    source_val_loader = DataLoader(source_val, batch_size=2, shuffle=False)
    target_train_loader = DataLoader(target_train, batch_size=2, shuffle=True)
    target_val_loader = DataLoader(target_val, batch_size=2, shuffle=False)

    # Create model
    feature_extractor = FeatureExtractor(config.FEATURE_EXTRACTOR_MODEL, trainable_layers=0)
    base_model = BMA_MIL_Classifier(
        feature_extractor=feature_extractor,
        feature_dim=config.FEATURE_DIM,
        hidden_dim=config.IMAGE_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT_RATE
    )
    model = DomainAdaptationModel(
        base_model=base_model,
        feature_dim=config.IMAGE_HIDDEN_DIM,
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        dropout=config.DOMAIN_DROPOUT
    ).to(config.DEVICE)

    # Override config for testing
    test_config = Config()
    test_config.DA_MODE = 'supervised'
    test_config.NUM_EPOCHS = 2
    test_config.DEVICE = config.DEVICE

    # Train
    print("\n[SUPERVISED] Training for 2 epochs...")
    try:
        history = train_model_domain_adaptation(
            model, source_train_loader, source_val_loader,
            target_train_loader, target_val_loader,
            num_epochs=2, learning_rate=1e-4, config=test_config
        )
        print("[SUPERVISED] ✓ Training completed successfully!")
        return True
    except Exception as e:
        print(f"[SUPERVISED] ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unsupervised_da(config):
    """Test unsupervised domain adaptation"""
    print("\n" + "="*80)
    print("TEST 2: UNSUPERVISED DOMAIN ADAPTATION")
    print("="*80)

    # Generate data
    qld1_df = generate_dummy_csv(num_piles=6, images_per_pile=2, filename='data/test_qld1.csv')
    qld2_df = generate_dummy_csv(num_piles=6, images_per_pile=2, filename='data/test_qld2.csv')

    create_dummy_images(qld1_df, 'data/test_qld1_images')
    create_dummy_images(qld2_df, 'data/test_qld2_images')

    # Split data
    qld1_piles = qld1_df['pile'].unique().tolist()
    qld2_piles = qld2_df['pile'].unique().tolist()

    qld1_train = qld1_piles[:4]
    qld1_val = qld1_piles[4:]
    qld2_train = qld2_piles[:4]
    qld2_val = qld2_piles[4:]

    # Create datasets (source labeled, target UNLABELED)
    source_train = create_bag_dataset_from_piles(
        qld1_df, qld1_train, 'data/test_qld1_images',
        is_training=True, is_labeled=True
    )
    source_val = create_bag_dataset_from_piles(
        qld1_df, qld1_val, 'data/test_qld1_images',
        is_training=False, is_labeled=True
    )
    target_train = create_bag_dataset_from_piles(
        qld2_df, qld2_train, 'data/test_qld2_images',
        is_training=True, is_labeled=False  # UNLABELED
    )
    target_val = create_bag_dataset_from_piles(
        qld2_df, qld2_val, 'data/test_qld2_images',
        is_training=False, is_labeled=False  # UNLABELED
    )

    # Create dataloaders
    source_train_loader = DataLoader(source_train, batch_size=2, shuffle=True)
    source_val_loader = DataLoader(source_val, batch_size=2, shuffle=False)
    target_train_loader = DataLoader(target_train, batch_size=2, shuffle=True)
    target_val_loader = DataLoader(target_val, batch_size=2, shuffle=False)

    # Create model
    feature_extractor = FeatureExtractor(config.FEATURE_EXTRACTOR_MODEL, trainable_layers=0)
    base_model = BMA_MIL_Classifier(
        feature_extractor=feature_extractor,
        feature_dim=config.FEATURE_DIM,
        hidden_dim=config.IMAGE_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT_RATE
    )
    model = DomainAdaptationModel(
        base_model=base_model,
        feature_dim=config.IMAGE_HIDDEN_DIM,
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        dropout=config.DOMAIN_DROPOUT
    ).to(config.DEVICE)

    # Override config for testing
    test_config = Config()
    test_config.DA_MODE = 'unsupervised'
    test_config.NUM_EPOCHS = 2
    test_config.DEVICE = config.DEVICE
    test_config.USE_CLASS_COND_MMD = False  # Must be False for unsupervised

    # Train
    print("\n[UNSUPERVISED] Training for 2 epochs...")
    try:
        history = train_model_domain_adaptation(
            model, source_train_loader, source_val_loader,
            target_train_loader, target_val_loader,
            num_epochs=2, learning_rate=1e-4, config=test_config
        )
        print("[UNSUPERVISED] ✓ Training completed successfully!")
        return True
    except Exception as e:
        print(f"[UNSUPERVISED] ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_semi_supervised_da(config):
    """Test semi-supervised domain adaptation"""
    print("\n" + "="*80)
    print("TEST 3: SEMI-SUPERVISED DOMAIN ADAPTATION")
    print("="*80)

    # Generate data
    qld1_df = generate_dummy_csv(num_piles=9, images_per_pile=2, filename='data/test_qld1.csv')
    qld2_df = generate_dummy_csv(num_piles=9, images_per_pile=2, filename='data/test_qld2.csv')

    create_dummy_images(qld1_df, 'data/test_qld1_images')
    create_dummy_images(qld2_df, 'data/test_qld2_images')

    # Split data
    qld1_piles = qld1_df['pile'].unique().tolist()
    qld2_piles = qld2_df['pile'].unique().tolist()

    qld1_train = qld1_piles[:6]
    qld1_val = qld1_piles[6:]
    qld2_train = qld2_piles[:6]
    qld2_val = qld2_piles[6:]

    # For target: split into labeled and unlabeled
    qld2_train_labeled, qld2_train_unlabeled = split_semi_supervised_piles(
        qld2_train, qld2_df, labeled_ratio=0.33, random_seed=42
    )

    qld2_val_labeled, qld2_val_unlabeled = split_semi_supervised_piles(
        qld2_val, qld2_df, labeled_ratio=0.33, random_seed=42
    )

    # Print split info
    train_info = create_semi_supervised_info(qld2_train_labeled, qld2_train_unlabeled, qld2_df)
    print_semi_supervised_split_info(train_info, "Target Train")

    # Create datasets
    source_train = create_bag_dataset_from_piles(
        qld1_df, qld1_train, 'data/test_qld1_images',
        is_training=True, is_labeled=True
    )
    source_val = create_bag_dataset_from_piles(
        qld1_df, qld1_val, 'data/test_qld1_images',
        is_training=False, is_labeled=True
    )

    # Target: labeled subset
    target_train_labeled = create_bag_dataset_from_piles(
        qld2_df, qld2_train_labeled, 'data/test_qld2_images',
        is_training=True, is_labeled=True
    )
    target_val_labeled = create_bag_dataset_from_piles(
        qld2_df, qld2_val_labeled, 'data/test_qld2_images',
        is_training=False, is_labeled=True
    )

    # Target: unlabeled subset
    target_train_unlabeled = create_bag_dataset_from_piles(
        qld2_df, qld2_train_unlabeled, 'data/test_qld2_images',
        is_training=True, is_labeled=False
    )

    # Create dataloaders
    source_train_loader = DataLoader(source_train, batch_size=2, shuffle=True)
    source_val_loader = DataLoader(source_val, batch_size=2, shuffle=False)
    target_train_labeled_loader = DataLoader(target_train_labeled, batch_size=2, shuffle=True)
    target_val_labeled_loader = DataLoader(target_val_labeled, batch_size=2, shuffle=False)
    target_train_unlabeled_loader = DataLoader(target_train_unlabeled, batch_size=2, shuffle=True)

    # Create model
    feature_extractor = FeatureExtractor(config.FEATURE_EXTRACTOR_MODEL, trainable_layers=0)
    base_model = BMA_MIL_Classifier(
        feature_extractor=feature_extractor,
        feature_dim=config.FEATURE_DIM,
        hidden_dim=config.IMAGE_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT_RATE
    )
    model = DomainAdaptationModel(
        base_model=base_model,
        feature_dim=config.IMAGE_HIDDEN_DIM,
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        dropout=config.DOMAIN_DROPOUT
    ).to(config.DEVICE)

    # Override config for testing
    test_config = Config()
    test_config.DA_MODE = 'semi_supervised'
    test_config.NUM_EPOCHS = 2
    test_config.DEVICE = config.DEVICE
    test_config.SSDA_LABELED_RATIO = 0.33

    # Train
    print("\n[SEMI-SUPERVISED] Training for 2 epochs...")
    try:
        history = train_model_domain_adaptation(
            model, source_train_loader, source_val_loader,
            target_train_labeled_loader, target_val_labeled_loader,
            num_epochs=2, learning_rate=1e-4, config=test_config,
            target_train_loader_unlabeled=target_train_unlabeled_loader
        )
        print("[SEMI-SUPERVISED] ✓ Training completed successfully!")
        return True
    except Exception as e:
        print(f"[SEMI-SUPERVISED] ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TESTING ALL DOMAIN ADAPTATION MODES")
    print("="*80)

    # Setup
    config = Config()
    config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {config.DEVICE}")

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Run tests
    results = {}

    results['supervised'] = test_supervised_da(config)
    results['unsupervised'] = test_unsupervised_da(config)
    results['semi_supervised'] = test_semi_supervised_da(config)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for mode, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{mode.upper():20s}: {status}")

    # Overall result
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
