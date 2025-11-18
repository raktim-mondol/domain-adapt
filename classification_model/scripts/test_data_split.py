"""
Test script for new data split configuration functionality
Tests both percentage-based and per-class count-based splitting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.utils import split_domain_data, validate_split
from configs.config import Config


def create_dummy_data(n_piles_per_class=20, n_images_per_pile=5):
    """Create dummy data for testing"""
    print(f"Creating dummy data with {n_piles_per_class} piles per class...")

    data = []
    pile_id = 1

    for cls in [1, 2, 3]:  # 3 classes
        for _ in range(n_piles_per_class):
            pile_name = f"pile_{pile_id:03d}"
            for img in range(n_images_per_pile):
                data.append({
                    'Sl': len(data) + 1,
                    'pile': pile_name,
                    'image_path': f"{pile_name}_img_{img:02d}.jpg",
                    'BMA_label': cls
                })
            pile_id += 1

    df = pd.DataFrame(data)
    print(f"  Created {len(df)} images in {df['pile'].nunique()} piles")
    return df


def test_percentage_split():
    """Test percentage-based splitting"""
    print("\n" + "="*80)
    print("TEST 1: PERCENTAGE-BASED SPLITTING")
    print("="*80)

    df = create_dummy_data(n_piles_per_class=20)

    train_piles, val_piles, test_piles = split_domain_data(
        df,
        split_mode='percentage',
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        random_state=42
    )

    # Validate
    is_valid = validate_split(df, train_piles, val_piles, test_piles, verbose=True)

    if is_valid:
        print("\n✓ TEST 1 PASSED: Percentage-based split is valid!")
    else:
        print("\n✗ TEST 1 FAILED: Percentage-based split has issues!")

    return is_valid


def test_per_class_count_split():
    """Test per-class count-based splitting"""
    print("\n" + "="*80)
    print("TEST 2: PER-CLASS COUNT-BASED SPLITTING")
    print("="*80)

    df = create_dummy_data(n_piles_per_class=20)

    per_class_counts = {
        1: {'train': 12, 'val': 3, 'test': 5},
        2: {'train': 12, 'val': 3, 'test': 5},
        3: {'train': 12, 'val': 3, 'test': 5}
    }

    train_piles, val_piles, test_piles = split_domain_data(
        df,
        split_mode='per_class_count',
        per_class_counts=per_class_counts,
        random_state=42
    )

    # Validate
    is_valid = validate_split(df, train_piles, val_piles, test_piles, verbose=True)

    # Check exact counts per class
    pile_labels = df.groupby('pile')['BMA_label'].first()
    print("\nVerifying exact counts per class:")
    for cls in [1, 2, 3]:
        train_count = sum(pile_labels[pile_labels.index.isin(train_piles)] == cls)
        val_count = sum(pile_labels[pile_labels.index.isin(val_piles)] == cls)
        test_count = sum(pile_labels[pile_labels.index.isin(test_piles)] == cls)

        expected = per_class_counts[cls]
        actual = {'train': train_count, 'val': val_count, 'test': test_count}

        matches = (
            train_count == expected['train'] and
            val_count == expected['val'] and
            test_count == expected['test']
        )

        status = "✓" if matches else "✗"
        print(f"  {status} Class {cls}: Expected {expected}, Got {actual}")

    if is_valid:
        print("\n✓ TEST 2 PASSED: Per-class count split is valid!")
    else:
        print("\n✗ TEST 2 FAILED: Per-class count split has issues!")

    return is_valid


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*80)
    print("TEST 3: EDGE CASES")
    print("="*80)

    # Test 3a: Small dataset with per-class splitting
    print("\nTest 3a: Small dataset (5 piles per class)")
    df = create_dummy_data(n_piles_per_class=5)

    per_class_counts = {
        1: {'train': 3, 'val': 1, 'test': 1},
        2: {'train': 3, 'val': 1, 'test': 1},
        3: {'train': 3, 'val': 1, 'test': 1}
    }

    try:
        train_piles, val_piles, test_piles = split_domain_data(
            df,
            split_mode='per_class_count',
            per_class_counts=per_class_counts,
            random_state=42
        )
        validate_split(df, train_piles, val_piles, test_piles, verbose=False)
        print("  ✓ Small dataset split successful")
        test_3a_pass = True
    except Exception as e:
        print(f"  ✗ Small dataset split failed: {e}")
        test_3a_pass = False

    # Test 3b: Invalid counts (requesting more than available)
    print("\nTest 3b: Invalid counts (should fail gracefully)")
    df = create_dummy_data(n_piles_per_class=5)

    per_class_counts = {
        1: {'train': 10, 'val': 3, 'test': 5},  # Requesting 18 but only 5 available
        2: {'train': 10, 'val': 3, 'test': 5},
        3: {'train': 10, 'val': 3, 'test': 5}
    }

    try:
        train_piles, val_piles, test_piles = split_domain_data(
            df,
            split_mode='per_class_count',
            per_class_counts=per_class_counts,
            random_state=42
        )
        print("  ✗ Should have raised ValueError but didn't")
        test_3b_pass = False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:80]}...")
        test_3b_pass = True

    # Test 3c: Invalid ratios
    print("\nTest 3c: Invalid ratios (should fail gracefully)")
    df = create_dummy_data(n_piles_per_class=20)

    try:
        train_piles, val_piles, test_piles = split_domain_data(
            df,
            split_mode='percentage',
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.1,  # Sums to 0.9, not 1.0
            random_state=42
        )
        print("  ✗ Should have raised ValueError but didn't")
        test_3c_pass = False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:80]}...")
        test_3c_pass = True

    all_pass = test_3a_pass and test_3b_pass and test_3c_pass
    if all_pass:
        print("\n✓ TEST 3 PASSED: Edge cases handled correctly!")
    else:
        print("\n✗ TEST 3 FAILED: Some edge cases not handled correctly!")

    return all_pass


def test_config_integration():
    """Test integration with Config"""
    print("\n" + "="*80)
    print("TEST 4: CONFIGURATION INTEGRATION")
    print("="*80)

    print(f"\nCurrent Config Settings:")
    print(f"  DATA_SPLIT_MODE: {Config.DATA_SPLIT_MODE}")
    print(f"  TRAIN_RATIO: {Config.TRAIN_RATIO}")
    print(f"  VAL_RATIO: {Config.VAL_RATIO}")
    print(f"  TEST_RATIO: {Config.TEST_RATIO}")
    print(f"  QLD1_SPLIT_MODE: {Config.QLD1_SPLIT_MODE}")
    print(f"  QLD2_SPLIT_MODE: {Config.QLD2_SPLIT_MODE}")

    # Test with config values
    df = create_dummy_data(n_piles_per_class=20)

    try:
        train_piles, val_piles, test_piles = split_domain_data(
            df,
            split_mode=Config.DATA_SPLIT_MODE,
            train_ratio=Config.TRAIN_RATIO,
            val_ratio=Config.VAL_RATIO,
            test_ratio=Config.TEST_RATIO,
            per_class_counts=Config.PER_CLASS_SPLIT_COUNTS,
            random_state=Config.RANDOM_STATE
        )
        is_valid = validate_split(df, train_piles, val_piles, test_piles, verbose=False)

        if is_valid:
            print("\n✓ TEST 4 PASSED: Config integration works!")
            return True
        else:
            print("\n✗ TEST 4 FAILED: Config integration produced invalid split!")
            return False
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("DATA SPLIT CONFIGURATION - TEST SUITE")
    print("="*80)

    results = []

    # Run tests
    results.append(("Percentage Split", test_percentage_split()))
    results.append(("Per-Class Count Split", test_per_class_count_split()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Config Integration", test_config_integration()))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
