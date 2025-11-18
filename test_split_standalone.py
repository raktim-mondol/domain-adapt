"""
Standalone test for data split functionality (no torch dependency)
"""

import pandas as pd
import numpy as np
import sys
import importlib.util

# Direct import to avoid torch dependencies
spec = importlib.util.spec_from_file_location(
    'data_split_utils',
    'classification_model/src/utils/data_split_utils.py'
)
data_split_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_split_utils)

split_domain_data = data_split_utils.split_domain_data
validate_split = data_split_utils.validate_split


def create_dummy_data(n_piles_per_class=20):
    """Create dummy data for testing"""
    print(f"Creating dummy data with {n_piles_per_class} piles per class...")

    data = []
    pile_id = 1

    for cls in [1, 2, 3]:  # 3 classes
        for _ in range(n_piles_per_class):
            pile_name = f"pile_{pile_id:03d}"
            for img in range(5):
                data.append({
                    'pile': pile_name,
                    'image_path': f"{pile_name}_img_{img:02d}.jpg",
                    'BMA_label': cls
                })
            pile_id += 1

    df = pd.DataFrame(data)
    print(f"  Created {len(df)} images in {df['pile'].nunique()} piles\n")
    return df


print("="*80)
print("DATA SPLIT FUNCTIONALITY - QUICK TEST")
print("="*80)

# Test 1: Percentage split
print("\n" + "-"*80)
print("TEST 1: Percentage-based split (70/10/20)")
print("-"*80)

df = create_dummy_data(n_piles_per_class=20)
train, val, test = split_domain_data(
    df, split_mode='percentage',
    train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
    random_state=42
)
is_valid_1 = validate_split(df, train, val, test, verbose=True)

# Test 2: Per-class count split
print("\n" + "-"*80)
print("TEST 2: Per-class count split (12/3/5 per class)")
print("-"*80)

df = create_dummy_data(n_piles_per_class=20)
per_class_counts = {
    1: {'train': 12, 'val': 3, 'test': 5},
    2: {'train': 12, 'val': 3, 'test': 5},
    3: {'train': 12, 'val': 3, 'test': 5}
}
train, val, test = split_domain_data(
    df, split_mode='per_class_count',
    per_class_counts=per_class_counts,
    random_state=42
)
is_valid_2 = validate_split(df, train, val, test, verbose=True)

# Summary
print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)
print(f"  Percentage split: {'✓ PASS' if is_valid_1 else '✗ FAIL'}")
print(f"  Per-class count split: {'✓ PASS' if is_valid_2 else '✗ FAIL'}")

if is_valid_1 and is_valid_2:
    print("\n✓ ALL TESTS PASSED!")
    print("="*80 + "\n")
    sys.exit(0)
else:
    print("\n✗ SOME TESTS FAILED!")
    print("="*80 + "\n")
    sys.exit(1)
