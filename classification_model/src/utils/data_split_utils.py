"""
Flexible Data Splitting Utilities for BMA MIL Classifier
Supports both percentage-based and count-based (per-class) pile-level splitting
"""

import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Union


def split_data_percentage(
    df,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[List, List, List]:
    """
    Split domain data into train/val/test sets at pile level using percentage ratios

    Args:
        df: DataFrame with 'pile' and 'BMA_label' columns
        train_ratio: Ratio for training (default: 0.7)
        val_ratio: Ratio for validation (default: 0.1)
        test_ratio: Ratio for testing (default: 0.2)
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting (maintains class distribution)

    Returns:
        train_piles, val_piles, test_piles: Lists of pile IDs

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    # Get unique piles and labels
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
    unique_piles = pile_labels['pile'].values
    labels = pile_labels['BMA_label'].values - 1  # Convert to 0-indexed

    print(f"\nSplitting {len(unique_piles)} piles using percentage ratios:")
    print(f"  Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")

    # Try stratified split first
    if stratify:
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

            print("  ✓ Stratified split successful")

        except ValueError as e:
            print(f"  [WARNING] Stratified split failed: {e}")
            print("  Falling back to random split...")
            stratify = False

    # Fallback to random split if stratification fails or is disabled
    if not stratify:
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

        print("  ✓ Random split successful")

    # Print split statistics
    print(f"  Result: {len(train_piles)} train, {len(val_piles)} val, {len(test_piles)} test piles")

    return list(train_piles), list(val_piles), list(test_piles)


def split_data_per_class_count(
    df,
    per_class_counts: Dict[int, Dict[str, int]],
    random_state: int = 42
) -> Tuple[List, List, List]:
    """
    Split domain data into train/val/test sets using per-class pile counts

    Args:
        df: DataFrame with 'pile' and 'BMA_label' columns
        per_class_counts: Dictionary mapping class labels to split counts
                         Format: {class_label: {'train': N, 'val': N, 'test': N}}
                         Example: {1: {'train': 10, 'val': 3, 'test': 5}, ...}
        random_state: Random seed for reproducibility

    Returns:
        train_piles, val_piles, test_piles: Lists of pile IDs

    Raises:
        ValueError: If requested counts exceed available piles for any class
    """
    print(f"\nSplitting data using per-class pile counts:")

    # Get unique piles and labels
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()

    train_piles = []
    val_piles = []
    test_piles = []

    np.random.seed(random_state)

    # Process each class
    for class_label, counts in sorted(per_class_counts.items()):
        # Get piles for this class
        class_piles = pile_labels[pile_labels['BMA_label'] == class_label]['pile'].values
        num_available = len(class_piles)

        # Validate counts
        num_requested = counts['train'] + counts['val'] + counts['test']
        if num_requested > num_available:
            raise ValueError(
                f"Class {class_label}: Requested {num_requested} piles "
                f"(train={counts['train']}, val={counts['val']}, test={counts['test']}) "
                f"but only {num_available} available"
            )

        # Shuffle piles for this class
        shuffled_piles = np.random.permutation(class_piles)

        # Split according to counts
        train_end = counts['train']
        val_end = train_end + counts['val']
        test_end = val_end + counts['test']

        class_train = shuffled_piles[:train_end].tolist()
        class_val = shuffled_piles[train_end:val_end].tolist()
        class_test = shuffled_piles[val_end:test_end].tolist()

        train_piles.extend(class_train)
        val_piles.extend(class_val)
        test_piles.extend(class_test)

        print(f"  Class {class_label}: {len(class_train)} train, {len(class_val)} val, "
              f"{len(class_test)} test (from {num_available} available)")

    # Print total statistics
    print(f"  Total: {len(train_piles)} train, {len(val_piles)} val, {len(test_piles)} test piles")

    return train_piles, val_piles, test_piles


def split_domain_data(
    df,
    split_mode: str = 'percentage',
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    per_class_counts: Dict[int, Dict[str, int]] = None,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[List, List, List]:
    """
    Flexible data splitting function that supports both percentage and count-based modes

    Args:
        df: DataFrame with 'pile' and 'BMA_label' columns
        split_mode: 'percentage' or 'per_class_count'
        train_ratio: Train ratio (for percentage mode)
        val_ratio: Validation ratio (for percentage mode)
        test_ratio: Test ratio (for percentage mode)
        per_class_counts: Per-class split counts (for per_class_count mode)
                         Format: {class_label: {'train': N, 'val': N, 'test': N}}
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting (percentage mode only)

    Returns:
        train_piles, val_piles, test_piles: Lists of pile IDs

    Raises:
        ValueError: If split_mode is invalid or required parameters are missing
    """
    if split_mode == 'percentage':
        return split_data_percentage(
            df, train_ratio, val_ratio, test_ratio, random_state, stratify
        )

    elif split_mode == 'per_class_count':
        if per_class_counts is None:
            raise ValueError(
                "per_class_counts must be provided when split_mode='per_class_count'"
            )
        return split_data_per_class_count(df, per_class_counts, random_state)

    else:
        raise ValueError(
            f"Invalid split_mode: '{split_mode}'. Must be 'percentage' or 'per_class_count'"
        )


def validate_split(
    df,
    train_piles: List,
    val_piles: List,
    test_piles: List,
    verbose: bool = True
) -> bool:
    """
    Validate that data splits are correct (no overlap, all piles used, etc.)

    Args:
        df: DataFrame with 'pile' and 'BMA_label' columns
        train_piles: List of training pile IDs
        val_piles: List of validation pile IDs
        test_piles: List of test pile IDs
        verbose: Whether to print validation results

    Returns:
        True if validation passes, False otherwise
    """
    all_piles = set(df['pile'].unique())
    train_set = set(train_piles)
    val_set = set(val_piles)
    test_set = set(test_piles)

    # Check for overlaps
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    has_overlap = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    # Check coverage
    split_piles = train_set | val_set | test_set
    missing_piles = all_piles - split_piles
    extra_piles = split_piles - all_piles

    has_coverage_issues = bool(missing_piles or extra_piles)

    if verbose:
        print("\n" + "="*70)
        print("DATA SPLIT VALIDATION")
        print("="*70)

        # Overlap check
        if has_overlap:
            print("✗ OVERLAP DETECTED:")
            if train_val_overlap:
                print(f"  Train/Val overlap: {len(train_val_overlap)} piles")
            if train_test_overlap:
                print(f"  Train/Test overlap: {len(train_test_overlap)} piles")
            if val_test_overlap:
                print(f"  Val/Test overlap: {len(val_test_overlap)} piles")
        else:
            print("✓ No overlaps between splits")

        # Coverage check
        if has_coverage_issues:
            print("✗ COVERAGE ISSUES:")
            if missing_piles:
                print(f"  Missing piles: {len(missing_piles)}")
            if extra_piles:
                print(f"  Extra piles: {len(extra_piles)}")
        else:
            print("✓ All piles accounted for")

        # Class distribution
        pile_labels = df.groupby('pile')['BMA_label'].first()

        print("\nClass Distribution:")
        print(f"  {'Split':<10} {'Class 1':<10} {'Class 2':<10} {'Class 3':<10} {'Total':<10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for split_name, split_piles in [('Train', train_piles), ('Val', val_piles), ('Test', test_piles)]:
            split_labels = pile_labels[pile_labels.index.isin(split_piles)]
            class_counts = split_labels.value_counts().sort_index()

            counts_str = []
            for cls in [1, 2, 3]:
                count = class_counts.get(cls, 0)
                counts_str.append(f"{count:<10}")

            total = len(split_piles)
            print(f"  {split_name:<10} {counts_str[0]} {counts_str[1]} {counts_str[2]} {total:<10}")

        print("="*70 + "\n")

    return not (has_overlap or has_coverage_issues)
