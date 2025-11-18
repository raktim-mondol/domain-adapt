"""
Utility functions for Semi-Supervised Domain Adaptation
Handles labeled/unlabeled splitting for target domain
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def split_semi_supervised_piles(
    pile_ids: List[str],
    df: pd.DataFrame,
    labeled_ratio: float = 0.2,
    labeled_samples_per_class: int = None,
    random_seed: int = 42,
    num_classes: int = 3
) -> Tuple[List[str], List[str]]:
    """
    Split pile IDs into labeled and unlabeled subsets for semi-supervised learning

    Args:
        pile_ids: List of pile IDs to split
        df: DataFrame with 'pile' and 'BMA_label' columns
        labeled_ratio: Fraction of piles to label (0.0 to 1.0)
        labeled_samples_per_class: Alternative: specific number of piles per class
                                   If set, this overrides labeled_ratio
        random_seed: Random seed for reproducibility
        num_classes: Number of classes

    Returns:
        labeled_pile_ids: List of pile IDs that will have labels
        unlabeled_pile_ids: List of pile IDs that will be unlabeled
    """
    np.random.seed(random_seed)

    # Get pile labels
    pile_labels = {}
    for pile_id in pile_ids:
        pile_data = df[df['pile'] == pile_id]
        if len(pile_data) > 0:
            # Convert to 0-indexed
            pile_labels[pile_id] = pile_data['BMA_label'].iloc[0] - 1

    # Group piles by class
    piles_by_class = {c: [] for c in range(num_classes)}
    for pile_id, label in pile_labels.items():
        if label in piles_by_class:
            piles_by_class[label].append(pile_id)

    labeled_pile_ids = []
    unlabeled_pile_ids = []

    # Select labeled piles from each class
    for class_id in range(num_classes):
        class_piles = piles_by_class[class_id]

        if len(class_piles) == 0:
            continue

        # Shuffle piles for this class
        shuffled_piles = np.random.permutation(class_piles).tolist()

        # Determine how many to label
        if labeled_samples_per_class is not None:
            # Use specific number per class
            n_labeled = min(labeled_samples_per_class, len(shuffled_piles))
        else:
            # Use ratio
            n_labeled = max(1, int(len(shuffled_piles) * labeled_ratio))

        # Split
        labeled_pile_ids.extend(shuffled_piles[:n_labeled])
        unlabeled_pile_ids.extend(shuffled_piles[n_labeled:])

    return labeled_pile_ids, unlabeled_pile_ids


def create_semi_supervised_info(
    labeled_pile_ids: List[str],
    unlabeled_pile_ids: List[str],
    df: pd.DataFrame
) -> Dict:
    """
    Create summary information about semi-supervised split

    Args:
        labeled_pile_ids: List of labeled pile IDs
        unlabeled_pile_ids: List of unlabeled pile IDs
        df: DataFrame with pile information

    Returns:
        info: Dictionary with split statistics
    """
    total_piles = len(labeled_pile_ids) + len(unlabeled_pile_ids)
    labeled_ratio = len(labeled_pile_ids) / total_piles if total_piles > 0 else 0.0

    # Count images
    labeled_images = len(df[df['pile'].isin(labeled_pile_ids)])
    unlabeled_images = len(df[df['pile'].isin(unlabeled_pile_ids)])

    # Count labeled piles per class
    labeled_per_class = {}
    for pile_id in labeled_pile_ids:
        pile_data = df[df['pile'] == pile_id]
        if len(pile_data) > 0:
            label = pile_data['BMA_label'].iloc[0]
            labeled_per_class[label] = labeled_per_class.get(label, 0) + 1

    info = {
        'total_piles': total_piles,
        'labeled_piles': len(labeled_pile_ids),
        'unlabeled_piles': len(unlabeled_pile_ids),
        'labeled_ratio': labeled_ratio,
        'labeled_images': labeled_images,
        'unlabeled_images': unlabeled_images,
        'labeled_per_class': labeled_per_class
    }

    return info


def print_semi_supervised_split_info(info: Dict, domain_name: str = "Target"):
    """
    Print summary of semi-supervised split

    Args:
        info: Dictionary from create_semi_supervised_info
        domain_name: Name of domain (for display)
    """
    print(f"\n{'='*60}")
    print(f"Semi-Supervised Split - {domain_name} Domain")
    print(f"{'='*60}")
    print(f"Total piles: {info['total_piles']}")
    print(f"Labeled piles: {info['labeled_piles']} ({info['labeled_ratio']*100:.1f}%)")
    print(f"Unlabeled piles: {info['unlabeled_piles']} ({(1-info['labeled_ratio'])*100:.1f}%)")
    print(f"\nTotal images:")
    print(f"  Labeled: {info['labeled_images']}")
    print(f"  Unlabeled: {info['unlabeled_images']}")
    print(f"\nLabeled piles per class:")
    for label, count in sorted(info['labeled_per_class'].items()):
        print(f"  Class {label}: {count} piles")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test the utility functions
    print("Testing semi-supervised utilities...")

    # Create dummy data
    test_df = pd.DataFrame({
        'pile': ['pile_1', 'pile_1', 'pile_2', 'pile_2', 'pile_3', 'pile_3',
                 'pile_4', 'pile_4', 'pile_5', 'pile_5', 'pile_6', 'pile_6'],
        'image_path': [f'img_{i}.jpg' for i in range(12)],
        'BMA_label': [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3]
    })

    test_pile_ids = ['pile_1', 'pile_2', 'pile_3', 'pile_4', 'pile_5', 'pile_6']

    # Test ratio-based splitting
    print("\nTest 1: Ratio-based splitting (20%)")
    labeled, unlabeled = split_semi_supervised_piles(
        test_pile_ids, test_df, labeled_ratio=0.2, random_seed=42
    )
    print(f"Labeled: {labeled}")
    print(f"Unlabeled: {unlabeled}")

    info = create_semi_supervised_info(labeled, unlabeled, test_df)
    print_semi_supervised_split_info(info)

    # Test samples-per-class splitting
    print("\nTest 2: Samples-per-class splitting (1 per class)")
    labeled, unlabeled = split_semi_supervised_piles(
        test_pile_ids, test_df, labeled_samples_per_class=1, random_seed=42
    )
    print(f"Labeled: {labeled}")
    print(f"Unlabeled: {unlabeled}")

    info = create_semi_supervised_info(labeled, unlabeled, test_df)
    print_semi_supervised_split_info(info)

    print("\nTests complete!")
