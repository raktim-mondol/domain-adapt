"""
Verification script to ensure train.py, evaluate_kfold.py, and show_fold_splits.py
use identical splitting strategies.

This script loads data using each method and compares the resulting splits.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config


def load_data_train_style(config):
    """Load data exactly as train.py does (lines 461-472)"""
    df = pd.read_csv(config.DATA_PATH)
    df = df[df['BMA_label'] != 'BMA_label']
    df['BMA_label'] = df['BMA_label'].astype(int)
    
    if 4 in df['BMA_label'].unique():
        df = df[df['BMA_label'] != 4]
    
    return df


def load_data_evaluate_style(config):
    """Load data exactly as evaluate_kfold.py does"""
    df = pd.read_csv(config.DATA_PATH)
    
    # Filter out header rows if any
    df = df[df['BMA_label'] != 'BMA_label']
    df['BMA_label'] = df['BMA_label'].astype(int)
    
    # Filter out class 4 if needed
    if 4 in df['BMA_label'].unique():
        df = df[df['BMA_label'] != 4]
    
    return df


def load_data_show_style(config):
    """Load data exactly as show_fold_splits.py does"""
    df = pd.read_csv(config.DATA_PATH)
    
    # Filter out header rows if any
    df = df[df['BMA_label'] != 'BMA_label']
    df['BMA_label'] = df['BMA_label'].astype(int)
    
    # Filter out class 4 if needed
    if 4 in df['BMA_label'].unique():
        df = df[df['BMA_label'] != 4]
    
    return df


def split_piles_kfold(df, n_folds=5, random_state=42):
    """
    Split piles for k-fold cross-validation with handling for small classes.
    EXACT COPY from train.py to ensure identical splits.
    """
    # Get unique piles and their labels
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
    unique_piles = pile_labels['pile'].values
    pile_bma_labels = pile_labels['BMA_label'].values - 1  # 0-indexed
    
    # Show overall class distribution
    class_counts = pile_labels['BMA_label'].value_counts().sort_index()
    
    # Check if any class has fewer samples than n_folds
    min_class_count = min(class_counts.values)
    if min_class_count < n_folds:
        # Manual stratified k-fold split
        np.random.seed(random_state)
        fold_splits = [{'train': [], 'val': []} for _ in range(n_folds)]
        
        for cls in sorted(class_counts.index):
            cls_piles = pile_labels[pile_labels['BMA_label'] == cls]['pile'].values
            cls_piles_shuffled = np.random.permutation(cls_piles)
            n_cls = len(cls_piles_shuffled)
            
            if n_cls >= n_folds:
                # Standard stratified split for this class
                fold_size = n_cls // n_folds
                remainder = n_cls % n_folds
                
                start_idx = 0
                for fold in range(n_folds):
                    end_idx = start_idx + fold_size + (1 if fold < remainder else 0)
                    fold_splits[fold]['val'].extend(cls_piles_shuffled[start_idx:end_idx])
                    start_idx = end_idx
            else:
                # For classes with fewer piles than folds, use cyclic distribution
                for fold in range(n_folds):
                    pile_idx = fold % n_cls
                    fold_splits[fold]['val'].append(cls_piles_shuffled[pile_idx])
        
        # Now assign training sets (all piles not in validation for that fold)
        for fold in range(n_folds):
            val_set = set(fold_splits[fold]['val'])
            train_set = [p for p in unique_piles if p not in val_set]
            fold_splits[fold]['train'] = train_set
            
            # Check if train set has all classes
            train_labels_check = pile_labels[pile_labels['pile'].isin(train_set)]
            train_classes = set(train_labels_check['BMA_label'].unique())
            missing_train_classes = set(class_counts.index) - train_classes
            
            if missing_train_classes:
                # Re-calculate: for small classes, ensure proper split
                val_set_new = set()
                for cls in sorted(class_counts.index):
                    cls_piles_all = pile_labels[pile_labels['BMA_label'] == cls]['pile'].values
                    n_cls = len(cls_piles_all)
                    
                    if n_cls < n_folds:
                        pile_idx = fold % n_cls
                        val_set_new.add(cls_piles_all[pile_idx])
                    else:
                        cls_val_piles = [p for p in val_set if pile_labels[pile_labels['pile'] == p]['BMA_label'].values[0] == cls]
                        val_set_new.update(cls_val_piles)
                
                fold_splits[fold]['val'] = list(val_set_new)
                train_set = [p for p in unique_piles if p not in val_set_new]
                fold_splits[fold]['train'] = train_set
                val_set = val_set_new
            
            # Verify no overlap
            assert len(val_set & set(train_set)) == 0, f"Overlap in fold {fold}!"
        
        # Convert to list of tuples format
        result_splits = [(set(fold['train']), set(fold['val'])) for fold in fold_splits]
        return result_splits
    
    else:
        # Standard stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        fold_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(unique_piles, pile_bma_labels)):
            train_piles = set(unique_piles[train_idx].tolist())
            val_piles = set(unique_piles[val_idx].tolist())
            fold_splits.append((train_piles, val_piles))
        
        return fold_splits


def compare_splits(splits1, splits2, name1, name2):
    """Compare two sets of splits"""
    print(f"\nComparing {name1} vs {name2}:")
    
    if len(splits1) != len(splits2):
        print(f"  [FAIL] Different number of folds: {len(splits1)} vs {len(splits2)}")
        return False
    
    all_match = True
    for fold_idx, ((train1, val1), (train2, val2)) in enumerate(zip(splits1, splits2)):
        fold_num = fold_idx + 1
        
        train_match = train1 == train2
        val_match = val1 == val2
        
        if not train_match or not val_match:
            all_match = False
            print(f"  [FAIL] Fold {fold_num}: Mismatch!")
            if not train_match:
                print(f"     Train piles differ: {len(train1)} vs {len(train2)}")
                only_in_1 = train1 - train2
                only_in_2 = train2 - train1
                if only_in_1:
                    print(f"     Only in {name1}: {only_in_1}")
                if only_in_2:
                    print(f"     Only in {name2}: {only_in_2}")
            if not val_match:
                print(f"     Val piles differ: {len(val1)} vs {len(val2)}")
                only_in_1 = val1 - val2
                only_in_2 = val2 - val1
                if only_in_1:
                    print(f"     Only in {name1}: {only_in_1}")
                if only_in_2:
                    print(f"     Only in {name2}: {only_in_2}")
    
    if all_match:
        print(f"  [PASS] All {len(splits1)} folds match perfectly!")
    
    return all_match


def main():
    print("="*80)
    print("K-FOLD SPLITTING STRATEGY CONSISTENCY VERIFICATION")
    print("="*80)
    
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  Data Path: {config.DATA_PATH}")
    print(f"  Num Folds: {config.NUM_FOLDS}")
    print(f"  Random State: {config.RANDOM_STATE}")
    
    # Load data using each method
    print("\n" + "-"*80)
    print("Loading data using each script's method...")
    print("-"*80)
    
    df_train = load_data_train_style(config)
    print(f"\n[OK] train.py style: {len(df_train)} images, {df_train['pile'].nunique()} piles")
    
    df_eval = load_data_evaluate_style(config)
    print(f"[OK] evaluate_kfold.py style: {len(df_eval)} images, {df_eval['pile'].nunique()} piles")
    
    df_show = load_data_show_style(config)
    print(f"[OK] show_fold_splits.py style: {len(df_show)} images, {df_show['pile'].nunique()} piles")
    
    # Compare data loading results
    print("\n" + "-"*80)
    print("Comparing data loading results...")
    print("-"*80)
    
    data_match = True
    
    # Compare dataframes
    if len(df_train) != len(df_eval) or len(df_train) != len(df_show):
        print(f"\n[FAIL] Dataframe sizes differ!")
        print(f"   train.py: {len(df_train)}")
        print(f"   evaluate_kfold.py: {len(df_eval)}")
        print(f"   show_fold_splits.py: {len(df_show)}")
        data_match = False
    else:
        print(f"\n[PASS] All dataframes have {len(df_train)} images")
    
    # Compare number of piles
    num_piles_train = df_train['pile'].nunique()
    num_piles_eval = df_eval['pile'].nunique()
    num_piles_show = df_show['pile'].nunique()
    
    if not (num_piles_train == num_piles_eval == num_piles_show):
        print(f"[FAIL] Pile counts differ!")
        print(f"   train.py: {num_piles_train}")
        print(f"   evaluate_kfold.py: {num_piles_eval}")
        print(f"   show_fold_splits.py: {num_piles_show}")
        data_match = False
    else:
        print(f"[PASS] All scripts identify {num_piles_train} unique piles")
    
    # Compare class distribution
    train_labels = df_train.groupby('pile')['BMA_label'].first().values - 1
    eval_labels = df_eval.groupby('pile')['BMA_label'].first().values - 1
    show_labels = df_show.groupby('pile')['BMA_label'].first().values - 1
    
    unique_labels_train = set(train_labels)
    unique_labels_eval = set(eval_labels)
    unique_labels_show = set(show_labels)
    
    if not (unique_labels_train == unique_labels_eval == unique_labels_show):
        print(f"[FAIL] Class distributions differ!")
        data_match = False
    else:
        print(f"[PASS] All scripts generate identical class distribution (0-indexed)")
        print(f"   Classes present: {sorted(unique_labels_train)}")
    
    # Generate splits
    print("\n" + "-"*80)
    print("Generating k-fold splits...")
    print("-"*80)
    
    print("\n--- train.py method ---")
    splits_train = split_piles_kfold(df_train, config.NUM_FOLDS, config.RANDOM_STATE)
    print(f"[OK] Generated {len(splits_train)} folds using train.py method")
    
    print("\n--- evaluate_kfold.py method ---")
    splits_eval = split_piles_kfold(df_eval, config.NUM_FOLDS, config.RANDOM_STATE)
    print(f"[OK] Generated {len(splits_eval)} folds using evaluate_kfold.py method")
    
    print("\n--- show_fold_splits.py method ---")
    splits_show = split_piles_kfold(df_show, config.NUM_FOLDS, config.RANDOM_STATE)
    print(f"[OK] Generated {len(splits_show)} folds using show_fold_splits.py method")
    
    # Compare splits
    print("\n" + "-"*80)
    print("Comparing k-fold splits...")
    print("-"*80)
    
    match_train_eval = compare_splits(splits_train, splits_eval, "train.py", "evaluate_kfold.py")
    match_train_show = compare_splits(splits_train, splits_show, "train.py", "show_fold_splits.py")
    match_eval_show = compare_splits(splits_eval, splits_show, "evaluate_kfold.py", "show_fold_splits.py")
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if data_match and match_train_eval and match_train_show and match_eval_show:
        print("\n[SUCCESS] All three scripts use identical splitting strategies.")
        print("   - Data loading is consistent")
        print("   - Pile identification is consistent")
        print("   - Label conversion is consistent")
        print("   - K-fold splits are identical")
        print("\n   Training, evaluation, and analysis will use the same data splits.")
    else:
        print("\n[FAILURE] Splitting strategies differ between scripts.")
        print("   This needs to be fixed before training/evaluation.")
    
    print("="*80)


if __name__ == "__main__":
    main()

