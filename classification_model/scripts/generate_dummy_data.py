"""
Generate dummy data for testing domain adaptation
Creates synthetic CSV files and dummy images for QLD1 (source) and QLD2 (target)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image

def create_dummy_images(image_dir, num_images, image_size=(4032, 3024)):
    """
    Create dummy images with random pixel values

    Args:
        image_dir: Directory to save images
        num_images: Number of images to create
        image_size: Size of images (width, height)

    Returns:
        List of image filenames
    """
    os.makedirs(image_dir, exist_ok=True)

    image_filenames = []
    for i in range(num_images):
        # Create random RGB image
        img_array = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')

        # Save with unique filename
        filename = f"dummy_image_{i:04d}.jpg"
        filepath = os.path.join(image_dir, filename)
        img.save(filepath, quality=85)

        image_filenames.append(filename)

    return image_filenames

def create_dummy_csv(csv_path, image_filenames, num_piles, num_classes=3):
    """
    Create dummy CSV file with pile and label information

    Args:
        csv_path: Path to save CSV
        image_filenames: List of image filenames
        num_piles: Number of piles to create
        num_classes: Number of classes

    Returns:
        DataFrame
    """
    # Create pile assignments
    images_per_pile = len(image_filenames) // num_piles

    data = []
    for i, filename in enumerate(image_filenames):
        pile_idx = min(i // images_per_pile, num_piles - 1)
        pile_id = f"pile_{pile_idx:03d}"

        # Assign label to pile (consistent within pile)
        label = (pile_idx % num_classes) + 1  # Labels 1, 2, 3

        data.append({
            'pile': pile_id,
            'image_path': filename,
            'BMA_label': label
        })

    df = pd.DataFrame(data)

    # Save CSV with header row
    with open(csv_path, 'w') as f:
        f.write('pile,image_path,BMA_label\n')
        f.write('pile,image_path,BMA_label\n')  # Duplicate header (as in original data)

    df.to_csv(csv_path, mode='a', index=False, header=False)

    return df

def main():
    print("="*80)
    print("Creating Dummy Data for Domain Adaptation Testing")
    print("="*80)

    # Configuration
    qld1_config = {
        'num_images': 60,  # 60 images
        'num_piles': 12,   # 12 piles (5 images per pile)
        'image_dir': 'classification_model/data/qld1_images',
        'csv_path': 'classification_model/data/qld1_data.csv'
    }

    qld2_config = {
        'num_images': 48,  # 48 images
        'num_piles': 12,   # 12 piles (4 images per pile)
        'image_dir': 'classification_model/data/qld2_images',
        'csv_path': 'classification_model/data/qld2_data.csv'
    }

    # Create directories
    os.makedirs('classification_model/data', exist_ok=True)

    # Generate QLD1 (Source Domain)
    print("\n1. Creating QLD1 (Source Domain) Data...")
    print(f"   Images: {qld1_config['num_images']}")
    print(f"   Piles: {qld1_config['num_piles']}")

    qld1_images = create_dummy_images(
        qld1_config['image_dir'],
        qld1_config['num_images'],
        image_size=(4032, 3024)
    )
    print(f"   ✓ Created {len(qld1_images)} dummy images in {qld1_config['image_dir']}")

    qld1_df = create_dummy_csv(
        qld1_config['csv_path'],
        qld1_images,
        qld1_config['num_piles'],
        num_classes=3
    )
    print(f"   ✓ Created CSV file: {qld1_config['csv_path']}")
    print(f"   Class distribution:")
    class_dist = qld1_df.groupby('pile')['BMA_label'].first().value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"     Class {cls}: {count} piles")

    # Generate QLD2 (Target Domain)
    print("\n2. Creating QLD2 (Target Domain) Data...")
    print(f"   Images: {qld2_config['num_images']}")
    print(f"   Piles: {qld2_config['num_piles']}")

    qld2_images = create_dummy_images(
        qld2_config['image_dir'],
        qld2_config['num_images'],
        image_size=(4032, 3024)
    )
    print(f"   ✓ Created {len(qld2_images)} dummy images in {qld2_config['image_dir']}")

    qld2_df = create_dummy_csv(
        qld2_config['csv_path'],
        qld2_images,
        qld2_config['num_piles'],
        num_classes=3
    )
    print(f"   ✓ Created CSV file: {qld2_config['csv_path']}")
    print(f"   Class distribution:")
    class_dist = qld2_df.groupby('pile')['BMA_label'].first().value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"     Class {cls}: {count} piles")

    # Summary
    print("\n" + "="*80)
    print("Dummy Data Creation Complete!")
    print("="*80)
    print("\nGenerated Files:")
    print(f"  QLD1 CSV:    {qld1_config['csv_path']}")
    print(f"  QLD1 Images: {qld1_config['image_dir']}/ ({qld1_config['num_images']} files)")
    print(f"  QLD2 CSV:    {qld2_config['csv_path']}")
    print(f"  QLD2 Images: {qld2_config['image_dir']}/ ({qld2_config['num_images']} files)")

    print("\nData Summary:")
    print(f"  QLD1: {len(qld1_df)} images, {qld1_df['pile'].nunique()} piles, {len(qld1_df['BMA_label'].unique())} classes")
    print(f"  QLD2: {len(qld2_df)} images, {qld2_df['pile'].nunique()} piles, {len(qld2_df['BMA_label'].unique())} classes")

    print("\nYou can now test domain adaptation with these dummy datasets!")
    print("="*80)

if __name__ == '__main__':
    main()
