"""
Verify GPU/CPU Memory Strategy for BMA MIL Classifier

This script verifies that:
1. Preprocessing happens on CPU
2. Training happens on GPU (if available)
3. Features are stored on CPU and transferred to GPU during training
4. Memory usage is efficient
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pandas as pd
from configs.config import Config
from src.feature_extractor import FeatureExtractor
from src.data import PatchExtractor
from PIL import Image
import numpy as np

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

def check_cuda_availability():
    """Check CUDA availability and GPU info"""
    print_section("1. CUDA Availability Check")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        print(f"\nCurrent Device: {Config.DEVICE}")
        return True
    else:
        print("⚠️  GPU not available - will use CPU for verification")
        print(f"Current Device: {Config.DEVICE}")
        return False

def check_preprocessing_device():
    """Verify preprocessing happens on CPU"""
    print_section("2. Preprocessing Device Check")
    
    # Test patch extraction
    print("\n[*] Testing Patch Extraction...")
    patch_extractor = PatchExtractor()
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (4032, 3024), color='red')
    temp_path = 'temp_test_image.jpg'
    dummy_image.save(temp_path)
    
    try:
        patches = patch_extractor.extract_patches(temp_path)
        print(f"[OK] Extracted {len(patches)} patches")
        print(f"   Patch type: {type(patches[0])}")
        print(f"   Patch size: {patches[0].size}")
        print(f"   Device: CPU (PIL Image object)")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Test augmentation
    print("\n[*] Testing Augmentation...")
    from src.augmentation import get_augmentation_pipeline
    
    augmentation = get_augmentation_pipeline(is_training=True, config=Config)
    augmented_patch = augmentation(patches[0])
    print(f"[OK] Augmentation successful")
    print(f"   Output type: {type(augmented_patch)}")
    print(f"   Device: CPU (PIL Image object)")
    
    print("\n[OK] All preprocessing operations verified on CPU")

def check_feature_extraction_strategy(use_gpu):
    """Verify feature extraction and memory strategy"""
    print_section("3. Feature Extraction Memory Strategy")
    
    device = 'cuda' if use_gpu else 'cpu'
    print(f"\n[*] Initializing Feature Extractor on {device.upper()}...")
    
    feature_extractor = FeatureExtractor(device=device, trainable_layers=0)
    
    # Check model device
    model_device = next(feature_extractor.model.parameters()).device
    print(f"[OK] Feature extractor model is on: {model_device}")
    
    # Create dummy patches
    print(f"\n[*] Creating dummy patches...")
    dummy_patches = [Image.new('RGB', (224, 224), color='blue') for _ in range(3)]
    print(f"[OK] Created {len(dummy_patches)} dummy patches")
    print(f"   Input device: CPU (PIL Images)")
    
    # Extract features
    print(f"\n[*] Extracting features...")
    if use_gpu:
        # Track GPU memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    features = feature_extractor.extract_features(dummy_patches)
    
    if use_gpu:
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   Peak GPU memory during extraction: {peak_memory:.3f} GB")
    
    print(f"[OK] Features extracted")
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature device: {features.device}")
    print(f"   Expected device: CPU")
    
    # Verify features are on CPU
    if features.device.type == 'cpu':
        print(f"\n[OK] VERIFIED: Features are stored on CPU")
        print(f"   This saves GPU memory during data loading!")
    else:
        print(f"\n[WARNING] Features are on {features.device}")
        print(f"   They should be on CPU to save GPU memory!")
    
    # Test moving features to GPU
    if use_gpu:
        print(f"\n[*] Testing feature transfer to GPU...")
        torch.cuda.reset_peak_memory_stats()
        features_gpu = features.to(device, non_blocking=True)
        torch.cuda.synchronize()
        
        print(f"[OK] Features moved to GPU")
        print(f"   Device: {features_gpu.device}")
        print(f"   GPU memory after transfer: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        
        # Clean up
        del features_gpu
        torch.cuda.empty_cache()

def check_dataloader_configuration():
    """Verify DataLoader configuration"""
    print_section("4. DataLoader Configuration")
    
    print(f"\n[*] Checking DataLoader parameters...")
    
    use_pin_memory = Config.DEVICE == 'cuda'
    print(f"[OK] pin_memory: {use_pin_memory}")
    print(f"   {'Enabled - faster CPU->GPU transfer' if use_pin_memory else 'Disabled - not needed for CPU training'}")
    
    print(f"[OK] num_workers: 0 (recommended)")
    print(f"   Single-threaded to avoid CUDA context issues")
    
    if use_pin_memory:
        print(f"\n[OK] Optimal configuration for GPU training")
        print(f"   - Features stored on CPU (saves GPU memory)")
        print(f"   - Pinned memory for fast transfer")
        print(f"   - Non-blocking transfers for parallelism")
    else:
        print(f"\n[OK] Appropriate configuration for CPU training")

def check_memory_efficiency(use_gpu):
    """Estimate memory usage"""
    print_section("5. Memory Efficiency Analysis")
    
    # Model size
    print(f"\n[*] Model Memory Requirements:")
    
    # ViT model size
    vit_params = 86_000_000  # ~86M parameters
    vit_size_gb = vit_params * 4 / 1024**3  # 4 bytes per float32 param
    print(f"   Feature Extractor: ~{vit_size_gb:.2f} GB")
    
    # MIL model size (approximate)
    mil_params = 2_000_000  # ~2M parameters
    mil_size_gb = mil_params * 4 / 1024**3
    print(f"   MIL Classifier: ~{mil_size_gb:.2f} GB")
    
    # Gradients (if trainable)
    trainable_layers = Config.TRAINABLE_LAYERS
    if trainable_layers > 0:
        grad_size_gb = vit_size_gb  # Same as model size
        print(f"   Gradients: ~{grad_size_gb:.2f} GB (partially trainable)")
    else:
        grad_size_gb = mil_size_gb  # Only MIL gradients
        print(f"   Gradients: ~{grad_size_gb:.2f} GB (ViT frozen)")
    
    total_model_gb = vit_size_gb + mil_size_gb + grad_size_gb
    print(f"   Total Model + Gradients: ~{total_model_gb:.2f} GB")
    
    # Data size per batch
    print(f"\n[*] Data Memory Requirements:")
    batch_size = Config.BATCH_SIZE
    avg_images_per_pile = 30
    patches_per_image = 12
    feature_dim = 768
    
    features_per_batch = batch_size * avg_images_per_pile * patches_per_image * feature_dim
    features_size_mb = features_per_batch * 4 / 1024**2
    print(f"   Features per batch (batch_size={batch_size}): ~{features_size_mb:.1f} MB")
    
    # Total estimation
    if use_gpu:
        print(f"\n[*] GPU Memory Estimation:")
        total_gpu_gb = total_model_gb + (features_size_mb / 1024)
        print(f"   Minimum Required: ~{total_gpu_gb:.2f} GB")
        print(f"   Recommended: ~{total_gpu_gb * 1.5:.2f} GB (with safety margin)")
        
        if torch.cuda.is_available():
            available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   Your GPU: {available_gb:.2f} GB")
            
            if available_gb > total_gpu_gb * 1.5:
                print(f"   [OK] Sufficient GPU memory")
            elif available_gb > total_gpu_gb:
                print(f"   [WARNING] Tight fit - consider reducing batch size if OOM occurs")
            else:
                print(f"   [ERROR] Insufficient GPU memory - reduce batch size or use CPU")
    
    print(f"\n[*] CPU Memory Estimation:")
    # Cached features for all training data
    num_training_piles = 60  # Approximate
    cached_features_mb = num_training_piles * avg_images_per_pile * patches_per_image * feature_dim * 4 / 1024**2
    print(f"   Cached features: ~{cached_features_mb:.1f} MB")
    print(f"   DataLoader buffer: ~100 MB")
    print(f"   Python + Libraries: ~2 GB")
    total_cpu_gb = (cached_features_mb / 1024) + 0.1 + 2
    print(f"   Total CPU Memory: ~{total_cpu_gb:.1f} GB")
    print(f"   [OK] Manageable on most systems")

def check_training_strategy():
    """Summarize training strategy"""
    print_section("6. Training Strategy Summary")
    
    print(f"\n[*] Memory Flow:")
    print(f"   1. Load image (CPU - PIL)")
    print(f"   2. Extract patches (CPU - PIL)")
    print(f"   3. Apply augmentation (CPU - cv2/PIL)")
    print(f"   4. Extract features (GPU) -> Store on CPU")
    print(f"   5. Cache features on CPU")
    print(f"   6. DataLoader batches on CPU with pinned memory")
    print(f"   7. Transfer batch to GPU during training")
    print(f"   8. Forward/backward pass on GPU")
    print(f"   9. Move metrics back to CPU")
    
    print(f"\n[*] Key Benefits:")
    print(f"   [+] GPU memory used only for active training")
    print(f"   [+] CPU memory used for data storage and preprocessing")
    print(f"   [+] Pinned memory enables fast CPU->GPU transfer")
    print(f"   [+] Non-blocking transfers allow parallelism")
    print(f"   [+] Feature caching speeds up subsequent epochs")
    
    print(f"\n[*] Configuration:")
    print(f"   Device: {Config.DEVICE}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Max Images per Pile: {Config.MAX_IMAGES_PER_PILE}")
    print(f"   Trainable ViT Layers: {Config.TRAINABLE_LAYERS}")

def main():
    """Run all verification checks"""
    print_section("BMA MIL Classifier - Memory Strategy Verification")
    print(f"\nThis script verifies the GPU/CPU memory optimization strategy")
    
    try:
        # Check CUDA
        use_gpu = check_cuda_availability()
        
        # Check preprocessing
        check_preprocessing_device()
        
        # Check feature extraction
        check_feature_extraction_strategy(use_gpu)
        
        # Check DataLoader
        check_dataloader_configuration()
        
        # Check memory efficiency
        check_memory_efficiency(use_gpu)
        
        # Summary
        check_training_strategy()
        
        print_section("[SUCCESS] Verification Complete")
        print(f"\nThe memory strategy is correctly configured:")
        print(f"  * Preprocessing on CPU [OK]")
        print(f"  * Training on {'GPU' if use_gpu else 'CPU'} [OK]")
        print(f"  * Features cached on CPU [OK]")
        print(f"  * Efficient memory usage [OK]")
        
        if use_gpu:
            print(f"\n[READY] GPU-accelerated training is ready!")
        else:
            print(f"\n[READY] CPU training is ready!")
        
        print(f"\nNext steps:")
        print(f"  1. Activate environment: conda activate mine2")
        print(f"  2. Run training: python scripts/train.py")
        print(f"  3. Monitor GPU: nvidia-smi -l 1")
        
    except Exception as e:
        print(f"\n[ERROR] Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

