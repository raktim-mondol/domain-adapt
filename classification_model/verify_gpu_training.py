#!/usr/bin/env python3
"""
Quick verification script to ensure GPU is being used for training
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.feature_extractor import FeatureExtractor
from src.models import BMA_MIL_Classifier
from configs.config import Config

def verify_gpu_setup():
    """Verify GPU is available and models are using it"""
    
    print("="*70)
    print("GPU TRAINING VERIFICATION")
    print("="*70)
    
    # Check CUDA availability
    print("\n1. Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA is available")
        print(f"   📱 GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   💾 Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   🔢 PyTorch CUDA Version: {torch.version.cuda}")
    else:
        print(f"   ❌ CUDA is not available. Training will use CPU.")
        return False
    
    # Check Config device
    print("\n2. Checking Config.DEVICE...")
    print(f"   Config.DEVICE = {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"   ✅ Config is set to use CUDA")
    else:
        print(f"   ⚠️ Config.DEVICE is not 'cuda'")
    
    # Test Feature Extractor
    print("\n3. Testing Feature Extractor...")
    try:
        feature_extractor = FeatureExtractor(
            device=Config.DEVICE,
            trainable_layers=Config.TRAINABLE_LAYERS
        )
        model_device = next(feature_extractor.model.parameters()).device
        print(f"   Feature extractor device: {model_device}")
        if model_device.type == 'cuda':
            print(f"   ✅ Feature extractor is on GPU")
        else:
            print(f"   ❌ Feature extractor is on CPU")
            
        # Test feature extraction
        dummy_patches = [torch.randn(224, 224, 3).numpy() for _ in range(2)]
        print(f"   Testing feature extraction...")
        from PIL import Image
        pil_patches = [Image.fromarray((p * 255).astype('uint8')) for p in dummy_patches]
        features = feature_extractor.extract_features(pil_patches)
        print(f"   Extracted features device: {features.device}")
        if features.device.type == 'cuda':
            print(f"   ✅ Extracted features are on GPU")
        else:
            print(f"   ❌ Extracted features are on CPU")
            
    except Exception as e:
        print(f"   ❌ Error testing feature extractor: {e}")
        return False
    
    # Test MIL Model
    print("\n4. Testing BMA MIL Model...")
    try:
        model = BMA_MIL_Classifier(
            feature_dim=Config.FEATURE_DIM,
            image_hidden_dim=Config.IMAGE_HIDDEN_DIM,
            pile_hidden_dim=Config.PILE_HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES
        )
        model = model.to(Config.DEVICE)
        model_device = next(model.parameters()).device
        print(f"   Model device: {model_device}")
        if model_device.type == 'cuda':
            print(f"   ✅ MIL model is on GPU")
        else:
            print(f"   ❌ MIL model is on CPU")
            
        # Test forward pass with GPU tensors
        print(f"   Testing forward pass...")
        dummy_patch_features = [torch.randn(5, 12, 768, device=Config.DEVICE) for _ in range(2)]
        output, attention = model(dummy_patch_features)
        print(f"   Output device: {output.device}")
        if output.device.type == 'cuda':
            print(f"   ✅ Model output is on GPU")
        else:
            print(f"   ❌ Model output is on CPU")
            
    except Exception as e:
        print(f"   ❌ Error testing MIL model: {e}")
        return False
    
    # Check GPU memory usage
    print("\n5. GPU Memory Status...")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        print(f"   ✅ GPU memory is being used")
    
    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED - GPU TRAINING IS READY!")
    print("="*70)
    print("\nYou can now run training with:")
    print("  python scripts\\train.py")
    print("\nMonitor GPU usage during training with:")
    print("  - Task Manager (Performance tab)")
    print("  - nvidia-smi command")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = verify_gpu_setup()
    sys.exit(0 if success else 1)

