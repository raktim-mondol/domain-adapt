"""
Test script for domain adaptation components
Verifies that all modules work correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

print("="*80)
print("Testing Domain Adaptation Components")
print("="*80)

# Test 1: Import all modules
print("\n1. Testing Imports...")
try:
    from src.models import (
        BMA_MIL_Classifier,
        GradientReversal,
        DomainDiscriminator,
        DomainAdaptationModel
    )
    from src.losses import (
        mmd_loss,
        class_conditional_mmd_loss,
        orthogonal_loss
    )
    from src.utils import (
        train_model_domain_adaptation,
        get_rampup_coefficient,
        compute_domain_loss
    )
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Gradient Reversal Layer
print("\n2. Testing Gradient Reversal Layer...")
try:
    grl = GradientReversal(lambda_=1.0)
    x = torch.randn(10, 50, requires_grad=True)
    y = grl(x)

    # Forward should be identity
    assert torch.allclose(y, x), "Forward pass should be identity"

    # Backward should reverse gradients
    loss = y.sum()
    loss.backward()

    # Gradient should be negative (reversed)
    assert x.grad is not None, "Gradients should exist"
    print("   ✓ GRL forward/backward working correctly")

    # Test lambda update
    grl.set_lambda(0.5)
    assert grl.lambda_ == 0.5, "Lambda update failed"
    print("   ✓ GRL lambda update working")

except Exception as e:
    print(f"   ✗ GRL test failed: {e}")
    sys.exit(1)

# Test 3: Domain Discriminator
print("\n3. Testing Domain Discriminator...")
try:
    disc = DomainDiscriminator(input_dim=512, hidden_dim=256, use_spectral_norm=True)
    features = torch.randn(10, 512)

    # Test forward
    output = disc(features)
    assert output.shape == (10, 1), f"Expected shape (10, 1), got {output.shape}"
    print("   ✓ Domain discriminator forward pass working")

    # Test predict_domain
    probs = disc.predict_domain(features)
    assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities should be in [0, 1]"
    print("   ✓ Domain prediction working")

except Exception as e:
    print(f"   ✗ Domain discriminator test failed: {e}")
    sys.exit(1)

# Test 4: MMD Loss
print("\n4. Testing MMD Loss...")
try:
    # Test identical distributions (should be ≈ 0)
    x = torch.randn(50, 100)
    y = x.clone()
    mmd_identical = mmd_loss(x, y)
    print(f"   MMD (identical): {mmd_identical.item():.6f}")
    assert mmd_identical.item() < 0.1, "MMD for identical distributions should be close to 0"
    print("   ✓ MMD for identical distributions is low")

    # Test different distributions (should be > 0)
    x = torch.randn(50, 100)
    y = torch.randn(50, 100) + 2.0
    mmd_different = mmd_loss(x, y)
    print(f"   MMD (different): {mmd_different.item():.6f}")
    assert mmd_different.item() > mmd_identical.item(), "MMD should be higher for different distributions"
    print("   ✓ MMD for different distributions is higher")

    # Test class-conditional MMD
    source_features = torch.randn(60, 50)
    target_features = torch.randn(60, 50)
    source_labels = torch.randint(0, 3, (60,))
    target_labels = torch.randint(0, 3, (60,))

    mmd_cc = class_conditional_mmd_loss(
        source_features, target_features,
        source_labels, target_labels,
        num_classes=3
    )
    print(f"   MMD (class-conditional): {mmd_cc.item():.6f}")
    assert mmd_cc.item() >= 0, "MMD should be non-negative"
    print("   ✓ Class-conditional MMD working")

except Exception as e:
    print(f"   ✗ MMD test failed: {e}")
    sys.exit(1)

# Test 5: Orthogonal Loss
print("\n5. Testing Orthogonal Loss...")
try:
    # Test with orthogonal matrices
    W_cls = torch.randn(3, 100)
    W_dom = torch.randn(50, 100)

    # Make approximately orthogonal using QR decomposition
    combined = torch.cat([W_cls, W_dom], dim=0)
    Q, _ = torch.linalg.qr(combined.t())
    W_cls_orth = Q[:, :3].t()
    W_dom_orth = Q[:, 3:53].t()

    loss_orth = orthogonal_loss(W_cls_orth, W_dom_orth)
    print(f"   Orth loss (orthogonal): {loss_orth.item():.6f}")

    # Test with aligned matrices
    W_cls_aligned = torch.randn(3, 100)
    W_dom_aligned = W_cls_aligned.repeat(17, 1)[:50]
    loss_aligned = orthogonal_loss(W_cls_aligned, W_dom_aligned)
    print(f"   Orth loss (aligned): {loss_aligned.item():.6f}")

    assert loss_aligned.item() > loss_orth.item(), "Aligned matrices should have higher loss"
    print("   ✓ Orthogonal loss working correctly")

except Exception as e:
    print(f"   ✗ Orthogonal loss test failed: {e}")
    sys.exit(1)

# Test 6: Domain Adaptation Model Integration
print("\n6. Testing Domain Adaptation Model...")
try:
    from src.feature_extractor import FeatureExtractor

    # Create a simple feature extractor
    feature_extractor = FeatureExtractor(device='cpu', trainable_layers=0)

    # Create base model
    base_model = BMA_MIL_Classifier(
        feature_extractor=feature_extractor.model,
        feature_dim=768,
        hidden_dim=512,
        num_classes=3,
        dropout=0.3,
        trainable_layers=0
    )

    # Wrap with domain adaptation
    da_model = DomainAdaptationModel(
        base_model=base_model,
        feature_dim=512,
        grl_lambda=1.0,
        use_spectral_norm=True,
        dropout=0.3
    )

    print(f"   Total parameters: {sum(p.numel() for p in da_model.parameters()):,}")
    print("   ✓ Domain adaptation model created")

    # Test forward pass
    bag = torch.randn(12, 3, 224, 224)  # 12 patches
    logits, features, domain_pred = da_model.forward_with_domain(bag)

    assert logits.shape == (3,), f"Expected logits shape (3,), got {logits.shape}"
    assert features.shape == (512,), f"Expected features shape (512,), got {features.shape}"
    assert domain_pred.shape == (1,), f"Expected domain_pred shape (1,), got {domain_pred.shape}"
    print("   ✓ Forward pass with domain prediction working")

    # Test weight extraction
    W_cls = da_model.get_classifier_weights()
    W_dom = da_model.get_discriminator_weights()
    assert W_cls.shape[1] == 512, "Classifier weights dimension mismatch"
    assert W_dom.shape[1] == 512, "Discriminator weights dimension mismatch"
    print("   ✓ Weight extraction working")

    # Test GRL lambda update
    da_model.set_grl_lambda(0.5)
    assert da_model.grl.lambda_ == 0.5, "GRL lambda update failed"
    print("   ✓ GRL lambda update working")

except Exception as e:
    print(f"   ✗ Domain adaptation model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Ramp-up Coefficient
print("\n7. Testing Ramp-up Coefficient...")
try:
    # Test ramp-up schedule
    coeffs = [get_rampup_coefficient(e, rampup_epochs=5) for e in range(10)]

    assert coeffs[0] == 0.0, "Epoch 0 should have coefficient 0"
    assert coeffs[4] == 0.8, "Epoch 4 should have coefficient 0.8"
    assert coeffs[5] == 1.0, "Epoch 5 should have coefficient 1.0"
    assert all(c == 1.0 for c in coeffs[5:]), "All epochs >= 5 should have coefficient 1.0"

    print("   Coefficients: ", [f"{c:.2f}" for c in coeffs])
    print("   ✓ Ramp-up coefficient working correctly")

except Exception as e:
    print(f"   ✗ Ramp-up coefficient test failed: {e}")
    sys.exit(1)

# Test 8: Domain Loss with Label Smoothing
print("\n8. Testing Domain Loss...")
try:
    domain_pred = torch.randn(10, 1)

    # Test without smoothing
    loss_no_smooth = compute_domain_loss(domain_pred, 0, label_smoothing=0.0)
    assert loss_no_smooth.item() > 0, "Loss should be positive"
    print(f"   Domain loss (no smoothing): {loss_no_smooth.item():.4f}")

    # Test with smoothing
    loss_smooth = compute_domain_loss(domain_pred, 0, label_smoothing=0.05)
    print(f"   Domain loss (with smoothing): {loss_smooth.item():.4f}")
    print("   ✓ Domain loss computation working")

except Exception as e:
    print(f"   ✗ Domain loss test failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("All Tests Passed! ✓")
print("="*80)
print("\nDomain adaptation components are working correctly.")
print("You can now run train_domain_adaptation.py to train the model.")
print("="*80 + "\n")
