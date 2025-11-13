"""
Maximum Mean Discrepancy (MMD) Loss for Domain Adaptation
Implements multi-kernel RBF MMD with optional class-conditional variant
"""

import torch
import torch.nn as nn


def gaussian_kernel(x, y, sigma):
    """
    Compute Gaussian (RBF) kernel between samples

    Args:
        x: Source samples [n_samples, feature_dim]
        y: Target samples [m_samples, feature_dim]
        sigma: Kernel bandwidth

    Returns:
        kernel_matrix: [n_samples, m_samples]
    """
    # Compute pairwise squared Euclidean distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y^T
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # [n, 1]
    y_norm = (y ** 2).sum(dim=1, keepdim=True)  # [m, 1]

    distances = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())  # [n, m]

    # Apply Gaussian kernel
    kernel_matrix = torch.exp(-distances / (2 * sigma ** 2))

    return kernel_matrix


def mmd_loss(source_features, target_features, bandwidths=[0.5, 1.0, 2.0, 4.0]):
    """
    Compute multi-kernel MMD loss between source and target features

    MMD^2(X, Y) = E[k(x, x')] + E[k(y, y')] - 2*E[k(x, y)]

    Args:
        source_features: Source domain features [n_source, feature_dim]
        target_features: Target domain features [n_target, feature_dim]
        bandwidths: List of kernel bandwidths (sigma values)

    Returns:
        mmd: MMD loss (scalar)
    """
    # Handle single sample case by adding batch dimension
    if source_features.dim() == 1:
        source_features = source_features.unsqueeze(0)
    if target_features.dim() == 1:
        target_features = target_features.unsqueeze(0)

    n_source = source_features.size(0)
    n_target = target_features.size(0)

    if n_source == 0 or n_target == 0:
        return torch.tensor(0.0, device=source_features.device)

    total_mmd = 0.0

    # Compute MMD for each kernel bandwidth and sum
    for sigma in bandwidths:
        # k(x, x') - source vs source
        k_xx = gaussian_kernel(source_features, source_features, sigma)
        # Remove diagonal elements (i != j)
        k_xx_sum = (k_xx.sum() - k_xx.diagonal().sum()) / (n_source * (n_source - 1)) if n_source > 1 else k_xx.sum() / (n_source * n_source)

        # k(y, y') - target vs target
        k_yy = gaussian_kernel(target_features, target_features, sigma)
        k_yy_sum = (k_yy.sum() - k_yy.diagonal().sum()) / (n_target * (n_target - 1)) if n_target > 1 else k_yy.sum() / (n_target * n_target)

        # k(x, y) - source vs target
        k_xy = gaussian_kernel(source_features, target_features, sigma)
        k_xy_sum = k_xy.sum() / (n_source * n_target)

        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        mmd_for_sigma = k_xx_sum + k_yy_sum - 2 * k_xy_sum

        total_mmd += mmd_for_sigma

    return total_mmd


def class_conditional_mmd_loss(source_features, target_features,
                                source_labels, target_labels,
                                num_classes=3, bandwidths=[0.5, 1.0, 2.0, 4.0]):
    """
    Compute class-conditional MMD loss

    Computes MMD separately for each class and sums them:
    MMD_total = sum_c MMD(X_c, Y_c) where c is the class label

    Args:
        source_features: Source domain features [n_source, feature_dim]
        target_features: Target domain features [n_target, feature_dim]
        source_labels: Source domain labels [n_source]
        target_labels: Target domain labels [n_target]
        num_classes: Number of classes
        bandwidths: List of kernel bandwidths

    Returns:
        total_mmd: Class-conditional MMD loss (scalar)
    """
    # Handle single sample case
    if source_features.dim() == 1:
        source_features = source_features.unsqueeze(0)
    if target_features.dim() == 1:
        target_features = target_features.unsqueeze(0)
    if source_labels.dim() == 0:
        source_labels = source_labels.unsqueeze(0)
    if target_labels.dim() == 0:
        target_labels = target_labels.unsqueeze(0)

    total_mmd = 0.0
    num_valid_classes = 0

    # Compute MMD for each class separately
    for c in range(num_classes):
        # Get indices for this class
        source_mask = (source_labels == c)
        target_mask = (target_labels == c)

        source_class_features = source_features[source_mask]
        target_class_features = target_features[target_mask]

        # Skip if either domain has no samples for this class
        if source_class_features.size(0) == 0 or target_class_features.size(0) == 0:
            continue

        # Compute MMD for this class
        class_mmd = mmd_loss(source_class_features, target_class_features, bandwidths)
        total_mmd += class_mmd
        num_valid_classes += 1

    # Average over classes that have samples in both domains
    if num_valid_classes > 0:
        total_mmd = total_mmd / num_valid_classes

    return total_mmd


class MMDLoss(nn.Module):
    """
    MMD Loss module wrapper
    """

    def __init__(self, bandwidths=[0.5, 1.0, 2.0, 4.0], class_conditional=False, num_classes=3):
        """
        Args:
            bandwidths: List of kernel bandwidths
            class_conditional: Whether to use class-conditional MMD
            num_classes: Number of classes (for class-conditional variant)
        """
        super().__init__()
        self.bandwidths = bandwidths
        self.class_conditional = class_conditional
        self.num_classes = num_classes

    def forward(self, source_features, target_features,
                source_labels=None, target_labels=None):
        """
        Compute MMD loss

        Args:
            source_features: Source features [n_source, feature_dim]
            target_features: Target features [n_target, feature_dim]
            source_labels: Source labels (required if class_conditional=True)
            target_labels: Target labels (required if class_conditional=True)

        Returns:
            mmd: MMD loss (scalar)
        """
        if self.class_conditional:
            if source_labels is None or target_labels is None:
                raise ValueError("Labels required for class-conditional MMD")

            return class_conditional_mmd_loss(
                source_features, target_features,
                source_labels, target_labels,
                self.num_classes, self.bandwidths
            )
        else:
            return mmd_loss(source_features, target_features, self.bandwidths)


def test_mmd():
    """
    Test MMD implementation with self-test:
    - Same distribution should give MMD ≈ 0
    - Different distributions should give MMD > 0
    """
    print("\n" + "="*60)
    print("Testing MMD Implementation")
    print("="*60)

    # Test 1: Identical distributions (should be ≈ 0)
    print("\nTest 1: Identical distributions")
    x = torch.randn(100, 50)
    y = x.clone()
    mmd_identical = mmd_loss(x, y)
    print(f"  MMD (identical): {mmd_identical.item():.6f} (should be ≈ 0)")

    # Test 2: Different distributions (should be > 0)
    print("\nTest 2: Different distributions")
    x = torch.randn(100, 50)
    y = torch.randn(100, 50) + 2.0  # Shifted distribution
    mmd_different = mmd_loss(x, y)
    print(f"  MMD (different): {mmd_different.item():.6f} (should be > 0)")

    # Test 3: Class-conditional MMD
    print("\nTest 3: Class-conditional MMD")
    source_features = torch.randn(60, 50)
    target_features = torch.randn(60, 50)
    source_labels = torch.randint(0, 3, (60,))
    target_labels = torch.randint(0, 3, (60,))

    mmd_cc = class_conditional_mmd_loss(
        source_features, target_features,
        source_labels, target_labels,
        num_classes=3
    )
    print(f"  Class-conditional MMD: {mmd_cc.item():.6f}")

    # Test 4: Single sample handling
    print("\nTest 4: Single sample handling")
    x_single = torch.randn(50)
    y_single = torch.randn(50)
    mmd_single = mmd_loss(x_single, y_single)
    print(f"  MMD (single samples): {mmd_single.item():.6f}")

    print("\n" + "="*60)
    print("MMD Tests Complete")
    print("="*60 + "\n")

    return {
        'identical': mmd_identical.item(),
        'different': mmd_different.item(),
        'class_conditional': mmd_cc.item(),
        'single_sample': mmd_single.item()
    }


if __name__ == '__main__':
    test_mmd()
