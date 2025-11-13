"""
Orthogonal Regularization Loss for Domain Adaptation
Encourages task and domain feature representations to be orthogonal
"""

import torch
import torch.nn as nn


def orthogonal_loss(weight_classifier, weight_discriminator):
    """
    Compute orthogonal regularization between classifier and discriminator weights

    Encourages the classifier and discriminator to learn orthogonal features,
    preventing domain-specific features from interfering with task-specific features.

    Loss = ||W_cls * W_dom^T||_F^2 / (||W_cls||_F * ||W_dom||_F)

    where ||.||_F is the Frobenius norm

    Args:
        weight_classifier: Classifier weight matrix [num_classes or hidden_dim, feature_dim]
        weight_discriminator: Discriminator weight matrix [hidden_dim, feature_dim]

    Returns:
        loss: Orthogonal regularization loss (scalar)
    """
    # Compute the product W_cls * W_dom^T
    product = torch.mm(weight_classifier, weight_discriminator.t())

    # Compute Frobenius norm of the product
    product_norm = torch.norm(product, p='fro')

    # Normalize by the norms of the weight matrices to make loss scale-invariant
    cls_norm = torch.norm(weight_classifier, p='fro')
    dom_norm = torch.norm(weight_discriminator, p='fro')

    # Avoid division by zero
    normalizer = cls_norm * dom_norm + 1e-8

    # Normalized orthogonal loss
    loss = (product_norm ** 2) / normalizer

    return loss


def prototype_alignment_loss(source_features, target_features,
                             source_labels, target_labels,
                             num_classes=3, temperature=1.0):
    """
    Prototype alignment loss - prevents domain shift along class-separating axes

    Computes class prototypes (centroids) for each domain and penalizes
    misalignment between source and target prototypes along class-separating directions.

    For each class c:
        μ_s^c = mean(source_features[source_labels == c])
        μ_t^c = mean(target_features[target_labels == c])

    For each pair of classes (i, j):
        Penalize: cos_sim(μ_t^c - μ_s^c, μ_s^i - μ_s^j)

    This encourages target prototypes to shift orthogonally to class boundaries.

    Args:
        source_features: Source domain features [n_source, feature_dim]
        target_features: Target domain features [n_target, feature_dim]
        source_labels: Source domain labels [n_source]
        target_labels: Target domain labels [n_target]
        num_classes: Number of classes
        temperature: Temperature for softmax weighting (default: 1.0)

    Returns:
        loss: Prototype alignment loss (scalar)
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

    device = source_features.device

    # Compute class prototypes (centroids) for each domain
    source_prototypes = []
    target_prototypes = []
    valid_classes = []

    for c in range(num_classes):
        source_mask = (source_labels == c)
        target_mask = (target_labels == c)

        if source_mask.sum() > 0 and target_mask.sum() > 0:
            source_proto = source_features[source_mask].mean(dim=0)
            target_proto = target_features[target_mask].mean(dim=0)

            source_prototypes.append(source_proto)
            target_prototypes.append(target_proto)
            valid_classes.append(c)

    if len(valid_classes) < 2:
        # Need at least 2 classes to compute prototype alignment
        return torch.tensor(0.0, device=device)

    source_prototypes = torch.stack(source_prototypes)  # [num_valid_classes, feature_dim]
    target_prototypes = torch.stack(target_prototypes)  # [num_valid_classes, feature_dim]

    # Compute prototype shifts (target - source)
    prototype_shifts = target_prototypes - source_prototypes  # [num_valid_classes, feature_dim]

    # Compute class-separating directions (differences between source prototypes)
    total_loss = 0.0
    num_pairs = 0

    num_valid = len(valid_classes)
    for i in range(num_valid):
        for j in range(i + 1, num_valid):
            # Class-separating direction: μ_s^i - μ_s^j
            class_sep_dir = source_prototypes[i] - source_prototypes[j]
            class_sep_dir = class_sep_dir / (torch.norm(class_sep_dir) + 1e-8)  # Normalize

            # For each class, compute alignment of its shift with this separating direction
            for c in range(num_valid):
                shift = prototype_shifts[c]
                shift_norm = shift / (torch.norm(shift) + 1e-8)  # Normalize

                # Cosine similarity
                cos_sim = torch.dot(shift_norm, class_sep_dir)

                # Penalize alignment (we want orthogonality)
                # Use squared cosine similarity to penalize both positive and negative alignment
                total_loss += cos_sim ** 2
                num_pairs += 1

    if num_pairs > 0:
        total_loss = total_loss / num_pairs

    return total_loss


class OrthogonalLoss(nn.Module):
    """
    Orthogonal regularization loss module
    """

    def __init__(self, use_prototype_loss=False, num_classes=3, temperature=1.0):
        """
        Args:
            use_prototype_loss: Whether to include prototype alignment loss
            num_classes: Number of classes (for prototype loss)
            temperature: Temperature for prototype loss
        """
        super().__init__()
        self.use_prototype_loss = use_prototype_loss
        self.num_classes = num_classes
        self.temperature = temperature

    def forward(self, weight_classifier, weight_discriminator,
                source_features=None, target_features=None,
                source_labels=None, target_labels=None):
        """
        Compute orthogonal regularization loss

        Args:
            weight_classifier: Classifier weight matrix
            weight_discriminator: Discriminator weight matrix
            source_features: Source features (optional, for prototype loss)
            target_features: Target features (optional, for prototype loss)
            source_labels: Source labels (optional, for prototype loss)
            target_labels: Target labels (optional, for prototype loss)

        Returns:
            loss: Total orthogonal regularization loss
        """
        # Primary orthogonal loss
        loss = orthogonal_loss(weight_classifier, weight_discriminator)

        # Optional prototype alignment loss
        if self.use_prototype_loss and source_features is not None:
            proto_loss = prototype_alignment_loss(
                source_features, target_features,
                source_labels, target_labels,
                self.num_classes, self.temperature
            )
            loss = loss + proto_loss

        return loss


def test_orthogonal_loss():
    """
    Test orthogonal loss implementation
    """
    print("\n" + "="*60)
    print("Testing Orthogonal Loss Implementation")
    print("="*60)

    # Test 1: Orthogonal weights (should give low loss)
    print("\nTest 1: Orthogonal weights")
    feature_dim = 100
    W_cls = torch.randn(3, feature_dim)
    W_dom = torch.randn(50, feature_dim)

    # Make them approximately orthogonal using QR decomposition
    combined = torch.cat([W_cls, W_dom], dim=0)
    Q, _ = torch.linalg.qr(combined.t())
    W_cls_orth = Q[:, :3].t()
    W_dom_orth = Q[:, 3:53].t()

    loss_orth = orthogonal_loss(W_cls_orth, W_dom_orth)
    print(f"  Loss (orthogonal): {loss_orth.item():.6f} (should be low)")

    # Test 2: Aligned weights (should give high loss)
    print("\nTest 2: Aligned weights")
    W_cls_aligned = torch.randn(3, feature_dim)
    W_dom_aligned = W_cls_aligned.repeat(17, 1)[:50]  # Make discriminator weights similar

    loss_aligned = orthogonal_loss(W_cls_aligned, W_dom_aligned)
    print(f"  Loss (aligned): {loss_aligned.item():.6f} (should be high)")

    # Test 3: Prototype alignment loss
    print("\nTest 3: Prototype alignment loss")
    source_features = torch.randn(60, 50)
    target_features = torch.randn(60, 50)
    source_labels = torch.randint(0, 3, (60,))
    target_labels = torch.randint(0, 3, (60,))

    proto_loss = prototype_alignment_loss(
        source_features, target_features,
        source_labels, target_labels,
        num_classes=3
    )
    print(f"  Prototype alignment loss: {proto_loss.item():.6f}")

    # Test 4: Single sample handling
    print("\nTest 4: Edge cases")
    W_small = torch.randn(2, 10)
    loss_small = orthogonal_loss(W_small, W_small)
    print(f"  Loss (same matrix): {loss_small.item():.6f}")

    print("\n" + "="*60)
    print("Orthogonal Loss Tests Complete")
    print("="*60 + "\n")

    return {
        'orthogonal': loss_orth.item(),
        'aligned': loss_aligned.item(),
        'prototype': proto_loss.item(),
        'same_matrix': loss_small.item()
    }


if __name__ == '__main__':
    test_orthogonal_loss()
