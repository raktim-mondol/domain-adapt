"""
Domain Discriminator for Domain Adaptation with Gradient Reversal Layer (GRL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) - reverses gradients during backpropagation
    Forward pass: identity (y = x)
    Backward pass: reverse and scale gradients (dy/dx = -lambda * grad_output)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Args:
            x: Input tensor
            lambda_: Scaling factor for gradient reversal
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Reverse and scale gradients
        """
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_
        return grad_input, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer module
    """

    def __init__(self, lambda_=1.0):
        """
        Args:
            lambda_: Scaling factor for gradient reversal (default=1.0)
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        """Update the lambda value (useful for scheduling)"""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator with Spectral Normalization
    Architecture: Linear(d -> d/2) -> ReLU -> Dropout -> Linear(d/2 -> 1) -> Sigmoid
    """

    def __init__(self, input_dim, hidden_dim=None, dropout=0.3, use_spectral_norm=True):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (default: input_dim // 2)
            dropout: Dropout rate (default: 0.3)
            use_spectral_norm: Whether to use spectral normalization (default: True)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.use_spectral_norm = use_spectral_norm

        # Build discriminator
        if use_spectral_norm:
            self.fc1 = nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim))
            self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, input_dim] or [input_dim]

        Returns:
            domain_pred: Domain prediction logits [batch_size, 1] or [1]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def predict_domain(self, x):
        """
        Predict domain label with sigmoid activation

        Args:
            x: Input features

        Returns:
            domain_prob: Domain probability [batch_size, 1] or [1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


class DomainAdaptationModel(nn.Module):
    """
    Wrapper that combines base model with domain discriminator and GRL
    """

    def __init__(self, base_model, feature_dim, grl_lambda=1.0,
                 use_spectral_norm=True, dropout=0.3):
        """
        Args:
            base_model: Base MIL classifier model
            feature_dim: Dimension of bag-level features
            grl_lambda: Initial lambda for GRL (default: 1.0)
            use_spectral_norm: Whether to use spectral norm in discriminator
            dropout: Dropout rate for discriminator
        """
        super().__init__()

        self.base_model = base_model
        self.grl = GradientReversal(lambda_=grl_lambda)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=feature_dim,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm
        )

    def forward(self, bag, return_features=False):
        """
        Forward pass through base model

        Args:
            bag: Input bag [num_patches, 3, H, W]
            return_features: Whether to return bag-level features

        Returns:
            If return_features=False: (logits, attention_weights)
            If return_features=True: (logits, attention_weights, bag_features)
        """
        # Get predictions from base model
        logits, attention_weights = self.base_model(bag)

        if return_features:
            # Extract bag-level features from aggregator
            # We need to re-run through feature extractor and aggregator
            num_patches = bag.shape[0]

            if self.base_model.training and not any(p.requires_grad for p in self.base_model.feature_extractor.parameters()):
                with torch.no_grad():
                    patch_features = self.base_model.feature_extractor(bag)
            else:
                patch_features = self.base_model.feature_extractor(bag)

            bag_features, _ = self.base_model.aggregator(patch_features)
            return logits, attention_weights, bag_features

        return logits, attention_weights

    def forward_with_domain(self, bag):
        """
        Forward pass with domain prediction

        Args:
            bag: Input bag [num_patches, 3, H, W]

        Returns:
            logits: Classification logits [num_classes]
            bag_features: Bag-level features [feature_dim]
            domain_pred: Domain prediction logits [1]
        """
        # Get bag-level features
        logits, _, bag_features = self.forward(bag, return_features=True)

        # Apply GRL and domain discriminator
        reversed_features = self.grl(bag_features)
        domain_pred = self.domain_discriminator(reversed_features)

        return logits, bag_features, domain_pred

    def set_grl_lambda(self, lambda_):
        """Update GRL lambda (useful for scheduling)"""
        self.grl.set_lambda(lambda_)

    def get_classifier_weights(self):
        """Get classifier head weights for orthogonal regularization"""
        # Get weights from first layer of classifier
        return self.base_model.classifier[0].weight

    def get_discriminator_weights(self):
        """Get discriminator weights for orthogonal regularization"""
        return self.domain_discriminator.fc1.weight
