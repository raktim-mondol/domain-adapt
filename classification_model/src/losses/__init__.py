"""
Loss functions for domain adaptation
"""

from .mmd import mmd_loss, class_conditional_mmd_loss
from .orthogonal import orthogonal_loss, prototype_alignment_loss

__all__ = [
    'mmd_loss',
    'class_conditional_mmd_loss',
    'orthogonal_loss',
    'prototype_alignment_loss'
]
