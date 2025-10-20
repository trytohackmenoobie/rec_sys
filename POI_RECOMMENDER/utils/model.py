"""
Model utilities and custom loss functions for POI recommendation system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PersonalityAwareLoss(nn.Module):
    """
    Custom loss function that incorporates user personality traits
    into the recommendation loss calculation.
    """
    
    def __init__(
        self,
        personality_weight: float = 0.3,
        base_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.personality_weight = personality_weight
        self.base_weight = base_weight
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
        logger.info(f"Initialized PersonalityAwareLoss with personality_weight={personality_weight}, base_weight={base_weight}")
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        user_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the personality-aware loss.
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            user_features: User feature vectors [batch_size, feature_dim]
        
        Returns:
            Loss value
        """
        # Base cross-entropy loss
        ce_loss = self.cross_entropy(predictions, targets)
        
        # Simple regularization on user features
        feature_reg = torch.norm(user_features, p=2, dim=1)
        
        # Combine losses
        total_loss = self.base_weight * ce_loss + self.personality_weight * feature_reg
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss
