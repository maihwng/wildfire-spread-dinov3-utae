"""
Loss Functions for Spatio-Temporal Wildfire Spread Prediction.
Includes:
  - FocalLoss: Handles extreme class imbalance (unburned vs burned pixels)
  - DiceLoss: Directly optimizes segmentation overlap (F1 / Dice score)
  - CombinedLoss: Weighted combination of BCE/Focal + Dice loss
  - JaccardLoss: Optimizes Intersection over Union (IoU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for highly imbalanced wildfire segmentation.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type_as(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    """
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.type_as(logits)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class JaccardLoss(nn.Module):
    """
    Soft Jaccard / IoU Loss.
    """
    def __init__(self, smooth: float = 1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.type_as(logits)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        total = probs_flat.sum() + targets_flat.sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou


class CombinedLoss(nn.Module):
    """
    Combined Loss = w_focal * FocalLoss + w_dice * DiceLoss
    Default recommended loss for WildfireSpreadTS.
    """
    def __init__(
        self,
        w_focal: float = 1.0,
        w_dice: float = 1.0,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ):
        super(CombinedLoss, self).__init__()
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_focal * self.focal(logits, targets) + self.w_dice * self.dice(logits, targets)

