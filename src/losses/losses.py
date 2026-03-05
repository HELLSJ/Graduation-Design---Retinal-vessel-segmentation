import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        if target.dim() == 3 and pred.dim() == 4:
            target = target.unsqueeze(1)
        target = target.float()
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        if target.dim() == 3 and pred.dim() == 4:
            target = target.unsqueeze(1)
        target = target.float()
        
        n, c, h, w = pred.shape
        
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(pred.device)
        kernel[0, 0, 1, 1] = 0
        
        pred_boundary = F.conv2d(pred, kernel, padding=1)
        target_boundary = F.conv2d(target, kernel, padding=1)
        
        pred_boundary = (pred_boundary > 0).float()
        target_boundary = (target_boundary > 0).float()
        
        boundary_diff = torch.abs(pred_boundary - target_boundary)
        
        weight_map = torch.exp(-boundary_diff / self.theta)
        
        loss = (boundary_diff * weight_map).mean()
        
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.3, boundary_weight=0.2, 
                 smooth=1e-6, alpha=0.25, gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss(smooth)
        self.focal_loss = FocalLoss(alpha, gamma)
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (self.dice_weight * dice + 
                      self.focal_weight * focal + 
                      self.boundary_weight * boundary)
        
        return total_loss, dice, focal, boundary


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        if target.dim() == 3 and pred.dim() == 4:
            target = target.unsqueeze(1)
        target = target.float()
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        true_pos = (pred * target).sum()
        false_pos = ((1 - target) * pred).sum()
        false_neg = (target * (1 - pred)).sum()
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        return 1 - tversky


def get_loss(loss_type='combined', **kwargs):
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'boundary':
        return BoundaryLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'weighted_bce':
        return WeightedBCELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")