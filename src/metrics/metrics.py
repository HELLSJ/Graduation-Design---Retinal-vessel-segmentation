import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import torch.nn.functional as F


def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def sensitivity(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    true_pos = (pred * target).sum()
    false_neg = (target * (1 - pred)).sum()
    
    if (true_pos + false_neg) == 0:
        return 0.0
    
    sensitivity = true_pos / (true_pos + false_neg)
    
    return sensitivity.item()


def specificity(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    true_neg = ((1 - pred) * (1 - target)).sum()
    false_pos = (pred * (1 - target)).sum()
    
    if (true_neg + false_pos) == 0:
        return 0.0
    
    specificity = true_neg / (true_neg + false_pos)
    
    return specificity.item()


def accuracy(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    correct = (pred == target).sum()
    total = target.numel()
    
    accuracy = correct / total
    
    return accuracy.item()


def precision(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    true_pos = (pred * target).sum()
    false_pos = (pred * (1 - target)).sum()
    
    if (true_pos + false_pos) == 0:
        return 0.0
    
    precision = true_pos / (true_pos + false_pos)
    
    return precision.item()


def recall(pred, target):
    return sensitivity(pred, target)


def f1_score(pred, target):
    prec = precision(pred, target)
    rec = recall(pred, target)
    
    if (prec + rec) == 0:
        return 0.0
    
    f1 = 2 * prec * rec / (prec + rec)
    
    return f1


def calculate_auc(pred, target):
    pred = torch.sigmoid(pred)
    pred_flat = pred.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()
    
    try:
        auc_score = roc_auc_score(target_flat, pred_flat)
    except:
        auc_score = 0.0
    
    return auc_score


def calculate_metrics(pred, target):
    metrics = {
        'dice': dice_coefficient(pred, target),
        'iou': iou_score(pred, target),
        'sensitivity': sensitivity(pred, target),
        'specificity': specificity(pred, target),
        'accuracy': accuracy(pred, target),
        'precision': precision(pred, target),
        'recall': recall(pred, target),
        'f1': f1_score(pred, target),
        'auc': calculate_auc(pred, target)
    }
    
    return metrics


class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []
        self.accuracy_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.auc_scores = []
        self.all_preds = []
        self.all_targets = []
    
    def update(self, pred, target):
        self.dice_scores.append(dice_coefficient(pred, target))
        self.iou_scores.append(iou_score(pred, target))
        self.sensitivity_scores.append(sensitivity(pred, target))
        self.specificity_scores.append(specificity(pred, target))
        self.accuracy_scores.append(accuracy(pred, target))
        self.precision_scores.append(precision(pred, target))
        self.recall_scores.append(recall(pred, target))
        self.f1_scores.append(f1_score(pred, target))
        self.auc_scores.append(calculate_auc(pred, target))
        
        pred_sigmoid = torch.sigmoid(pred)
        self.all_preds.append(pred_sigmoid.cpu().numpy())
        self.all_targets.append(target.cpu().numpy())
    
    def compute(self):
        metrics = {
            'dice_mean': np.mean(self.dice_scores),
            'dice_std': np.std(self.dice_scores),
            'iou_mean': np.mean(self.iou_scores),
            'iou_std': np.std(self.iou_scores),
            'sensitivity_mean': np.mean(self.sensitivity_scores),
            'sensitivity_std': np.std(self.sensitivity_scores),
            'specificity_mean': np.mean(self.specificity_scores),
            'specificity_std': np.std(self.specificity_scores),
            'accuracy_mean': np.mean(self.accuracy_scores),
            'accuracy_std': np.std(self.accuracy_scores),
            'precision_mean': np.mean(self.precision_scores),
            'precision_std': np.std(self.precision_scores),
            'recall_mean': np.mean(self.recall_scores),
            'recall_std': np.std(self.recall_scores),
            'f1_mean': np.mean(self.f1_scores),
            'f1_std': np.std(self.f1_scores),
            'auc_mean': np.mean(self.auc_scores),
            'auc_std': np.std(self.auc_scores)
        }
        
        return metrics
    
    def get_roc_curve(self):
        all_preds = np.concatenate([p.flatten() for p in self.all_preds])
        all_targets = np.concatenate([t.flatten() for t in self.all_targets])
        
        fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, auc_score
    
    def get_pr_curve(self):
        all_preds = np.concatenate([p.flatten() for p in self.all_preds])
        all_targets = np.concatenate([t.flatten() for t in self.all_targets])
        
        precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
        pr_auc = auc(recall, precision)
        
        return precision, recall, pr_auc