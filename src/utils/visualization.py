import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cv2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_image(tensor):
    image = denormalize(tensor)
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def mask_to_image(mask):
    mask = mask.cpu().numpy().squeeze()
    return (mask * 255).astype(np.uint8)


def save_prediction(image, mask, pred, save_path, dataset_name=''):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    img = tensor_to_image(image)
    msk = mask_to_image(mask)
    pred_mask = mask_to_image(torch.sigmoid(pred))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(msk, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    if dataset_name:
        fig.suptitle(f'Dataset: {dataset_name}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    train_dice = [m['dice_mean'] for m in train_metrics]
    val_dice = [m['dice_mean'] for m in val_metrics]
    axes[0, 1].plot(epochs, train_dice, 'b-', label='Train')
    axes[0, 1].plot(epochs, val_dice, 'r-', label='Val')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    train_iou = [m['iou_mean'] for m in train_metrics]
    val_iou = [m['iou_mean'] for m in val_metrics]
    axes[0, 2].plot(epochs, train_iou, 'b-', label='Train')
    axes[0, 2].plot(epochs, val_iou, 'r-', label='Val')
    axes[0, 2].set_title('IoU Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('IoU')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    train_sens = [m['sensitivity_mean'] for m in train_metrics]
    val_sens = [m['sensitivity_mean'] for m in val_metrics]
    axes[1, 0].plot(epochs, train_sens, 'b-', label='Train')
    axes[1, 0].plot(epochs, val_sens, 'r-', label='Val')
    axes[1, 0].set_title('Sensitivity')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sensitivity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    train_spec = [m['specificity_mean'] for m in train_metrics]
    val_spec = [m['specificity_mean'] for m in val_metrics]
    axes[1, 1].plot(epochs, train_spec, 'b-', label='Train')
    axes[1, 1].plot(epochs, val_spec, 'r-', label='Val')
    axes[1, 1].set_title('Specificity')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Specificity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    train_acc = [m['accuracy_mean'] for m in train_metrics]
    val_acc = [m['accuracy_mean'] for m in val_metrics]
    axes[1, 2].plot(epochs, train_acc, 'b-', label='Train')
    axes[1, 2].plot(epochs, val_acc, 'r-', label='Val')
    axes[1, 2].set_title('Accuracy')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curve(precision, recall, pr_auc, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_results_table(metrics, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    data = []
    for key, value in metrics.items():
        if isinstance(value, float):
            data.append([key, f'{value:.4f}'])
        else:
            data.append([key, str(value)])
    
    table = ax.table(cellText=data, colLabels=['Metric', 'Value'], 
                     cellLoc='center', loc='center', colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    for i in range(len(data) + 1):
        table[(i, 0)].set_facecolor('#4CAF50')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        if i == 0:
            table[(i, 1)].set_facecolor('#4CAF50')
            table[(i, 1)].set_text_props(weight='bold', color='white')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()