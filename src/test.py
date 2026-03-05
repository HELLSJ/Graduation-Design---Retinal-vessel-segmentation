import os
import torch
import numpy as np
from tqdm import tqdm

from src.config import Config
from src.models import get_model
from src.losses import get_loss
from src.metrics import MetricsCalculator
from src.utils import (
    set_seed, save_prediction, plot_training_history,
    plot_roc_curve, plot_pr_curve, create_results_table
)


def evaluate_model(model, test_loader, criterion, device, save_dir):
    model.eval()
    metrics_calc = MetricsCalculator()
    
    os.makedirs(save_dir, exist_ok=True)
    
    sample_count = 0
    max_samples = 20
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch_idx, (images, masks, datasets) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            if isinstance(criterion, type(get_loss('combined'))):
                loss, dice_loss, focal_loss, boundary_loss = criterion(outputs, masks)
            else:
                loss = criterion(outputs, masks)
            
            metrics_calc.update(outputs, masks)
            
            if sample_count < max_samples:
                for i in range(images.shape[0]):
                    if sample_count >= max_samples:
                        break
                    
                    save_path = os.path.join(save_dir, f'prediction_{sample_count}_{datasets[i]}.png')
                    save_prediction(images[i], masks[i], outputs[i], save_path, datasets[i])
                    sample_count += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    metrics = metrics_calc.compute()
    
    return metrics, metrics_calc


def test_model(test_loader, checkpoint_path=None):
    set_seed(Config.RANDOM_SEED)
    device = Config.DEVICE
    
    print(f'Using device: {device}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    model = get_model('improved_attention_unet', in_channels=3, out_channels=Config.NUM_CLASSES)
    model = model.to(device)
    
    criterion = get_loss(
        'combined',
        dice_weight=Config.LOSS_WEIGHTS['dice'],
        focal_weight=Config.LOSS_WEIGHTS['focal'],
        boundary_weight=Config.LOSS_WEIGHTS['boundary'],
        smooth=Config.DICE_SMOOTH,
        alpha=Config.FOCAL_ALPHA,
        gamma=Config.FOCAL_GAMMA
    )
    
    if checkpoint_path:
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Checkpoint loaded successfully')
    else:
        best_checkpoint = os.path.join(Config.CHECKPOINT_DIR, 'checkpoint_epoch_*_best.pth')
        import glob
        checkpoints = glob.glob(best_checkpoint)
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            print(f'Loading best checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Best checkpoint loaded successfully')
        else:
            print('No checkpoint found. Please train the model first.')
            return
    
    save_dir = Config.RESULT_DIR
    metrics, metrics_calc = evaluate_model(model, test_loader, criterion, device, save_dir)
    
    print('\n' + '='*70)
    print('Test Results Table')
    print('='*70)
    print(f"{'Metric':<25} {'Value':<15} {'Standard Deviation':<15}")
    print('-'*70)
    
    # 提取并打印主要指标
    main_metrics = {
        'Dice Coefficient': 'dice_mean',
        'IoU Score': 'iou_mean',
        'Sensitivity': 'sensitivity_mean',
        'Specificity': 'specificity_mean',
        'Accuracy': 'accuracy_mean',
        'Precision': 'precision_mean',
        'Recall': 'recall_mean',
        'F1 Score': 'f1_mean',
        'AUC': 'auc_mean'
    }
    
    for display_name, metric_key in main_metrics.items():
        if metric_key in metrics:
            mean_value = metrics[metric_key]
            std_key = metric_key.replace('mean', 'std')
            std_value = metrics.get(std_key, 0.0)
            print(f"{display_name:<25} {mean_value:.4f}         {std_value:.4f}")
    
    print('-'*70)
    print('='*70)
    
    fpr, tpr, auc_score = metrics_calc.get_roc_curve()
    precision, recall, pr_auc = metrics_calc.get_pr_curve()
    
    plot_roc_curve(fpr, tpr, auc_score, os.path.join(save_dir, 'roc_curve.png'))
    plot_pr_curve(precision, recall, pr_auc, os.path.join(save_dir, 'pr_curve.png'))
    create_results_table(metrics, os.path.join(save_dir, 'results_table.png'))
    
    if checkpoint_path and 'train_losses' in checkpoint:
        plot_training_history(
            checkpoint['train_losses'],
            checkpoint['val_losses'],
            checkpoint['train_metrics'],
            checkpoint['val_metrics'],
            os.path.join(save_dir, 'training_history.png')
        )
    
    # 保存结果为CSV文件
    import csv
    csv_path = os.path.join(save_dir, 'test_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Standard Deviation'])
        for display_name, metric_key in main_metrics.items():
            if metric_key in metrics:
                mean_value = metrics[metric_key]
                std_key = metric_key.replace('mean', 'std')
                std_value = metrics.get(std_key, 0.0)
                writer.writerow([display_name, f'{mean_value:.4f}', f'{std_value:.4f}'])
    
    print(f'\nResults saved to {save_dir}')
    print(f'CSV results saved to {csv_path}')
    
    return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the retinal vessel segmentation model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    args = parser.parse_args()
    
    test_model(args.checkpoint)