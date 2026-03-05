import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from src.config import Config
from src.models import get_model
from src.losses import get_loss, CombinedLoss
from src.metrics import MetricsCalculator


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        metrics_calc = MetricsCalculator()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            if isinstance(self.criterion, CombinedLoss):
                loss, dice_loss, focal_loss, boundary_loss = self.criterion(outputs, masks)
            else:
                loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            metrics_calc.update(outputs, masks)
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.train_loader)
        metrics = metrics_calc.compute()
        
        self.train_losses.append(avg_loss)
        self.train_metrics.append(metrics)
        
        return avg_loss, metrics
    
    def validate(self, epoch):
        self.model.eval()
        epoch_loss = 0.0
        metrics_calc = MetricsCalculator()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, (images, masks, _) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                if isinstance(self.criterion, CombinedLoss):
                    loss, dice_loss, focal_loss, boundary_loss = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
                
                epoch_loss += loss.item()
                metrics_calc.update(outputs, masks)
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.val_loader)
        metrics = metrics_calc.compute()
        
        self.val_losses.append(avg_loss)
        self.val_metrics.append(metrics)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, save_path, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_dice': self.best_dice,
            'best_iou': self.best_iou
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.best_dice = checkpoint.get('best_dice', 0.0)
        self.best_iou = checkpoint.get('best_iou', 0.0)
        
        return checkpoint['epoch']
    
    def train(self, num_epochs, save_dir, save_freq=5):
        os.makedirs(save_dir, exist_ok=True)
        
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            
            epoch_time = time.time() - start_time
            
            # 打印表格形式的结果
            if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
                print('\n' + '='*80)
                print(f'Epoch {epoch}/{num_epochs} - {epoch_time:.2f}s')
                print('='*80)
                print(f"{'Metric':<20} {'Train':<15} {'Validation':<15}")
                print('-'*80)
                
                # 打印主要指标
                print(f"{'Loss':<20} {train_loss:.4f}          {val_loss:.4f}")
                print(f"{'Dice':<20} {train_metrics['dice_mean']:.4f}          {val_metrics['dice_mean']:.4f}")
                print(f"{'IoU':<20} {train_metrics['iou_mean']:.4f}          {val_metrics['iou_mean']:.4f}")
                print(f"{'Sensitivity':<20} {train_metrics['sensitivity_mean']:.4f}          {val_metrics['sensitivity_mean']:.4f}")
                print(f"{'Specificity':<20} {train_metrics['specificity_mean']:.4f}          {val_metrics['specificity_mean']:.4f}")
                print(f"{'Accuracy':<20} {train_metrics['accuracy_mean']:.4f}          {val_metrics['accuracy_mean']:.4f}")
                print(f"{'F1 Score':<20} {train_metrics['f1_mean']:.4f}          {val_metrics['f1_mean']:.4f}")
                print(f"{'AUC':<20} {train_metrics['auc_mean']:.4f}          {val_metrics['auc_mean']:.4f}")
                print('-'*80)
                print('='*80)
            else:
                # 简洁输出
                if self.logger:
                    self.logger.info(f'Epoch {epoch}/{num_epochs} - {epoch_time:.2f}s')
                    self.logger.info(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
                    self.logger.info(f'Train Dice: {train_metrics["dice_mean"]:.4f} | Val Dice: {val_metrics["dice_mean"]:.4f}')
                else:
                    print(f'Epoch {epoch}/{num_epochs} - {epoch_time:.2f}s')
                    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
                    print(f'Train Dice: {train_metrics["dice_mean"]:.4f} | Val Dice: {val_metrics["dice_mean"]:.4f}')
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/train', train_metrics['dice_mean'], epoch)
            writer.add_scalar('Dice/val', val_metrics['dice_mean'], epoch)
            writer.add_scalar('IoU/train', train_metrics['iou_mean'], epoch)
            writer.add_scalar('IoU/val', val_metrics['iou_mean'], epoch)
            
            is_best = False
            if val_metrics['dice_mean'] > self.best_dice:
                self.best_dice = val_metrics['dice_mean']
                is_best = True
            
            if val_metrics['iou_mean'] > self.best_iou:
                self.best_iou = val_metrics['iou_mean']
            
            if epoch % save_freq == 0 or is_best:
                save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(epoch, save_path, is_best)
        
        writer.close()
        
        if self.logger:
            self.logger.info(f'Training completed!')
            self.logger.info(f'Best Dice: {self.best_dice:.4f}')
            self.logger.info(f'Best IoU: {self.best_iou:.4f}')
        else:
            print(f'Training completed!')
            print(f'Best Dice: {self.best_dice:.4f}')
            print(f'Best IoU: {self.best_iou:.4f}')


def train_model(train_loader, val_loader, config=None):
    device = Config.DEVICE
    
    print(f'Using device: {device}')
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
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
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    trainer.train(Config.NUM_EPOCHS, Config.CHECKPOINT_DIR, Config.SAVE_FREQ)
    
    return model, trainer


if __name__ == '__main__':
    model, trainer = train_model(Config)