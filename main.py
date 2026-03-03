import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.utils import set_seed
from src.train import train_model
from src.test import test_model
from src.utils import analyze_model_attention


def main():
    parser = argparse.ArgumentParser(description='Retinal Vessel Segmentation with Attention U-Net')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'grad_cam'],
                        help='Mode: train, test, or grad_cam')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (for testing or grad_cam)')
    parser.add_argument('--data_root', type=str, default=Config.DATA_ROOT,
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                        help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    Config.DATA_ROOT = args.data_root
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr
    Config.RANDOM_SEED = args.seed
    
    print('='*60)
    print('Retinal Vessel Segmentation with Attention U-Net')
    print('='*60)
    print(f'Mode: {args.mode}')
    print(f'Device: {Config.DEVICE}')
    print(f'Data Root: {Config.DATA_ROOT}')
    print(f'Batch Size: {Config.BATCH_SIZE}')
    print(f'Epochs: {Config.NUM_EPOCHS}')
    print(f'Learning Rate: {Config.LEARNING_RATE}')
    print('='*60)
    
    if args.mode == 'train':
        print('\nStarting training...')
        model, trainer = train_model(Config)
        print('\nTraining completed!')
        
    elif args.mode == 'test':
        print('\nStarting testing...')
        metrics = test_model(args.checkpoint)
        print('\nTesting completed!')
        
    elif args.mode == 'grad_cam':
        print('\nGenerating Grad-CAM visualizations...')
        from src.data import get_dataloaders
        from src.models import get_model
        
        _, _, test_loader = get_dataloaders(
            Config.DATA_ROOT,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
        
        model = get_model('improved_attention_unet', in_channels=3, out_channels=Config.NUM_CLASSES)
        model = model.to(Config.DEVICE)
        
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Checkpoint loaded successfully')
        else:
            best_checkpoint = os.path.join(Config.CHECKPOINT_DIR, 'checkpoint_epoch_*_best.pth')
            import glob
            checkpoints = glob.glob(best_checkpoint)
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                print('Best checkpoint loaded successfully')
            else:
                print('No checkpoint found. Please train the model first.')
                return
        
        analyze_model_attention(model, test_loader, Config.DEVICE, num_samples=10)
        print('\nGrad-CAM visualization completed!')
    
    print('\n' + '='*60)
    print('All tasks completed!')
    print('='*60)


if __name__ == '__main__':
    main()