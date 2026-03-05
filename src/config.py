import torch

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_ROOT = 'data'
    
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 1e-5
    
    NUM_CLASSES = 1
    
    SAVE_FREQ = 5
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    RESULT_DIR = 'results'
    
    RANDOM_SEED = 42
    
    AUGMENTATION = True
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    
    LOSS_WEIGHTS = {
        'dice': 0.5,
        'focal': 0.3,
        'boundary': 0.2
    }
    
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    DICE_SMOOTH = 1e-6
    
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1