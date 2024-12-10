import argparse
import torch
from config.config import Config
from preprocessing.dataset import get_dataloaders
from models import (
    EfficientNet,
    SwinTransformer,
    TwoStreamNetwork,
    Xception,
    CNNTransformer,
    CrossAttentionHybrid
)
from training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Deepfake Detection Training')
    parser.add_argument('--model', type=str, default='efficientnet',
                        choices=['efficientnet', 'swin', 'two_stream', 
                                'xception', 'cnn_transformer', 'cross_attention'],
                        help='Model architecture to use')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--test', action='store_true',
                        help='Evaluate model on test set')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (default: from config)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size for training (default: from config)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    return parser.parse_args()

def get_model(model_name):
    model_map = {
        'efficientnet': EfficientNet,
        'swin': SwinTransformer,
        'two_stream': TwoStreamNetwork,
        'xception': Xception,
        'cnn_transformer': CNNTransformer,
        'cross_attention': CrossAttentionHybrid
    }
    
    if model_name not in model_map:
        raise ValueError(f'Model {model_name} not implemented. Available models: {list(model_map.keys())}')
    
    model = model_map[model_name]()
    return model

def main():
    # Parse arguments
    args = parse_args()
    
    # Update Config paths for Kaggle
    Config.DATA_ROOT = "/kaggle/input/3body-filtered-v2-10k"
    Config.CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    Config.LOG_DIR = "/kaggle/working/logs"
    
    # Update number of epochs if provided
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
    
    # Update batch size if provided
    if args.batch is not None:
        if isinstance(Config.BATCH_SIZE, dict):
            Config.BATCH_SIZE[args.model] = args.batch
        else:
            Config.BATCH_SIZE = args.batch
    
    # Set device and print hardware info
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        print(f"\nUsing GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("\nNo GPU available. Training will be slower.")
        if Config.BATCH_SIZE > 16:
            print(f"Automatically reducing batch size from {Config.BATCH_SIZE} to 16 for CPU training")
            Config.BATCH_SIZE = 16
    
    print(f"Device: {device}")
    
    # Get dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(args.model)
    
    # Create model
    print(f"\nInitializing {args.model} model...")
    model = get_model(args.model)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, device)
    
    if args.test:
        # Test mode
        print("\nRunning in test mode...")
        trainer.test()
    else:
        # Training mode
        print("\nStarting training...")
        if args.resume:
            print("Resuming from checkpoint...")
        trainer.train(resume=args.resume)

if __name__ == '__main__':
    main() 