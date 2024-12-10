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
    args = parse_args()
    
    # Set device and print hardware info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        torch.cuda.set_device(Config.GPU_IDS[0])
    else:
        print("No GPU available. Training will be slower.")
        print("Consider reducing batch size if memory issues occur.")
        if Config.BATCH_SIZE > 16:
            print(f"Automatically reducing batch size from {Config.BATCH_SIZE} to 16 for CPU training")
            Config.BATCH_SIZE = 16
    
    # Get dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(args.model)
    
    # Create model
    print(f"\nInitializing {args.model} model...")
    model = get_model(args.model)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader)
    
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