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
    
    # Set device
    torch.cuda.set_device(Config.GPU_IDS[0])
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.model)
    
    # Create model
    model = get_model(args.model)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader)
    
    if args.test:
        # Test mode
        trainer.test()
    else:
        # Training mode
        trainer.train(resume=args.resume)

if __name__ == '__main__':
    main() 