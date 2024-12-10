import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    setup_ddp(rank, world_size)
    
    # Update Config paths for Kaggle
    Config.DATA_ROOT = "/kaggle/input/3body-filtered-v2-10k"
    Config.CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    Config.LOG_DIR = "/kaggle/working/logs"
    
    # Update number of epochs if provided
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        print(f"\nUsing device: {device}")
        print(f"World size: {world_size}")
    
    # Get dataloaders with DDP sampler
    train_loader, val_loader, test_loader = get_dataloaders(
        args.model,
        rank=rank,
        world_size=world_size,
        distributed=True
    )
    
    # Create model
    if rank == 0:
        print(f"\nInitializing {args.model} model...")
    model = get_model(args.model)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, rank)
    
    if args.test:
        # Test mode
        if rank == 0:
            print("\nRunning in test mode...")
        trainer.test()
    else:
        # Training mode
        if rank == 0:
            print("\nStarting training...")
            if args.resume:
                print("Resuming from checkpoint...")
        trainer.train(resume=args.resume)
    
    cleanup_ddp()

def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size
        )
    else:
        print("No GPU available. Training will be slower.")
        print("Consider reducing batch size if memory issues occur.")
        if Config.BATCH_SIZE > 16:
            print(f"Automatically reducing batch size from {Config.BATCH_SIZE} to 16 for CPU training")
            Config.BATCH_SIZE = 16
        main_worker(0, 1, args)

if __name__ == '__main__':
    main() 