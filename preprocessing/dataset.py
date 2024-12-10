import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from config.config import Config
from .transforms import get_train_transforms, get_val_transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Path to dataset
            transform (callable, optional): Transform to be applied on images
            split (str): 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Get all image paths
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        
        # Get sorted list of images to ensure deterministic order
        real_images = sorted([os.path.join('real', img) for img in os.listdir(self.real_dir)])
        fake_images = sorted([os.path.join('fake', img) for img in os.listdir(self.fake_dir)])
        
        # Calculate split indices
        n_real = len(real_images)
        n_fake = len(fake_images)
        
        real_train_idx = int(n_real * Config.TRAIN_SPLIT)
        real_val_idx = real_train_idx + int(n_real * Config.VAL_SPLIT)
        
        fake_train_idx = int(n_fake * Config.TRAIN_SPLIT)
        fake_val_idx = fake_train_idx + int(n_fake * Config.VAL_SPLIT)
        
        # Split datasets
        if split == 'train':
            self.real_images = real_images[:real_train_idx]
            self.fake_images = fake_images[:fake_train_idx]
        elif split == 'val':
            self.real_images = real_images[real_train_idx:real_val_idx]
            self.fake_images = fake_images[fake_train_idx:fake_val_idx]
        else:  # test
            self.real_images = real_images[real_val_idx:]
            self.fake_images = fake_images[fake_val_idx:]
        
        # Combine real and fake images (no shuffling needed)
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        
        print(f"{split} set size: {len(self.image_paths)} images "
              f"({len(self.real_images)} real, {len(self.fake_images)} fake)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(model_name, rank=0, world_size=1, distributed=False):
    """Create train, validation, and test dataloaders"""
    # Create transforms
    train_transform = get_train_transforms(model_name)
    val_transform = get_val_transforms(model_name)
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        Config.DATA_ROOT, 
        transform=train_transform,
        split='train'
    )
    
    val_dataset = DeepfakeDataset(
        Config.DATA_ROOT,
        transform=val_transform,
        split='val'
    )
    
    test_dataset = DeepfakeDataset(
        Config.DATA_ROOT,
        transform=val_transform,
        split='test'
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # Important for DDP to have same batch size across GPUs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        sampler=test_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 