import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
        
        real_images = [os.path.join('real', img) for img in os.listdir(self.real_dir)]
        fake_images = [os.path.join('fake', img) for img in os.listdir(self.fake_dir)]
        
        # Shuffle images
        np.random.seed(42)  # for reproducibility
        np.random.shuffle(real_images)
        np.random.shuffle(fake_images)
        
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
        
        # Combine and shuffle data
        all_images = list(zip(self.real_images + self.fake_images, 
                             [0] * len(self.real_images) + [1] * len(self.fake_images)))
        np.random.shuffle(all_images)
        self.image_paths, self.labels = zip(*all_images)
        self.image_paths = list(self.image_paths)
        self.labels = list(self.labels)
        
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

def get_dataloaders(model_name):
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 