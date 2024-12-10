import torch
import torchvision.transforms as T
from config.config import Config

def get_train_transforms(model_name):
    """Get training data transforms"""
    # Ensure correct image size for model
    if model_name not in Config.IMAGE_SIZE:
        raise ValueError(f"Image size not defined for model: {model_name}")
    
    size = Config.IMAGE_SIZE[model_name]
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(model_name):
    """Get validation/test data transforms"""
    # Ensure correct image size for model
    if model_name not in Config.IMAGE_SIZE:
        raise ValueError(f"Image size not defined for model: {model_name}")
    
    size = Config.IMAGE_SIZE[model_name]
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ]) 