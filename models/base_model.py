import torch
import torch.nn as nn
from config.config import Config

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Only set model_name if not already set by child class
        if not hasattr(self, 'model_name'):
            self.model_name = self.__class__.__name__.lower()
    
    def prepare_model(self):
        """Prepare model for training"""
        return self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def get_optimizer(self):
        """Get optimizer for training"""
        return torch.optim.Adam(
            self.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
    
    def get_criterion(self):
        """Get loss criterion"""
        return nn.BCEWithLogitsLoss()