import torch
import torch.nn as nn
from config.config import Config

class BaseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.config = Config.MODEL_CONFIGS[model_name]
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
    
    def get_criterion(self):
        return nn.BCEWithLogitsLoss()
    
    def prepare_model(self):
        """Prepare model for training (multi-GPU if needed)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            # GPU available
            self.to(device)
            if Config.MULTI_GPU and torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                # Preserve model attributes before wrapping
                model_attributes = {key: value for key, value in self.__dict__.items()
                                 if not key.startswith('_')}
                self = nn.DataParallel(self, device_ids=Config.GPU_IDS)
                # Restore attributes to DataParallel object
                for key, value in model_attributes.items():
                    setattr(self.module, key, value)
        else:
            # CPU only
            print("No GPU available. Using CPU. This will be slower!")
            print("Consider reducing batch size if memory issues occur.")
            self.to(device)
        
        return self
    
    def get_device(self):
        """Get the device the model is on"""
        if hasattr(self, 'module'):
            # Handle DataParallel
            return next(self.module.parameters()).device
        return next(self.parameters()).device