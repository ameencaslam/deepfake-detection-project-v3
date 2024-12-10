import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet as EffNet
from .base_model import BaseModel
from config.config import Config
class EfficientNet(BaseModel):
    def __init__(self):
        self.model_name = 'efficientnet'
        super().__init__()
        self.config = Config.MODEL_CONFIGS[self.model_name]
        
        # Load pretrained model
        self.backbone = EffNet.from_pretrained(
            f'efficientnet-{self.config["version"]}'
        )
        
        # Get the number of features from the backbone
        in_features = self.backbone._fc.in_features
        
        # Replace classifier with custom head
        self.backbone._fc = nn.Sequential(
            nn.BatchNorm1d(in_features),  # Added normalization
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),  # Added normalization
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Added normalization
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )
        
        # Initialize the new layers
        self._init_weights()
    
    def _init_weights(self):
        for m in self.backbone._fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_optimizer(self):
        # Different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        # Separate backbone and classifier parameters
        for name, param in self.named_parameters():
            if '_fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': self.config.get('backbone_lr', 1e-5)},
            {'params': classifier_params, 'lr': self.config.get('classifier_lr', 1e-4)}
        ], weight_decay=1e-5)
        
        return optimizer 