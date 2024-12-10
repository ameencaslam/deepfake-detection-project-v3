import torch
import torch.nn as nn
import timm
from .base_model import BaseModel
from config.config import Config

class SwinTransformer(BaseModel):
    def __init__(self):
        self.model_name = 'swin'  # Set model name before super().__init__()
        super().__init__()
        self.config = Config.MODEL_CONFIGS[self.model_name]
        
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224_in22k',
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension using correct image size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, Config.IMAGE_SIZE[self.model_name], Config.IMAGE_SIZE[self.model_name])
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Create custom classification head with more normalization
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),  # Added normalization
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),  # Added normalization
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Added normalization
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )
        
        # Initialize the classifier
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_optimizer(self):
        # Different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        # Separate backbone and classifier parameters
        for name, param in self.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': classifier_params, 'lr': 1e-4}
        ], weight_decay=0.01)
        
        return optimizer 