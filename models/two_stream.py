import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .base_model import BaseModel

class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet18 for frequency stream
        self.backbone = models.resnet18(pretrained=True)
        # Modify first conv layer to accept frequency input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add batch normalization after backbone
        self.norm = nn.BatchNorm2d(512)
        
    def forward(self, x):
        # Convert to frequency domain
        x_freq = torch.fft.fft2(x.mean(dim=1, keepdim=True)).abs()
        x_freq = torch.log(x_freq + 1)  # Log scale
        features = self.backbone(x_freq)
        return self.norm(features)

class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet18 for spatial stream
        self.backbone = models.resnet18(pretrained=True)
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add batch normalization after backbone
        self.norm = nn.BatchNorm2d(512)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.norm(features)

class TwoStreamNetwork(BaseModel):
    def __init__(self):
        super().__init__('two_stream')
        
        # Create branches
        self.spatial_branch = SpatialBranch()
        self.frequency_branch = FrequencyBranch()
        
        # Feature dimensions (512 from each ResNet18 branch)
        self.feature_dim = 512 * 2
        
        # Fusion and classification layers with additional normalization
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),  # Added normalization
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),  # Added normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Added normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Added normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get features from both branches
        spatial_features = self.spatial_branch(x)
        freq_features = self.frequency_branch(x)
        
        # Flatten features
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        freq_features = freq_features.view(freq_features.size(0), -1)
        
        # Concatenate features
        combined_features = torch.cat([spatial_features, freq_features], dim=1)
        
        # Final classification
        return self.fusion(combined_features)
    
    def get_optimizer(self):
        # Different learning rates for different components
        params = [
            {'params': self.spatial_branch.parameters(), 'lr': 1e-5},
            {'params': self.frequency_branch.parameters(), 'lr': 1e-5},
            {'params': self.fusion.parameters(), 'lr': 1e-4}
        ]
        
        return torch.optim.Adam(params, weight_decay=1e-5) 