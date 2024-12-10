import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Input normalization
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x1, x2):
        B, N, C = x1.shape
        
        # Apply input normalization
        x1 = self.norm_q(x1)
        x2 = self.norm_k(x2)
        
        # QKV projections
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        # Add & Norm
        x1 = self.norm1(x1 + x)
        # FFN
        x1 = self.norm2(x1 + self.mlp(x1))
        
        return x1

class FeatureExtractor(nn.Module):
    def __init__(self, is_freq=False):
        super().__init__()
        # Use ResNet18 as backbone
        resnet = models.resnet18(pretrained=True)
        
        # Modify first layer for frequency branch
        if is_freq:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add normalization after backbone
        self.post_norm = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = self.backbone(x)
        return self.post_norm(x)

class CrossAttentionHybrid(BaseModel):
    def __init__(self):
        super().__init__('cross_attention')
        
        # Feature extractors
        self.spatial_extractor = FeatureExtractor(is_freq=False)
        self.freq_extractor = FeatureExtractor(is_freq=True)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            spatial_features = self.spatial_extractor(dummy_input)
            self.feature_dim = spatial_features.shape[1]
            self.seq_len = spatial_features.shape[2] * spatial_features.shape[3]
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(
                dim=self.feature_dim,
                num_heads=self.config['num_heads']
            ) for _ in range(3)
        ])
        
        # Feature normalization
        self.spatial_norm = nn.LayerNorm(self.feature_dim)
        self.freq_norm = nn.LayerNorm(self.feature_dim)
        
        # Classification head with additional normalization
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Extract spatial features
        spatial_features = self.spatial_extractor(x)
        B, C, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, C, -1).permute(0, 2, 1)
        spatial_features = self.spatial_norm(spatial_features)
        
        # Extract frequency features
        x_freq = torch.fft.fft2(x.mean(dim=1, keepdim=True)).abs()
        x_freq = torch.log(x_freq + 1)
        freq_features = self.freq_extractor(x_freq)
        freq_features = freq_features.view(B, C, -1).permute(0, 2, 1)
        freq_features = self.freq_norm(freq_features)
        
        # Apply cross-attention
        spatial_attended = spatial_features
        freq_attended = freq_features
        
        for layer in self.cross_attention_layers:
            # Cross attention in both directions
            spatial_attended = layer(spatial_attended, freq_attended)
            freq_attended = layer(freq_attended, spatial_attended)
        
        # Global pooling
        spatial_features = spatial_attended.mean(dim=1)
        freq_features = freq_attended.mean(dim=1)
        
        # Concatenate features
        combined_features = torch.cat([spatial_features, freq_features], dim=1)
        
        # Classification
        return self.classifier(combined_features)
    
    def get_optimizer(self):
        # Different learning rates for different components
        feature_params = []
        attention_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'extractor' in name:
                feature_params.append(param)
            elif 'cross_attention' in name:
                attention_params.append(param)
            else:
                classifier_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': feature_params, 'lr': 1e-5},
            {'params': attention_params, 'lr': 5e-5},
            {'params': classifier_params, 'lr': 1e-4}
        ], weight_decay=0.01)
        
        return optimizer 