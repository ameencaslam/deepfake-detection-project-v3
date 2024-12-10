import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel
from config.config import Config
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        # Reshape: (batch, embed_dim, height, width) -> (batch, seq_len, embed_dim)
        batch_size, embed_dim, height, width = x.shape
        seq_len = height * width
        x = x.view(batch_size, embed_dim, seq_len).permute(2, 0, 1)  # (seq_len, batch, embed_dim)
        
        # Self-attention with first normalization
        attn_output, _ = self.mha(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output  # Skip connection
        
        # MLP block with second normalization
        x = x + self.mlp(self.norm2(x))  # Skip connection
        
        # Return to original shape: (seq_len, batch, embed_dim) -> (batch, embed_dim, height, width)
        x = x.permute(1, 2, 0).view(batch_size, embed_dim, height, width)
        return x


class CNNTransformer(BaseModel):
    def __init__(self):
        self.model_name = 'cnn_transformer'
        super().__init__()
        self.config = Config.MODEL_CONFIGS[self.model_name]
        
        # CNN Backbone (ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add normalization after backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]  # Number of channels
            self.seq_len = features.shape[2] * features.shape[3]  # H*W
            del dummy_input, features
            torch.cuda.empty_cache()
        
        self.post_backbone_norm = nn.BatchNorm2d(self.feature_dim)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            MultiHeadAttention(
                embed_dim=self.feature_dim,
                num_heads=self.config['num_heads']
            ) for _ in range(3)
        ])
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head with additional normalization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN feature extraction with normalization
        features = self.backbone(x)
        features = self.post_backbone_norm(features)
        batch_size = features.shape[0]
        
        # Apply transformer blocks
        x = features
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Global pooling
        x = self.gap(x)
        x = x.view(batch_size, -1)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def get_optimizer(self):
        # Different learning rates for different components
        backbone_params = []
        transformer_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'transformer' in name:
                transformer_params.append(param)
            else:
                classifier_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': transformer_params, 'lr': 5e-5},
            {'params': classifier_params, 'lr': 1e-4}
        ], weight_decay=0.01)
        
        return optimizer 