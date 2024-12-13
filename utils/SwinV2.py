import torch.nn as nn
import torchvision.models as models
from .STN import STN

class VanillaSwinV2(nn.Module):
    
    def __init__(self, n_classes, img_size, size):
        
        super().__init__()

        if size == 'Tiny':
            self.model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            in_features = self.model.head.in_features
        elif size == 'Small':
            self.model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
            in_features = self.model.head.in_features
        elif size == 'Base':
            self.model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            in_features = self.model.head.in_features

        self.model.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

        self.stn = STN(img_size=img_size)

    
    def forward(self, x):
            
        x = self.stn(x)
        x = self.model(x)
        
        return x