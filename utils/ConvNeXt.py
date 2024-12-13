import torch.nn as nn
import torchvision.models as models
from .STN import STN 

class ConvNeXt(nn.Module):

    def __init__(self, n_classes, img_size, size):
        
        super().__init__()
    
        if size == 'Tiny':
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        elif size == 'Small':
            self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        elif size == 'Base':
            self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        elif size == 'Large':
            self.model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)

        self.model.classifier[2] = nn.Sequential(
            nn.Linear(in_features=1536, out_features=256, bias=True),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

        self.stn = STN(img_size=img_size)

    
    def forward(self, x):
        
        x = self.stn(x)
        x = self.model(x)
        
        return x