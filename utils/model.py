import torch.nn as nn
import torchvision.models as models

class ConvNext(nn.Module):
    
    def __init__(self, n_classes):
        super().__init__()
        
        self.conv_model = models.convnext_base(pretrained=True)
        self.conv_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.conv_model.classifier[0].in_features, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):

        x = self.conv_model(x)

        return x