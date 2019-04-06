import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import resnet50

class GTAResNet(nn.Module):
    def __init__(self):
        super(GTAResNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.resnet.forward(x)