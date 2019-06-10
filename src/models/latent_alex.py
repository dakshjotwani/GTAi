import torch
from torch import nn

class LatentAlex(nn.Module):
    def __init__(self):
        super(LatentAlex, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Not used
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 8192),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(8192, 8192),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(8192, 4096),
            nn.ELU(),
            nn.Linear(4096, 2048),
            nn.ELU(),
            nn.Linear(2048, 512),
            nn.Tanh(),
            nn.Linear(512, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 1, -1)
        return x