import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import resnet50

class GTAResNet(nn.Module):
    def __init__(self):
        super(GTAResNet, self).__init__()
        self.resnet = resnet50(pretrained=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Tanh()
        )

    def conv_forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.conv_forward(x)
        x = self.resnet.fc(x)

        return x