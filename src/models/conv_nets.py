import torch
from torch import nn
import torchvision
from torchvision import transforms
from models.semantic_seg.models import ModelBuilder, SegmentationModule
from PIL import Image

import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv = nn.Sequential(
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

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ELU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), 1, -1))
        return x

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()

        self.conv = nn.Sequential(
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

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(4096, 4096)
            #nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), 1, -1))
        return x

class MobileNetV2(nn.Module):

    def __init__(self, num_classes=128, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = MobileNetV2.InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = []
        features.append(MobileNetV2.ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(MobileNetV2.ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(2).mean(2)
        x = self.classifier(x)
        return x
    class ConvBNReLU(nn.Sequential):
        def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
            padding = (kernel_size - 1) // 2
            super(MobileNetV2.ConvBNReLU, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True),
            )
    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio):
            super(MobileNetV2.InvertedResidual, self).__init__()
            self.stride = stride
            assert stride in [1, 2]

            hidden_dim = int(round(inp * expand_ratio))
            self.use_res_connect = self.stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # pw
                layers.append(MobileNetV2.ConvBNReLU(inp, hidden_dim, kernel_size=1))
            layers.extend([
                # dw
                MobileNetV2.ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)
    class debug(nn.Module):
        def __init__(self, *args):
            super(MobileNetV2.debug, self).__init__()
            self.args = args

        def forward(self, x):
            print('debug')
            for n in self.args:
                print(n)
            print(x.size())
            return x

class SemanticSegHuge(nn.Module):
    def __init__(self, device):
        super(SemanticSegHuge, self).__init__()

        builder = ModelBuilder()
        net_encoder = builder.build_encoder()
        net_decoder = builder.build_decoder()
        crit = nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        self.segmentation_module.eval()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(21, 64, kernel_size=11, stride=4, padding=2),
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

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(4096, 3),
            nn.Tanh()
        )

    def forward(self, x, orig):
        shape = x.size()
        # x = torch.cat((x, torch.zeros(shape[0], shape[1], 4, shape[3]).to(self.device)), dim=2)
        # feed_dict = {'img_data': x}
        orig[:,:240,:] = 0
        orig[:,:,80:] = 0

        # img = Image.fromarray(orig[0].numpy(), 'RGB')
        # img.show()

        with torch.no_grad():
            x = self.segmentation_module(x, segSize=(300, 400))
        x = x[:,torch.tensor([1,2,3,4,7,10,12,13,14,21,73,81,84,88,103,117,128]),:,:]
        # print(orig.size())
        # print(x.size())
        x = torch.cat((x, orig.permute(0,3,1,2).type(torch.FloatTensor).to(self.device)), dim=1)
        # print(x.size())
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), 1, -1))
        return x

#1-4 wall,building,sky,floor
#7 road
#10 grass
#12 sidewalk
#13 person
#14 ground
#21 car
#73 tree
#81 bus
#84 truck
#88 street;lamp
#103 van
#117 motorbike
#128 bike