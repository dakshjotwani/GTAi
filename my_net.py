import torch
from torch import nn
from torch.optim import Adam
import torchvision
from torchvision import models
from torchvision import transforms
from torchvision import datasets

from gta5Loader import gta5Loader

class MyNet(nn.Module):
    def __init__(self):
        