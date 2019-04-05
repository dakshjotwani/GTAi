import torch
from torch import nn
from latent_alex import LatentAlex
from latent_lstm import LatentLSTM

class AlexLSTM(nn.Module):
    def __init__(self):
        super(AlexLSTM, self).__init__()
        self.alex = LatentAlex()
        self.lstm = LatentLSTM()

    def forward(self):
        