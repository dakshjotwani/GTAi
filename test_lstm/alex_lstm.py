import torch
from torch import nn
from latent_alex import LatentAlex
from latent_lstm import LatentLSTM
from GTA_data import GTASequenceDataset, SequenceSampler

class AlexLSTM(nn.Module):
    def __init__(self, train_conv=False):
        super(AlexLSTM, self).__init__()
        self.conv = LatentAlex()
        self.lstm = LatentLSTM()
        self.train_conv = train_conv
        if self.train_conv is False:
            self.conv.eval()

    def forward(self, x, hidden=None):
        num_batches, seq_len = x.size(0), x.size(1)

        # Pass through CNN
        x = torch.flatten(x, start_dim=0, end_dim=1)
        if self.train_conv is False:
            with torch.no_grad():
                x = self.conv(x) #hopefully this will work with 16 * 8
        else:
            x = self.conv(x)
        
        # Pass through LSTM
        x = x.view(num_batches, seq_len, 256 * 6 * 6)
        if hidden is None:
            x, h_state = self.lstm(x)
        else:
            x, h_state = self.lstm(x, hidden)
        
        return x, h_state
    
    def load_conv(self, path_to_alex):
        self.conv.load_state_dict(torch.load(path_to_alex))