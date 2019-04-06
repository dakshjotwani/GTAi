import torch
from torch import nn

class LatentLSTM(nn.Module):
    def __init__(self):
        super(LatentLSTM, self).__init__()
        self.lstm = nn.LSTM(256 * 6 * 6, 512, 2, batch_first=True)
        self.linear = nn.Linear(512, 3)
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden=None):
        # x should be (batch_size, seq_len, 256 * 6 * 6)
        batch_size, seq_len = x.size(0), x.size(1)
        
        if hidden is not None:
            out, hidden = self.lstm(x, hidden)
        else:
            out, hidden = self.lstm(x)
        
        out = out.contiguous().view(-1, 512)
        out = self.linear(out)
        out = self.tanh(out)
        out = out.view(batch_size, seq_len, 3)

        return out, hidden