import torch
from torch import nn

class LatentLSTM(nn.Module):
    def __init__(self):
        super(LatentLSTM, self).__init__()
        self.lstm = nn.LSTM(256 * 6 * 6, 512, 1)
        self.linear = nn.Linear(512, 3)
        self.tanh = nn.Tanh()
        
    def forward(self, embeddings, hidden=None):
        # embeddings should be (seq_len, 1, 256 * 6 * 6)
        if hidden is not None:
            out, hidden = self.lstm(embeddings, hidden)
        else:
            out, hidden = self.lstm(embeddings)
        out = out.reshape(-1, 512)
        out = self.linear(out)
        out = self.tanh(out)
        return out, hidden