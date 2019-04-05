import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class EmbedLSTM(nn.Module):
    def __init__(self):
        """Set the hyper-parameters and build the layers."""
        super(EmbedLSTM, self).__init__()
        self.lstm = nn.LSTM(4096, 512, 1)
        self.linear = nn.Linear(512, 3)
        self.tanh = nn.Tanh()
        
    def forward(self, embeddings, hidden=None):
        """Decode image feature vectors and generates captions."""
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        # hiddens, _ = self.lstm(packed)
        # outputs = self.linear(hiddens[0])
        # return outputs
        
        # embeddings should be (seq_len, 1, 4096)
        if hidden is not None:
            out, hidden = self.lstm(embeddings, hidden)
        else:
            out, hidden = self.lstm(embeddings)
        out = out.reshape(-1, 512)
        out = self.linear(out)
        out = self.tanh(out)
        return out, hidden


def example():
    rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
    input = torch.randn(5, 3, 10) # seq_len, batch, input_size
    h0 = torch.randn(2, 3, 20) # num_layers, batch, hidden_size
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output.size()) # seq_len, batch, hidden_size
    print(hn.size()) # num_layers, batch, hidden_size

# if __name__ == "__main__":
#     example()