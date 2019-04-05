import time
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import BatchSampler
from test_lstm import my_datasampler as mds
from torch.utils.data import SequentialSampler
from torchvision import transforms

from embed_lstm import EmbedLSTM
from conv_nets import ConvNet
from gta5Loader import gta5Loader

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.conv = ConvNet()
        self.lstm = EmbedLSTM()

    def forward(self, x):
        x = self.conv(x)
        # edge case if batchsampler is set to not drop_last
        # if x.size()[0]%4 != 0:
        #     x = x[:-(a.size()[0]%4)].view(4, 31, -1)
        # else:
        #     x = x.view(4, 32, -1)
        x = x.view(4, 32, -1)
        x, _ = self.lstm(x)
        return x


def train(device):
    batch_size = 4*32
    learning_rate = 0.0001

    dataset = gta5Loader('./datasets/gta5train/')
    sampler = mds.ShuffledBatchSampler(dataset, batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train, sampler=sampler)
    model = ConvLSTM()
    model = model.to(device)

    loss_func = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr = learning_rate)

    # variables to change
    stopping_epoch = 100
    # stopping_val_acc = 1.00
    mini_batches_per_print = len(train_loader)//100
    # mini_batches_per_val = len(train_loader)//10

    start_time = time.time()
    epochs = 0
    # best_val_acc = 0
    stopping_criteria = False
    loss_list = []
    # val_accs = []
    while epochs < stopping_epoch and not stopping_criteria:
        print('new epoch')
        current_loss = []
        
        for i, n in enumerate(train_loader):
            n[0] = n[0].squeeze(0)
            n[1] = n[1].squeeze(0)

            # forward pass
            prediction = model(n[0].to(device))

            # calculate loss and backprop
            loss = loss_func(prediction, n[1].to(device))
            current_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print
            if (i+1)%(mini_batches_per_print) == 0 and i != 0:
                print(f'epoch {epochs:d} ({100*i/len(train_loader):.0f}%) | batch {i:d} ({i*batch_size:d})' +
                f' | loss {loss.item():.2f} | time {time.time()-start_time:.1f}s')
                loss_list.append(np.average(current_loss))
                current_loss = []
                print(prediction[0], n[1][0])
                print(prediction[1], n[1][1])
                print(prediction[2], n[1][2])
            
        torch.save(model.state_dict(), './models/conv_lstm.pt')
        epochs += 1

# a = torch.tensor([[1,2],[3,4],[3,4],[3,4],[3,4],[3,4],[3,4],[3,4]])
# print(a[:-(a.size()[0]%4)])
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    train(device)
    