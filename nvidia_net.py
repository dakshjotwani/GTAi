import torch
import time
from torch import nn
import numpy as np

import torchvision
import torchvision.models as models
from gta5Loader import gta5Loader
from torch.optim import Adam

class NvidiaNet(nn.Module):
    def __init__(self):
        super(NvidiaNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*18*24, out_features=4096),
            nn.ELU(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ELU(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ELU(),
            nn.Linear(in_features=512, out_features=3),
            nn.Tanh()
        )

    def forward(self, input):
        #input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

def train(device):
    # variables to change
    learning_rate = 0.0001
    stopping_epoch = 100
    stopping_val_acc = 1.00
    batch_size = 32

    # Set model
    model = NvidiaNet()
    model.to(device)

    # Set data loader
    dataset = gta5Loader('./gta5train/')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mini_batches_per_print = len(train_loader)//100
    mini_batches_per_val = len(train_loader)//10

    start_time = time.time()
    epochs = 0
    best_val_acc = 0
    stopping_criteria = False
    loss_list = []
    val_accs = []

    loss_fn = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr = learning_rate)

    while epochs < stopping_epoch and not stopping_criteria:
        print('new epoch')
        correct = 0
        correct_current_print = 0
        total = 0
        total_current_print = 0
        current_loss = []
        
        for i, n in enumerate(train_loader):
            optimizer.zero_grad()

            # forward pass
            prediction = model.forward(n[0].to(device))

            # calculate loss and backprop
            target = torch.stack([torch.DoubleTensor(n[1][0]), torch.DoubleTensor(n[1][1]), torch.DoubleTensor(n[1][2])])
            target = target.transpose(0, 1).float()

            # print(prediction)
            # print(target)

            # compute loss, append to loss list for print output
            loss = loss_fn(prediction, target.to(device))
            current_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            
            # print
            if (i+1)%(mini_batches_per_print) == 0 and i != 0:
                print(f'epoch {epochs:d} ({100*i/len(train_loader):.0f}%) | batch {i:d} ({i*batch_size:d})' +
                f' | loss {loss.item():.2f} | time {time.time()-start_time:.1f}s')
                loss_list.append(np.average(current_loss))
                current_loss = []
                print(prediction[0], target[0])
                print(prediction[1], target[1])
                print(prediction[2], target[2])
            
        torch.save(model.state_dict(), './nvidia-checkpoint.pt')
        epochs += 1

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train(device)