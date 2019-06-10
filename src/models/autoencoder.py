import numpy as np
import time

import torch
from torch import nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam

from conv_nets import MobileNetV2
from first_train import Alexnet
from gta5Loader import gta5Loader


class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size, stride):
        super(CDAutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=2, padding=0), 
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        # Add noise, but use the original lossless input as the target.
        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        y = self.forward_pass(x_noisy)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = CDAutoEncoder(3, 128, 2)
        self.ae2 = CDAutoEncoder(128, 256, 2)
        self.ae3 = CDAutoEncoder(256, 512, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)

        if self.training:
            return a3

        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
            a2_reconstruct = self.ae3.reconstruct(x)
            a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
            x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
            return x_reconstruct


def train_autoencoder(device, model, epochs, lr, filename='model'):
    learning_rate = lr
    max_epoch = epochs
    batch_size = 32
    val_batch_size = 32

    # Set data loader
    dataset = gta5Loader('./datasets/gta5train/')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset2 = gta5Loader('./datasets/gta5val/')
    val_loader = torch.utils.data.DataLoader(dataset2, batch_size=val_batch_size, shuffle=True)

    mini_batches_per_print = len(train_loader)//100
    mini_batches_per_print = max(mini_batches_per_print, 1)
    mini_batches_per_val = len(train_loader)//3
    mini_batches_per_val = max(mini_batches_per_val, 1)

    start_time = time.time()
    current_epoch = 1
    best_val_loss = float('inf')
    stopping_criteria = False
    loss_list = []
    val_losses = []

    classifier = nn.Linear(512 * 16, 10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    while current_epoch <= max_epoch and not stopping_criteria:
        print('new epoch')
        current_loss = []
        
        for i, n in enumerate(train_loader):
            img, target = n
            target = torch.tensor(target[0]).cuda()
            zeros = torch.zeros(32, 3, 4, 400)
            img = torch.cat((img, zeros), dim=2)
            img = Variable(img).cuda()
            features = model(img).detach()
            prediction = classifier(features.view(features.size(0), -1))
            print(prediction.size())
            print(target.size())
            loss = criterion(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = prediction.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            # print
            if (i+1)%(mini_batches_per_print) == 0:
                print(f'epoch {current_epoch:d} ({100*i/len(train_loader):.0f}%) | batch {i:d} ({i*batch_size:d})' +
                f' | loss {loss.item():.2f} | time {time.time()-start_time:.1f}s', end='')
                loss_list.append(np.average(current_loss))
                current_loss = []

            # if (i+1)%(mini_batches_per_val) == 0:
            #     val_loss = val(device, val_loader, model, criterion)
            #     val_losses.append(val_loss)
            #     print('val: ', val_loss)
            #     if val_loss < best_val_loss:
            #         best_val_loss = val_loss
            #         torch.save(model.state_dict(), './models/' + str(filename) + '.pt')
            #         print('SAVED!')
        current_epoch += 1
        torch.save(model.state_dict(), './models/' + str(filename) + '-epochs.pt')

def val(device, dataloader, model, loss_fn):
    losses = []
    model.eval()
    with torch.no_grad():
        for i, n in enumerate(dataloader):
            # forward pass
            prediction = model(n[0].to(device))
            # calculate loss and backprop
            target = torch.stack([torch.DoubleTensor(n[1][0]), torch.DoubleTensor(n[1][1]), torch.DoubleTensor(n[1][2])])
            target = target.transpose(0, 1).float()
            loss = loss_fn(prediction, target.to(device))
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

def get_control_str(controls):
    result = ''
    if controls[0] < -0.1:
        result += 'left'
    elif controls[0] > 0.1:
        result += 'right'
    else:
        result += 'str8'
    result += '|'
    if controls[1] > -0.5:
        result += 'yes'
    else:
        result += 'no'
    result += '|'
    if controls[2] > -0.5:
        result += 'yes'
    else:
        result += 'no'
    return result

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model = StackedAutoEncoder()
    name = 'AE'
    # model.load_state_dict(torch.load('./models/' + name + '.pt'))
    model.to(device)
    train_autoencoder(device, model, filename=name, epochs=10, lr=0.0001)

if __name__ == "__main__":
    main()
