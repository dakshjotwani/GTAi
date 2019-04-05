import torch
import time
from torch import nn
import numpy as np

import torchvision
import torchvision.models as models
from gta5Loader import gta5Loader
from torch.optim import Adam

from conv_nets import MobileNetV2
from first_train import Alexnet

def train(device, model, epochs, lr, filename='model'):
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

    loss_fn = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr = learning_rate)

    while current_epoch <= max_epoch and not stopping_criteria:
        print('new epoch')
        current_loss = []
        
        for i, n in enumerate(train_loader):
            optimizer.zero_grad()

            # forward pass
            prediction = model(n[0].to(device))
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
            optimizer.zero_grad()
            
            # print
            if (i+1)%(mini_batches_per_print) == 0:
                print(f'epoch {current_epoch:d} ({100*i/len(train_loader):.0f}%) | batch {i:d} ({i*batch_size:d})' +
                f' | loss {loss.item():.2f} | time {time.time()-start_time:.1f}s', end='')
                loss_list.append(np.average(current_loss))
                current_loss = []
                print(' |', get_control_str(prediction[0]), '|', get_control_str(target[0]))

            if (i+1)%(mini_batches_per_val) == 0:
                val_loss = val(device, val_loader, model, loss_fn)
                val_losses.append(val_loss)
                print('val: ', val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), './models/' + str(filename) + '.pt')
                    print('SAVED!')
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
    model = Alexnet()
    name = 'alexnet'
    model.load_state_dict(torch.load('./models/' + name + '.pt'))
    model.to(device)
    train(device, model, filename=name, epochs=10, lr=0.0001)

if __name__ == "__main__":
    main()
