import torch
import torch.nn as nn
import numpy as np
import time
from torch.optim import Adam

from models import GTAResNet, AlexLSTM
from data import GTADataset, GTASequenceDataset, SequenceSampler

def val(device, dataloader, model, loss_fn):
    losses = []
    model.eval()
    with torch.no_grad():
        for i, n in enumerate(dataloader):
            prediction = model(n[0].to(device))
            target = n[1].to(device)
            if isinstance(prediction, tuple):
                prediction = prediction[0]

            loss = loss_fn(prediction, target)
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

def train(model, loss_fn, optimizer, epochs, train_loader, val_loader,
          model_name='model',
          model_save_dir='./',
          device=torch.device('cpu')):
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

    while current_epoch <= epochs and not stopping_criteria:
        print('new epoch')
        current_loss = []
        
        for i, n in enumerate(train_loader):
            optimizer.zero_grad()

            # forward pass
            prediction = model(n[0].to(device))
            target = n[1].to(device)
            if isinstance(prediction, tuple):
                prediction = prediction[0]

            # compute loss, append to loss list for print output
            loss = loss_fn(prediction, target)
            current_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            
            # Occasionally print train loss
            if (i+1)%(mini_batches_per_print) == 0:
                print(f'epoch {current_epoch:d} ({100*i/len(train_loader):.0f}%)' +
                f' | loss {loss.item():.2f} | time {time.time()-start_time:.1f}s')
                loss_list.append(np.average(current_loss))
                current_loss = []

            # Validate once a while
            if (i+1)%(mini_batches_per_val) == 0:
                val_loss = val(device, val_loader, model, loss_fn)
                val_losses.append(val_loss)
                print('val: ', val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_dir + model_name + '.pt')
                    print('SAVED!')

        current_epoch += 1
        torch.save(model.state_dict(), model_save_dir + model_name + '-epochs.pt')

def train_GTAResNet():
    # Train params
    model = GTAResNet()
    loss_fn = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 10
    model_name = 'GTAResNet'
    save_dir = '../models/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize data loaders
    batch_size = 16
    val_batch_size = 16

    train_dataset = GTADataset('../datasets/gta5train/', steer_only=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_dataset = GTADataset('../datasets/gta5val/', steer_only=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=val_batch_size,
                                             shuffle=True)
    
    # Train model
    model.to(device)
    train(model, loss_fn, optimizer, epochs, train_loader, val_loader,
          model_name=model_name,
          model_save_dir=save_dir,
          device=device)

def train_ConvLSTM():
    # Train params
    model = AlexLSTM(train_conv=False)
    model.load_conv('../models/GTAResNet.pt')
    loss_fn = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 10
    model_name = 'ConvLSTM'
    save_dir = '../models/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize data loaders
    batch_size = 12
    seq_len = 16
    val_batch_size = 8
    val_seq_len = 16

    # Set data loader
    train_dataset = GTASequenceDataset('../datasets/gta5train/', seq_len)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        sampler=SequenceSampler(train_dataset.end_len, seq_len))

    val_dataset = GTASequenceDataset('../datasets/gta5val/', val_seq_len)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size,
        sampler=SequenceSampler(val_dataset.end_len, val_seq_len))
    
    # Train model
    model.to(device)
    train(model, loss_fn, optimizer, epochs, train_loader, val_loader,
          model_name=model_name,
          model_save_dir=save_dir,
          device=device)

if __name__ == '__main__':
    #train_ConvLSTM()
    train_GTAResNet()