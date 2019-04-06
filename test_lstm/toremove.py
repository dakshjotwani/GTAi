import numpy as np
import time
from torch.optim import Adam

def train(device, model, epochs, lr, filename='model'):
    learning_rate = lr
    max_epoch = epochs
    batch_size = 16
    seq_len = 8
    val_batch_size = 16
    val_seq_len = 8

    # Set data loader
    train_dataset = GTASequenceDataset('./datasets/gta5train/', seq_len)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        sampler=SequenceSampler(train_dataset.end_len, seq_len))

    val_dataset = GTASequenceDataset('./datasets/gta5val/', val_seq_len)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=batch_size,
        sampler=SequenceSampler(val_dataset.end_len, val_seq_len))

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
            prediction, _ = model(n[0].to(device))
            target = n[1].to(device)

            # calculate loss and backprop
            # print(prediction)
            # print(target)

            # compute loss, append to loss list for print output
            loss = loss_fn(prediction, target)
            current_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            
            # print
            if (i+1)%(mini_batches_per_print) == 0:
                print(f'epoch {current_epoch:d} ({100*i/len(train_loader):.0f}%) | batch {i:d} ({i*batch_size:d})' +
                f' | loss {loss.item():.2f} | time {time.time()-start_time:.1f}s')
                loss_list.append(np.average(current_loss))
                current_loss = []
#                print(' |', get_control_str(prediction[0]), '|', get_control_str(target[0]))

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
            prediction, _ = model(n[0].to(device))
            target = n[1].to(device)
            loss = loss_fn(prediction, target)
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