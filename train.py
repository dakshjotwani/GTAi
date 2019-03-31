import torch
from torch import nn
from torch.optim import Adam, SGD
import torchvision
from torchvision import models
from torchvision import transforms
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import datetime
import cv2
import os

from gta5Loader import gta5Loader

class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()

        self.batch_size = 32
        self.learning_rate = 0.0001

        train = gta5Loader('./gta5train/', transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Tanh(),
            nn.Linear(4096, 3),
            nn.Tanh(),
        )
        self.loss = nn.SmoothL1Loss()
        self.optimizer = Adam(self.parameters(), lr = self.learning_rate)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


    def train(self, device):
        # variables to change
        stopping_epoch = 100
        stopping_val_acc = 1.00
        mini_batches_per_print = len(self.train_loader)//100
        mini_batches_per_val = len(self.train_loader)//10

        start_time = time.time()
        epochs = 0
        best_val_acc = 0
        stopping_criteria = False
        loss_list = []
        val_accs = []
        while epochs < stopping_epoch and not stopping_criteria:
            print('new epoch')
            correct = 0
            correct_current_print = 0
            total = 0
            total_current_print = 0
            current_loss = []
            
            for i, n in enumerate(self.train_loader):
                # forward pass
                prediction = self.forward(n[0].to(device))
                # calculate loss and backprop
                target = torch.stack([torch.DoubleTensor(n[1][0]), torch.DoubleTensor(n[1][1]), torch.DoubleTensor(n[1][2])])
                target = target.transpose(0, 1).float()
                # print(prediction)
                # print(target)
                loss = self.loss(prediction, target.to(device))
                current_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # print
                if (i+1)%(mini_batches_per_print) == 0 and i != 0:
                    print(f'epoch {epochs:d} ({100*i/len(self.train_loader):.0f}%) | batch {i:d} ({i*self.batch_size:d})' +
                    f' | loss {loss.item():.2f} | time {time.time()-start_time:.1f}s')
                    loss_list.append(np.average(current_loss))
                    current_loss = []
                    print(prediction[0], target[0])
                    print(prediction[1], target[1])
                    print(prediction[2], target[2])
                
            torch.save(self.state_dict(), './saved.pt')
            epochs += 1
        print('training time: ', time.time()-start_time)
        plt.plot([x*self.batch_size*mini_batches_per_print for x in range(len(loss_list))], loss_list)
        plt.xlabel('# of training samples')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Cross entropy loss vs number of training samples')
        plt.show()

        plt.plot([x*self.batch_size*mini_batches_per_val for x in range(len(val_accs))], val_accs)
        plt.xlabel('# of training samples')
        plt.ylabel('Validation accuracy')
        plt.title('Validation accuracy vs number of training samples')
        plt.show()
    
    def val(self, device):
        return 0
        correct = 0
        total = 0
        
        for i, n in enumerate(self.val_loader):
            if (i/len(self.val_loader)) > 0.33:
                break
            
            prediction = self.forward(n[0].to(device))
            _, prediction_num = prediction.max(1)
            correct += torch.sum(prediction_num == n[1].to(device)).item()
            total += self.batch_size
        return correct/total

if __name__ == "__main__":
    a = Alexnet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    a = a.to(device)
    a.train(device)
    