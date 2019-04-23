import torch
from models import GTAResNet
from data import img_2_vec

model_name = 'FinalResNet50-2'
model = GTAResNet()
model.load_state_dict(torch.load('../models/FinalResNet50-2.pt'))
batch_size = 128
dset_path = '../datasets/gta5train/'
device = torch.device('cuda:0')
model.to(device)

img_2_vec(model, model_name, batch_size, dset_path, device)

dset_path = '../datasets/gta5val/'

img_2_vec(model, model_name, batch_size, dset_path, device)