import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import glob

'''
Adapted from https://discuss.pytorch.org/t/how-upload-sequence-of-image-on-video-classification/24865/9
'''
class SequenceSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, seq_length):
        lengths = [0, *lengths]
        indices = []
        
        for i in range(len(lengths) - 1):
            start = lengths[i]
            end = lengths[i+1] - seq_length
            if start < end:
                indices.append(torch.arange(start, end + 1))
        
        self.indices = torch.cat(indices)
    
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.indices)

class GTADataset(Dataset):
    def __init__(self, root, balance=True):
        self.root = root
        self.loader = default_loader
        self.transform = transforms.ToTensor()

        self.images = []
        
        

        for filename in glob.glob(self.root + '*/*.txt'):
            curr_dir = filename.replace('\\', '/')
            curr_dir = curr_dir[0:curr_dir.rfind('/') + 1]
            with open(filename) as f:
                num_straight = 10
                for line in f.readlines():
                    split_line = line[:-1].split('\t')
                    sample_path = split_line[0]
                    target = torch.Tensor([float(x) for x in split_line[1:]])
                    if balance:
                        # turn = target[0], gas = target[1], break = target[2]
                        if (target[0] > 0.2 or target[0] < -0.2) or (target[2] > -0.4):
                            self.images.append((curr_dir + sample_path, target))
                            num_straight += 1
                        elif num_straight > 0:
                            self.images.append((curr_dir + sample_path, target))
                            num_straight -= 3
                    else:
                        self.images.append((curr_dir + sample_path, target))
        
        print(len(self.images))
    
    def __getitem__(self, index):
        img_path, target = self.images[index]
        img = self.transform(self.loader(img_path))
        return img, target
    
    def __len__(self):
        return len(self.images)

class GTASequenceDataset(Dataset):
    def __init__(self, root, conv_model_name, seq_length):
        self.root = root
        self.loader = default_loader
        self.transform = transforms.ToTensor()
        self.seq_length = seq_length
        self.end_len = []
        self.curr_len = 0

        self.embeds = []
        self.targets = []

        for filename in glob.glob(self.root + '*/' + conv_model_name + '-embeds.pth'):
            embed = torch.load(filename)
            if embed.size(1) != 2048:
                print(embed.size())
            self.curr_len += embed.size(0)
            self.end_len.append(self.curr_len)
            self.embeds.append(embed)

        for filename in glob.glob(self.root + '*/' + conv_model_name + '-targets.pth'):
            targets = torch.load(filename)
            self.targets.append(targets)
        
        self.embeds = torch.cat(self.embeds, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

    def __getitem__(self, index):
        start = index
        end = self.seq_length

        return self.embeds.narrow(0, start, end), self.targets.narrow(0, start, end)