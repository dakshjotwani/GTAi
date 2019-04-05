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

class GTASequenceDataset(Dataset):
    def __init__(self, root, seq_length):
        self.root = root
        self.loader = default_loader
        self.transform = transforms.ToTensor()
        self.seq_length = seq_length
        self.end_len = []

        self.images = []
        for filename in glob.glob(self.root + '*.txt'):
            with open(filename) as f:
                for line in f.readlines():
                    split_line = line[:-1].split('\t')
                    sample_path = split_line[0]
                    target = torch.Tensor([float(x) for x in split_line[1:]])
                    self.images.append((root + sample_path, target))

            self.end_len.append(len(self.images))

    def __getitem__(self, index):
        start = index
        end = index + self.seq_length

        images = []
        targets = []
        for i in range(start, end):
            img_path, target = self.images[i]
            img = self.transform(self.loader(img_path))
            images.append(img)
            targets.append(target)
        
        return torch.stack(images), torch.stack(targets) 