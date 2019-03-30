import torch.utils.data as data
from torchvision.datasets.folder import default_loader

from PIL import Image
import glob

import os
import os.path
import sys

class gta5Loader(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.loader = default_loader

        self.images = []
        labels = {}
        for filename in glob.glob(self.root + '*.txt'):
            with open(filename) as f:
                for line in f.readlines():
                    split_line = line[:-1].split('\t')
                    labels[split_line[0]] = [(float(x)+1)/2 for x in split_line[1:]]
        for filename in glob.glob(self.root + '*.jpg'):
            self.images.append((filename, labels[filename[len(self.root):]]))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.images[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
