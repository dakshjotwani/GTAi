import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torchvision import transforms

from PIL import Image
import glob

import os
import os.path
import sys

class gta5Loader(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.loader = default_loader

        self.images = []
        labels = {}
        for filename in glob.glob(self.root + '*.txt'):
            with open(filename) as f:
                for line in f.readlines():
                    split_line = line[:-1].split('\t')
                    labels[split_line[0]] = [float(x) for x in split_line[1:]]
        for filename in glob.glob(self.root + '*.jpg'):
            self.images.append((filename, labels[filename[len(self.root):]]))

        self.transform = transforms.ToTensor()
        self.target_transform = None

    def __getitem__(self, index):
        #print(index)
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if isinstance(index, int):
            path, target = self.images[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        elif isinstance(index, list):
            samples, targets = [], []
            for i in index:
                path, target = self.images[i]
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                samples.append(sample.unsqueeze(0))
                targets.append(target)
            return torch.cat(samples).squeeze(0), torch.tensor(targets)


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

# import os
# root = 'D:/gta5train/'
# img_names = [n[len(root):] for n in glob.glob(root + '*.jpg')]
# txts = [n[len(root):] for n in glob.glob(root + '*.txt')]
# # root = './tmp/'
# for txt in txts:
#     name = txt[:-4]
#     # print(name)
#     folder = root + name
#     os.mkdir(folder)
#     for jpg in img_names:
#         if not jpg.startswith(name):
#             continue
#         # print(root + jpg, folder + '/' + jpg)
#         os.rename(root + jpg, folder + '/' + jpg)
#     os.rename(root + txt, folder + '/' + txt)
#     # print(root + txt, folder + '/' + txt)
#     # break 
