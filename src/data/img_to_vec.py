import torch
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import numpy as np
import glob

def img_2_vec(model, model_name, batch_size, dset_path, device):
    model.eval()
    loader = default_loader
    transform = transforms.ToTensor()

    for filename in glob.glob(dset_path + '*/*.txt'):
        img_paths = []
        targets = []
        embeds = []

        curr_dir = filename.replace('\\', '/')
        dir_name = curr_dir[curr_dir.rfind('/') + 1 : -4]
        curr_dir = curr_dir[0:curr_dir.rfind('/') + 1]

        print('Processing ' + dir_name + '. ', end='')

        with open(filename) as f:
            for line in f.readlines():
                split_line = line[:-1].split('\t')
                sample_path = split_line[0]
                target = torch.Tensor([float(x) for x in split_line[1:]])

                img_paths.append(curr_dir + sample_path)
                targets.append(target)
                
                if len(img_paths) == batch_size:
                    imgs = [transform(loader(path)) for path in img_paths]
                    imgs = torch.stack(imgs).to(device)
                    
                    with torch.no_grad():
                        embeds.append(model.conv_forward(imgs).cpu())
                    
                    img_paths = []
            
            if len(img_paths) > 0:
                imgs = [transform(loader(path)) for path in img_paths]
                imgs = torch.stack(imgs).to(device)
                
                with torch.no_grad():
                    embeds.append(model.conv_forward(imgs).cpu())
                
                img_paths = []
        
        targets = torch.stack(targets)
        embeds = torch.cat(embeds, dim=0)

        torch.save(embeds, curr_dir + model_name + '-embeds.pth')
        torch.save(targets, curr_dir + model_name + '-targets.pth')
        
        print('Done.')