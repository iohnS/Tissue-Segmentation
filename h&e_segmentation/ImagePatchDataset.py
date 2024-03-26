import torch
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import random as rnd

# Lägg alla patches i en folder som heter train_patches och en tetch_patches. 
        
    
def partition_patches(type: str, distribution: float = 0.75):
    types = ["train", "test"]
    if type not in types:
        raise ValueError("Invalid sim type. Expected one of: %s" % types)
    
    path_dir = 'train_patches' if type == "train" else 'test_patches'
    label_dir = 'Labels/'
    dir_path = os.path.join(path_dir, label_dir)
    count = 0
    trainIndexes = []
    labelIndexes = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            if rnd.random() < distribution:                                     # 75% of the patches are used for training, 25% for validation
                trainIndexes.append(count)
            else:
                labelIndexes.append(count)
            count += 1
    if len(trainIndexes) == 0:
        return PatchDataset(labelIndexes, path_dir)
    elif len(labelIndexes) == 0:
        return PatchDataset(trainIndexes, path_dir)
    else:
        return PatchDataset(trainIndexes, path_dir), PatchDataset(labelIndexes, path_dir)

class PatchDataset(Dataset):
    def __init__(self, indxs, path_dir):
        self.label_dir = 'Labels/'
        self.train_dir = 'Patches/'
        self.path_dir = path_dir
        self.length = len(indxs)
        self.indxs = indxs
        self.datapairs = [(
            read_image(os.path.join(self.path_dir, self.train_dir, 'patch_' + str(i) + '.png')), 
            read_image(os.path.join(self.path_dir, self.label_dir, 'labels_' + str(i) + '.png'))
                )for i in indxs]
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.datapairs[index]