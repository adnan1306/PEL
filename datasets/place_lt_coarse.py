import os
from .lt_data_coarse import LT_Dataset
import numpy as np

class Places_LT_coarse(LT_Dataset):
    classnames_txt = "datasets/Places_LT/reduced_classnames.txt"
    train_txt = "datasets/Places_LT/Places_LT_train.txt"
    test_txt = "datasets/Places_LT/Places_LT_test.txt"

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)

        self.classnames = self.read_classnames()
        self.map = np.load('datasets/Places_LT/reduced_map.npy')
        
        
        self.names = []
        with open(self.txt) as f:
            for line in f:
                self.names.append(self.classnames[int(self.map[int(line.split()[1])])])

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                # line = line.strip().split(" ")
                folder = line
                classnames.append(folder)
        return classnames
    
