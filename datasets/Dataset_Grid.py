
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt


"""
Generates a dataset of 8 Gaussian distributions with their means distributed along a ring 
"""
class Dataset_GRID(Dataset):
    def __init__(self, std_dev = 0.01, num_samples=100000, seed=0):
        np.random.seed(seed)
        """
        Args:
            by defualt the RING is generated along the unit radius circle
            radius changes the radius of the circle by that factor
            num_samples is the size of the dataset generated
        """
        self._d_size = (num_samples//25)*25

        _rows = [0,1,2,3,4]
        _cols = [0,1,2,3,4]

        centers = []

        for _x in _cols:
            for _y in _rows:
                centers.append(_x, _y)


        self.dataset = []
        for center in centers:
            points = np.random.randn(self._d_size//25, 2)*std_dev
            points[:, 0] += center[0]
            points[: ,1] += center[1]
            self.dataset.append(points)

        self.dataset = np.asarray(self.dataset)
        _shape = self.dataset.shape
        self.dataset = np.reshape(self.dataset, (_shape[0]*_shape[1], _shape[2]))


    def __len__(self):
        return self._d_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[idx]

