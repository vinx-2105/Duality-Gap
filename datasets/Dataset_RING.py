
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt


"""
Generates a dataset of 8 Gaussian distributions with their means distributed along a ring 
"""
class Dataset_RING(Dataset):
    def __init__(self, radius=1, num_samples=100000):
        np.random.seed(0)
        """
        Args:
            by defualt the RING is generated along the unit radius circle
            radius changes the radius of the circle by that factor
            num_samples is the size of the dataset generated
        """
        self._d_size = (num_samples//8)*8
        self._radius = radius

        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            ( 0, -1),
            (1. /np.sqrt(2), 1. /np.sqrt(2)),
            (-1. /np.sqrt(2), 1. /np.sqrt(2)),
            (-1. /np.sqrt(2), -1. /np.sqrt(2)),
            (1. /np.sqrt(2), -1. /np.sqrt(2)),
        ]

        centers = [(radius*x, radius*y) for x, y in centers]

        self.dataset = []
        for center in centers:
            points = np.random.randn(self._d_size//8, 2)*0.02
            points[:, 0] += center[0]
            points[: ,1] += center[1]
            # print(points)
            self.dataset.append(points)

        self.dataset = np.asarray(self.dataset)
        _shape = self.dataset.shape
        self.dataset = np.reshape(self.dataset, (_shape[0]*_shape[1], _shape[2]))
        print(self.dataset)


    def __len__(self):
        return self._radius

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[idx]

