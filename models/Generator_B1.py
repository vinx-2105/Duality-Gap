import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter 

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

class Generator_B1(nn.Module):
    def __init__(self, d_z=100, d_h=128):
        super(Generator_B1, self).__init__()
        self.d_z = d_z
        self.d_h = d_h
        self.main = nn.Sequential(
            nn.Linear(self.d_z, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, 2),
        )

    def forward(self, input):
        return self.main(input)