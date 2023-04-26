# modified from https://www.kaggle.com/code/pragyanbo/pytorch-mnist-dense-neural-network
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable



class SmallDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)



    def forward(self, x):
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def SmallDenseEntropy():
    return (SmallDense(), torch.nn.CrossEntropyLoss())

def SmallDensenll():
    return (SmallDense(), torch.nn.NLLLoss())

def Densekldiv():
    return (Dense(), torch.nn.KLDivLoss())

def Densepoissoinnll():
    return (Dense(), torch.nn.PoissonNLLLoss())

def Densegaussiannll():
    return (Dense(), torch.nn.GaussianNLLLoss())
