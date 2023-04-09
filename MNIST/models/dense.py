# modified from https://www.kaggle.com/code/pragyanbo/pytorch-mnist-dense-neural-network
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable



class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)



    def forward(self, x):
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
    
def DenseEntropy():
    return (Dense(), nn.CrossEntropyLoss)

def Densenll():
    return (Dense(), torch.nn.NLLLoss)

def Densekldiv():
    return (Dense(), torch.nn.KLDivLoss)

def Densepoissoinnll():
    return (Dense(), torch.nn.PoissonNLLLoss)

def Densegaussiannll():
    return (Dense(), torch.nn.GaussianNLLLoss)
