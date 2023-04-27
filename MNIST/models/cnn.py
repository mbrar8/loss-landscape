import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(3, 5, 3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(5 * 2 * 2, 10)


    def forward(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.flat(x)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        return x




def CNNentropy():
    return (CNN(), torch.nn.CrossEntropyLoss())

def CNNnll():
    return(CNN(), torch.nn.NLLLoss())

def CNNmargin():
    return (CNN(), torch.nn.MultiMarginLoss())

def CNNkldiv():
    return (CNN(), torch.nn.KLDivLoss())

def CNNpoissoinnll():
    return (CNN(), torch.nn.PoissonNLLLoss())

def CNNgaussiannll():
    return (CNN(), torch.nn.GaussianNLLLoss())

        
