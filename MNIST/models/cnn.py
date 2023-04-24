import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(6, 9, 3)
        self.conv3 = nn.Conv2d(9, 12, 3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(12 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flat(F.relu(self.conv3(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def CNNentropy():
    return (CNN(), torch.nn.CrossEntropyLoss)

def CNNnll():
    return(CNN(), torch.nn.NLLLoss)

def CNNkldiv():
    return (CNN(), torch.nn.KLDivLoss)

def CNNpoissoinnll():
    return (CNN(), torch.nn.PoissonNLLLoss)

def CNNgaussiannll():
    return (CNN(), torch.nn.GaussianNLLLoss)

        
