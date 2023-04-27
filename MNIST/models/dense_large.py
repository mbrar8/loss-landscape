# modified from https://www.kaggle.com/code/pragyanbo/pytorch-mnist-dense-neural-network
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable



class LargeDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 712)
        self.fc2 = nn.Linear(712, 680)
        self.fc3 = nn.Linear(680, 600)
        self.fc4 = nn.Linear(600, 512)
        self.fc5 = nn.Linear(512, 496)
        self.fc6 = nn.Linear(496, 412)
        self.fc7 = nn.Linear(412, 380)
        self.fc8 = nn.Linear(380, 315)
        self.fc9 = nn.Linear(315, 256)
        self.fc10 = nn.Linear(256, 212)
        self.fc11 = nn.Linear(212, 196)
        self.fc12 = nn.Linear(196, 165)
        self.fc13 = nn.Linear(165, 128)
        self.fc14 = nn.Linear(128, 100)
        self.fc15 = nn.Linear(100, 75)
        self.fc16 = nn.Linear(75, 64)
        self.fc17 = nn.Linear(64, 50)
        self.fc18 = nn.Linear(50, 32)
        self.fc19 = nn.Linear(32, 16)
        self.fc20 = nn.Linear(16, 10)



    def forward(self, x):
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc16(x))
        x = F.relu(self.fc17(x))
        x = F.relu(self.fc18(x))
        x = F.relu(self.fc19(x))
        x = self.fc20(x)

        return x
    
def LargeDenseEntropy():
    return (LargeDense(), torch.nn.CrossEntropyLoss())

def LargeDensenll():
    return (LargeDense(), torch.nn.NLLLoss())


def LargeDensemargin():
    return (LargeDense(), torch.nn.MultiMarginLoss())
