import os
import torch
import torchvision
from torchvision import transforms


def load_dataset(dataset='MNIST', batch_size=128, threads=2):


    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mnistTrainSet = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform)
    kwargs = {'num_workers': 2, 'pin_memory': True}
    mnistTrainLoader = torch.utils.data.DataLoader(mnistTrainSet, batch_size=batch_size,
                                                   shuffle=False, **kwargs)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=threads)

    return mnistTrainLoader, test_loader
load_dataset()