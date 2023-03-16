import os
import numpy as np

import torch
import torch.nn as nn

import model_loader
import dataloader



def train(trainloader, net, lossfxn, optimizer, use_cuda=True):
    #TODO: Determine what if this is still necessary
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_size = inputs.size(0)
        total += batch_size
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = lossfxn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*batch_size
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()


    return train_loss/total, 100 - 100.*correct/total


def test(testloader, net, lossfxn, use_cuda=True):
    print("TODO")


def save():
    print("TODO")


if __name__ == '__main__':
    # Parser stuff
    use_cuda = torch.cuda.is_available()


    # Rest of stuff

    # Change model_loader to this
    net, loss = model_loader.load(args.model)
    



