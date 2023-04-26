import os
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel

import model_loader
import dataloader


def train(trainloader, net, lossfxn, optimizer, use_cuda=True):
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
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        batch_size = inputs.size(0)
        total += batch_size

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = lossfxn(outputs, targets)
        test_loss += loss.item()*batch_size
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

    return test_loss/total, 100 - 100.*correct/total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='dense_entropy')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    #print('Current devices: ' + str(torch.cuda.current_device()))
    #print('Device ct: ' + str(torch.cuda.device_count()))

    lr = 0.0001
    st_epoch = 1

    save_folder = args.model
    if not os.path.exists('trained/' + save_folder):
        os.makedirs('trained/' + save_folder)

    #f = open('trained/' + save_folder + '/log.out', 'a', 0)

    trainloader, testloader = dataloader.load_dataset()

    net, lossfxn = model_loader.load(args.model)
    
    if use_cuda:
        net.cuda()
        lossfxn = lossfxn.cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)

    for epoch in range(1, 11):
        print('Epoch: ' + str(epoch))
        model.train(True)
        running_loss = 0.0
        last_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            print("data")
            print(outputs)
            print(labels)
            loss = lossfxn(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print('   batch {} loss: {}'.format(i+1, last_loss))
                running_loss = 0.0 
        model.train(False)
        val_loss = 0.0
        for i, vdata in enumerate(testloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = lossfxn(voutputs, vlabels)
            val_loss += vloss
        avg_vloss = val_loss / (i + 1)
        print('LOSS train {} valid {}'.format(last_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '{}model_{}'.format(save_path, epoch) 
            torch.save(model.state_dict(), model_path)


	epoch += 1
