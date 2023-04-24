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
        train_loss, train_err = train(trainloader, net, lossfxn, optimizer, use_cuda)
        test_loss, test_err = test(testloader, net, lossfxn, use_cuda)
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (epoch, train_loss, train_err, test_err, test_loss)
        print(status)
        #f.write(status)

        acc = 100 - test_err
        if epoch == 10:
           state = {
             'acc': acc,
             'epoch': epoch,
             'state_dict': net.state_dict()
           }
           opt_state = {
              'optimizer': optimizer.state_dict()
           }
           torch.save(state, 'trained/' + save_folder + '/model_' + str(epoch) + '.t7')
           torch.save(opt_state, 'trained/' + save_folder + '/opt_state_' + str(epoch) + '.t7')

    #f.close()   
    



