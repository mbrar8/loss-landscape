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
    save_path = 'trained/' + save_folder + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainloader, testloader = dataloader.load_dataset()

    model, lossfxn = model_loader.load(args.model)
    
    #if use_cuda:
    #    model.cuda()
    #    lossfxn = lossfxn.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

    best_vloss = 100000000

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
            if isinstance(lossfxn, torch.nn.GaussianNLLLoss):
                print(outputs.shape)
                print(outputs.size())
                var = torch.ones(128, 1, requires_grad=True)
                print(var.size())
                loss = lossfxn(outputs, labels, var)
            else:
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
