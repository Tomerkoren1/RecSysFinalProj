#model.py
#Modified by ImKe on 2019/9/4.
#Copyright © 2019 ImKe. All rights reserved.

import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import optim, nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb


import networks as nets

class Model:
    def __init__(self, hidden, learning_rate, batch_size, dropout, device, use_wandb):
        self.batch_size = batch_size
        self.net = nets.AutoEncoder(hidden,dropout=dropout)

        self.device = device

        self.net = self.net.to(device=self.device)
        
        if(use_wandb):
            wandb.watch(self.net)

        # self.opt = optim.Adam(self.net.parameters(), learning_rate, weight_decay=1*0.5)
        self.opt = optim.SGD(self.net.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
                                                         
        self.feature_size = hidden[0] # n_user/n_item

    def run(self, trainset, testlist, num_epoch, use_wandb, plot = True):
        RMSE = []
        pbar = tqdm(range(1, num_epoch + 1))
        for epoch in pbar:
            #print "Epoch %d, at %s" % (epoch, datetime.now())
            train_loader = DataLoader(trainset, self.batch_size, shuffle=True)
            self.train(train_loader, epoch, use_wandb)
            rmse = self.test(trainset, testlist, epoch)
            if(use_wandb):
                wandb.log({"rmse": rmse}, step=epoch)
            else:
                pbar.set_postfix({'rmse': rmse})
            RMSE.append(rmse)
        if plot:
            x_label = np.arange(0,num_epoch,1)
            plt.plot(x_label, RMSE, 'b-.')
            my_x_ticks = np.arange(0, num_epoch, 50)
            plt.xticks(my_x_ticks)
            plt.title("RMSE of testing data")
            plt.xlabel("Number of epoch")
            plt.ylabel("RMSE")
            plt.grid()
            plt.show()


    #批训练
    def train(self, train_loader, epoch, use_wandb):
        self.net.train()
        features = Variable(torch.FloatTensor(self.batch_size, self.feature_size).to(device=self.device))
        masks = Variable(torch.FloatTensor(self.batch_size, self.feature_size).to(device=self.device))

        for bid, (feature, mask) in enumerate(train_loader):
            feature = feature.to(device=self.device)
            mask = mask.to(device=self.device)
            if mask.shape[0] == self.batch_size:
                features.data.copy_(feature)
                masks.data.copy_(mask)
            else:
                features = Variable(feature)
                masks = Variable(mask)
            self.opt.zero_grad()
            output = self.net(features)
            loss = torch.sqrt(F.mse_loss(output* masks, features* masks, reduction='sum')/masks.sum().item())
            loss = loss.to(device=self.device)
            if(use_wandb):
                wandb.log({"train_loss": loss}, step=epoch)
            loss.backward()
            self.opt.step()

    def test(self, trainset, testlist, epoch, display_step = 10):
        self.net.eval()
        x_mat, mask, user_based = trainset.get_mat()
        features = Variable(x_mat.to(device=self.device))
        xc = self.net(features)
        if not user_based:
            xc = xc.t()
        xc = xc.cpu().data.numpy()

        rmse = 0.0
        for (i, j, r) in testlist:
            rmse += (xc[i][j]-r)*(xc[i][j]-r)
        rmse = math.sqrt(rmse / len(testlist))

        # if (epoch % display_step == 0):
        #     print (" Test RMSE = %f" % rmse)
        return rmse



class OurModel:
    def __init__(self, hidden, learning_rate, batch_size, act_type, dropout, momentum, weight_decay, device, use_wandb):
        self.batch_size = batch_size
        self.net = nets.OurAutoEncoder(hidden,act_type,dropout=dropout)

        self.device = device

        self.net = self.net.to(device=self.device)
        
        if(use_wandb):
            wandb.watch(self.net)

        # self.opt = optim.Adagrad(self.net.parameters(), weight_decay=1e-5)
        # self.opt = optim.Adam(self.net.parameters(),lr=1e-5,weight_decay=1e-7)
        self.opt = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                                                         
        self.feature_size = hidden[0] # n_user/n_item

    def run(self, trainset, testlist, num_epoch, use_wandb, plot = True):
        RMSE = []
        pbar = tqdm(range(1, num_epoch + 1))
        for epoch in pbar:
            #print "Epoch %d, at %s" % (epoch, datetime.now())
            train_loader = DataLoader(trainset, self.batch_size, shuffle=True)
            self.train(train_loader, epoch, use_wandb)
            rmse = self.test(trainset, testlist, epoch)
            if(use_wandb):
                wandb.log({"rmse": rmse}, step=epoch)
            else:
                pbar.set_postfix({'rmse': rmse})
                
            RMSE.append(rmse)
        if plot:
            x_label = np.arange(0,num_epoch,1)
            plt.plot(x_label, RMSE, 'b-.')
            my_x_ticks = np.arange(0, num_epoch, 50)
            plt.xticks(my_x_ticks)
            plt.title("RMSE of testing data")
            plt.xlabel("Number of epoch")
            plt.ylabel("RMSE")
            plt.grid()
            plt.show()


    #批训练
    def train(self, train_loader, epoch, use_wandb):
        self.net.train()
        features = Variable(torch.FloatTensor(self.batch_size, self.feature_size).to(device=self.device))
        masks = Variable(torch.FloatTensor(self.batch_size, self.feature_size).to(device=self.device))

        for bid, (feature, mask) in enumerate(train_loader):
            feature = feature.to(device=self.device)
            mask = mask.to(device=self.device)
            if mask.shape[0] == self.batch_size:
                features.data.copy_(feature)
                masks.data.copy_(mask)
            else:
                features = Variable(feature)
                masks = Variable(mask)
            self.opt.zero_grad()
            output = self.net(features)
            loss = torch.sqrt(F.mse_loss(output* masks, features* masks, reduction='sum')/masks.sum().item())
            loss = loss.to(device=self.device)
            if(use_wandb):
                wandb.log({"train_loss": loss}, step=epoch)
            loss.backward()
            self.opt.step()

    def test(self, trainset, testlist, epoch, display_step = 10):
        self.net.eval()
        x_mat, mask, user_based = trainset.get_mat()
        features = Variable(x_mat.to(device=self.device))
        xc = self.net(features)
        if not user_based:
            xc = xc.t()
        xc = xc.cpu().data.numpy()

        rmse = 0.0
        for (i, j, r) in testlist:
            rmse += (xc[i][j]-r)*(xc[i][j]-r)
        rmse = math.sqrt(rmse / len(testlist))

        # if (epoch % display_step == 0):
        #     print (" Test RMSE = %f" % rmse)
        return rmse