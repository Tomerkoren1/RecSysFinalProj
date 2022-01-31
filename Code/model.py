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
    def __init__(self, hidden, learning_rate, batch_size, device, use_logs):
        self.batch_size = batch_size
        self.net = nets.AutoEncoder(hidden)

        self.device = device

        self.net = self.net.to(device=self.device)
        
        if(use_logs):
            wandb.watch(self.net)

        #self.opt = optim.Adam(self.net.parameters(), learning_rate)
        self.opt = optim.SGD(self.net.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
        # self.opt = self.opt.to(device=device)                                                           
        self.feature_size = hidden[0] # n_user/n_item

    def run(self, trainset, testlist, num_epoch, use_logs, plot = True):
        RMSE = []
        for epoch in tqdm(range(1, num_epoch + 1)):
            #print "Epoch %d, at %s" % (epoch, datetime.now())
            train_loader = DataLoader(trainset, self.batch_size, shuffle=True, pin_memory=True)
            self.train(train_loader, epoch, use_logs)
            rmse = self.test(trainset, testlist, epoch)
            if(use_logs):
                wandb.log({"rmse": rmse}, step=epoch)
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
    def train(self, train_loader, epoch, use_logs):
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
            loss = torch.sqrt(F.mse_loss(output* masks, features* masks))
            loss = loss.to(device=self.device)
            if(use_logs):
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
