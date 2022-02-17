'''
@Author: Yu Di
@Date: 2019-08-08 14:18:50
@LastEditors: Yudi
@LastEditTime: 2019-08-13 16:08:18
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import torch
import math
import wandb
import numpy as np
from tqdm import tqdm
import metrics

class BiasMF(torch.nn.Module):
    def __init__(self, params):
        super(BiasMF, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']
        self.mu = params['global_mean']
        self.lr = params['lr']
        self.device = params['device']

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim,device=self.device)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim,device=self.device)

        self.user_bias = torch.nn.Embedding(self.num_users, 1, device=self.device)
        self.user_bias.weight.data = torch.zeros(self.num_users, 1, device=self.device).float()
        self.item_bias = torch.nn.Embedding(self.num_items, 1, device=self.device)
        self.item_bias.weight.data = torch.zeros(self.num_items, 1, device=self.device).float()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        dot = torch.mul(user_vec, item_vec).sum(dim=-1)

        rating = dot + self.mu + self.user_bias(user_indices).view(-1) + self.item_bias(item_indices).view(-1) + self.mu

        return rating

    def fit(self, train_loader, val_dataset, num_epoch, use_wandb):

        pbar = tqdm(range(1, num_epoch//10 + 1))
        for epoch in pbar:
            for bid, batch in enumerate(train_loader):
                u, i, r = batch[0].to(device=self.device), batch[1].to(device=self.device), batch[2].to(device=self.device)
                r = r.float()
                # forward pass
                preds = self.forward(u, i)
                loss = torch.sqrt(self.criterion(preds, r))
                loss = loss.to(device=self.device)
                if(use_wandb):
                    wandb.log({"train_loss": loss}, step=epoch)
                # backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            rmse, mrr, nDCG = self.validate(val_dataset)
            if(use_wandb):
                wandb.log({"rmse": rmse, 'MRR': mrr, 'nDCG': nDCG}, step=epoch)
            pbar.set_postfix({'rmse': rmse, 'MRR': mrr, 'nDCG': nDCG})

    def validate(self, val_dataset):
        self.eval()
        rmse = 0.0
        users_true = np.zeros((self.num_users,self.num_items))
        users_pred = np.zeros((self.num_users,self.num_items))
        for bid, batch in enumerate(val_dataset):
            u, i, r = batch[0].to(device=self.device), batch[1].to(device=self.device), batch[2].to(device=self.device)
            preds = self.forward(u, i)
            rmse += (preds-r)*(preds-r)
            users_true[u][i] = r.item()
            users_pred[u][i] = preds.item()
        rmse = math.sqrt(rmse / len(val_dataset))

        users_num = users_pred.shape[0]
        total_MRR = 0
        total_nDCG = 0
        zero_cnt = 0
        for u in range(users_num):
            user_true = users_true[u]
            user_pred = users_pred[u]
            if np.all(user_true == 0):
                zero_cnt += 1
                continue

            MRR_of_user = metrics.MRR_for_user(user_true, user_pred)
            NDCG_of_user = metrics.NDCG_for_user(user_true,user_pred)
            total_MRR += MRR_of_user
            total_nDCG += NDCG_of_user
        total_MRR /= (users_num - zero_cnt)
        total_nDCG /= (users_num - zero_cnt)

        return rmse, total_MRR, total_nDCG