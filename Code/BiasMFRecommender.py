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

class BiasMF(torch.nn.Module):
    def __init__(self, params):
        super(BiasMF, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']
        self.mu = params['global_mean']

        self.device = params['device']

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim,device=self.device)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim,device=self.device)

        self.user_bias = torch.nn.Embedding(self.num_users, 1, device=self.device)
        self.user_bias.weight.data = torch.zeros(self.num_users, 1, device=self.device).float()
        self.item_bias = torch.nn.Embedding(self.num_items, 1, device=self.device)
        self.item_bias.weight.data = torch.zeros(self.num_items, 1, device=self.device).float()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        dot = torch.mul(user_vec, item_vec).sum(dim=-1)

        rating = dot + self.mu + self.user_bias(user_indices).view(-1) + self.item_bias(item_indices).view(-1) + self.mu

        return rating

    def fit(self, train_loader, val_dataset, num_epoch, use_logs):
        for epoch in range(num_epoch//10):
            for bid, batch in enumerate(train_loader):
                u, i, r = batch[0].to(device=self.device), batch[1].to(device=self.device), batch[2].to(device=self.device)
                r = r.float()
                # forward pass
                preds = self.forward(u, i)
                loss = torch.sqrt(self.criterion(preds, r))
                loss = loss.to(device=self.device)
                if(use_logs):
                    wandb.log({"train_loss": loss}, step=epoch)
                # backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            rmse = self.validate(val_dataset)
            if(use_logs):
                wandb.log({"rmse": rmse}, step=epoch)
            print('Epoch [{}/30], Loss: {:.4f}, RMSE: {:.4f}'.format(epoch + 1, loss.item(), rmse))

    def validate(self, val_dataset):
        self.eval()
        rmse = 0.0
        for bid, batch in enumerate(val_dataset):
            u, i, r = batch[0].to(device=self.device), batch[1].to(device=self.device), batch[2].to(device=self.device)
            preds = self.forward(u, i)
            rmse += (preds-r)*(preds-r)
        rmse = math.sqrt(rmse / len(val_dataset))
        return rmse