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

class BiasMF(torch.nn.Module):
    def __init__(self, params):
        super(BiasMF, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']
        self.mu = params['global_mean']

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        print('neural network device: %s (CUDA available: %s, count: %d)',
                self.device, torch.cuda.is_available(), torch.cuda.device_count())

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim,device=self.device)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim,device=self.device)

        self.user_bias = torch.nn.Embedding(self.num_users, 1, device=self.device)
        self.user_bias.weight.data = torch.zeros(self.num_users, 1, device=self.device).float()
        self.item_bias = torch.nn.Embedding(self.num_items, 1, device=self.device)
        self.item_bias.weight.data = torch.zeros(self.num_items, 1, device=self.device).float()

    def forward(self, user_indices, item_indices,train=True):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        dot = torch.mul(user_vec, item_vec).sum(dim=-1)

        rating = dot + self.mu + self.user_bias(user_indices).view(-1) + self.item_bias(item_indices).view(-1) + self.mu

        return rating
