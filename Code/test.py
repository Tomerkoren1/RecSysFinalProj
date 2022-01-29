'''
@Author: Yu Di
@Date: 2019-08-09 14:04:38
@LastEditors: Yudi
@LastEditTime: 2019-08-15 16:05:55
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description:  this is a demo for SVD++ recommendation
'''
from itertools import count
import torch
from torch.utils.data import DataLoader, Dataset

from BiasMFRecommender import BiasMF
from dataloader import load_data
import math

class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)

def test(model, test_dataset, device):
    model.eval()
    rmse = 0.0
    for bid, batch in enumerate(test_dataset):
        u, i, r = batch[0].to(device=device), batch[1].to(device=device), batch[2].to(device=device)
        preds = model(u, i, False)
        if(r==0):
            print('Nir')
        rmse += (preds-r)*(preds-r)
    rmse = math.sqrt(rmse / len(test_dataset))
    return rmse

train_list, test_list, n_user, n_item = load_data('ratings', 0.9)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print('neural network device: %s (CUDA available: %s, count: %d)',
        device, torch.cuda.is_available(), torch.cuda.device_count())

# Train
user_tensor = torch.LongTensor([val[0] for val in train_list]).to(device = device)
item_tensor = torch.LongTensor([val[1] for val in train_list]).to(device = device)
rating_tensor = torch.FloatTensor([val[2] for val in train_list]).to(device = device)
dataset = RateDataset(user_tensor, item_tensor, rating_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Test
user_tensor = torch.LongTensor([val[0] for val in test_list]).to(device = device)
item_tensor = torch.LongTensor([val[1] for val in test_list]).to(device = device)
rating_tensor = torch.FloatTensor([val[2] for val in test_list]).to(device = device)
test_dataset = RateDataset(user_tensor, item_tensor, rating_tensor)

params = {'num_users': n_user, 
          'num_items': n_item,
          'global_mean': 3, 
          'latent_dim': 20
        }

model = BiasMF(params)
model = model.to(device=device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



for epoch in range(30):
    for bid, batch in enumerate(train_loader):
        u, i, r = batch[0].to(device=device), batch[1].to(device=device), batch[2].to(device=device)
        r = r.float()
        # forward pass
        preds = model(u, i)
        loss = torch.sqrt(criterion(preds, r))
        loss = loss.to(device=device)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    rmse = test(model,test_dataset,device)
    print('Epoch [{}/30], Loss: {:.4f}, RMSE: {:.4f}'.format(epoch + 1, loss.item(), rmse))
    
    
    