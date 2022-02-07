#main.py
#Modified by ImKe on 2019/9/5.
#Copyright © 2019 ImKe. All rights reserved.

from re import A
import setdata as sd
import model as model
from datetime import datetime
from dataloader import load_data
import argparse
import wandb
import torch
from dataloader import load_data
from BiasMFRecommender import BiasMF
from torch.utils.data import DataLoader, Dataset
import json


display_step = 10
plotbool = False
user_based = False
use_wandb = False

def cli():
    """ Handle argument parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.05,
                        help='learning_rate')
    parser.add_argument('--epoch_num', default=400,
                        help='epoch number')
    parser.add_argument('--batch_size', default=64,
                        help='batch_size')
    parser.add_argument('--hidden_num', default='[1024, 512]',
                        help='hidden layers number')
    parser.add_argument('--latent_dim', default=20,
                        help='MF latent dime')
    parser.add_argument('--model_name', default='OurAutoRec',
                        help='model name')
    parser.add_argument('--act_type', default='sigmoid',
                        help='activation type')
    parser.add_argument('--dropout', default=0.15,
                        help='dropout value')
    parser.add_argument('--momentum', default=0.996,
                        help='momentum value')
    parser.add_argument('--weight_decay', default=1e-5,
                        help='weight decay value')
    parser.add_argument('--dataset', default='ml-1m',
                        help='dataset type')

    args = parser.parse_args()

    return args

def trainOurAutoRec(args, train_list, val_list, n_user, n_item, user_based, device):
    trainset = sd.Dataset(train_list, n_user, n_item, user_based)
    if user_based:
        h = n_item
    else:
        h = n_user
    
    hidden_num = json.loads(userArgs.hidden_num)
    
    setmod = model.OurModel(hidden=[h, *hidden_num],
                      learning_rate=args.learning_rate,
                      batch_size=args.batch_size,
                      act_type = args.act_type,
                      dropout= args.dropout,
                      momentum= args.momentum,
                      weight_decay= args.weight_decay,
                      device = device,
                      use_wandb = use_wandb)

    RMSE = setmod.run(trainset, val_list, num_epoch = args.epoch_num, plot = plotbool, use_wandb = use_wandb)

def trainAutoRec(args, train_list, val_list, n_user, n_item, user_based, device):
    trainset = sd.Dataset(train_list, n_user, n_item, user_based)
    if user_based:
        h = n_item
    else:
        h = n_user

    setmod = model.Model(hidden=[h, *args.hidden_num],
                      learning_rate=args.learning_rate,
                      batch_size=args.batch_size,
                      dropout= args.dropout,
                      device = device,
                      use_wandb = use_wandb)

    RMSE = setmod.run(trainset, val_list, num_epoch = args.epoch_num, plot = plotbool, use_wandb = use_wandb)


def trainBiasMF(args, train_list, val_list, n_user, n_item, device):

    # Train
    user_tensor = torch.LongTensor([val[0] for val in train_list]).to(device = device)
    item_tensor = torch.LongTensor([val[1] for val in train_list]).to(device = device)
    rating_tensor = torch.FloatTensor([val[2] for val in train_list]).to(device = device)
    dataset = sd.RateDataset(user_tensor, item_tensor, rating_tensor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Test
    user_tensor = torch.LongTensor([val[0] for val in val_list]).to(device = device)
    item_tensor = torch.LongTensor([val[1] for val in val_list]).to(device = device)
    rating_tensor = torch.FloatTensor([val[2] for val in val_list]).to(device = device)
    val_dataset = sd.RateDataset(user_tensor, item_tensor, rating_tensor)

    params = {'num_users': n_user, 
            'num_items': n_item,
            'global_mean': 3, 
            'latent_dim': args.latent_dim,
            'device': device
            }

    model = BiasMF(params)
    model = model.to(device=device)

    if(use_wandb):
            wandb.watch(model)

    model.fit(train_loader=train_loader, val_dataset=val_dataset, num_epoch = args.epoch_num, use_wandb = use_wandb)


if __name__ == '__main__':
    start = datetime.now()
    userArgs = cli()
    if(use_wandb):
            wandb.init(project="RecFinalProject2", entity="tomerkoren", config=vars(userArgs))
            userArgs = wandb.config
            
    print(vars(userArgs))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('neural network device: %s (CUDA available: %s, count: %d)', device, torch.cuda.is_available(), torch.cuda.device_count())

    train_list, val_list, n_user, n_item = load_data(dataset=userArgs.dataset, file='ratings', train_ratio = 0.9)

    if( userArgs.model_name == 'AutoRec'):
        trainAutoRec(userArgs, train_list, val_list, n_user, n_item, user_based, device)
    elif( userArgs.model_name == 'BiasMF'):
        trainBiasMF(userArgs, train_list, val_list, n_user, n_item, device)
    elif( userArgs.model_name == 'OurAutoRec'):
        trainOurAutoRec(userArgs, train_list, val_list, n_user, n_item, user_based, device)

    end = datetime.now()
    print ("Total time: %s" % str(end-start))

