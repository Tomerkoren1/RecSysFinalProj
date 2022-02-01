#main.py
#Modified by ImKe on 2019/9/5.
#Copyright © 2019 ImKe. All rights reserved.

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


display_step = 10
plotbool = False
user_based = False
use_logs = True
run_sweep = True

def cli():
    """ Handle argument parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.2,
                        help='learning_rate')
    parser.add_argument('--epoch_num', default=500,
                        help='epoch number')
    parser.add_argument('--batch_size', default=128,
                        help='batch_size')
    parser.add_argument('--hidden_num', default=500,
                        help='hidden layers number')
    parser.add_argument('--latent_dim', default=20,
                        help='MF latent dime')
    parser.add_argument('--model_name', default='AutoRec',
                        help='model name')

    args = parser.parse_args()
    return args

def trainAutoRec(args, train_list, val_list, n_user, n_item, user_based, device):
    trainset = sd.Dataset(train_list, n_user, n_item, user_based)
    if user_based:
        h = n_item
    else:
        h = n_user

    setmod = model.Model(hidden=[h, args.hidden_num],
                      learning_rate=args.learning_rate,
                      batch_size=args.batch_size,
                      device = device,
                      use_logs = use_logs)

    RMSE = setmod.run(trainset, val_list, num_epoch = args.epoch_num, plot = plotbool, use_logs = use_logs)


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

    if(use_logs):
            wandb.watch(model)

    model.fit(train_loader=train_loader, val_dataset=val_dataset, num_epoch = args.epoch_num, use_logs = use_logs)


if __name__ == '__main__':
    start = datetime.now()
    args = cli()
    if(use_logs):
        if(not run_sweep):
            config = {
                    "learning_rate": args.learning_rate,
                    "epochs": args.epoch_num,
                    "batch_size": args.batch_size,
                    "hidden_num": args.hidden_num,
                    "latent_dim": args.latent_dim,
                    "model_name": args.model_name,
                    }
            wandb.init(project="RecFinalProject", entity="tomerkoren", config=config)
        else:
            wandb.init(project="RecFinalProject", entity="tomerkoren")
            args = wandb.config

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('neural network device: %s (CUDA available: %s, count: %d)',
        device, torch.cuda.is_available(), torch.cuda.device_count())

    train_list, val_list, n_user, n_item = load_data('ratings', 0.9)

    if( args.model_name == 'AutoRec'):
        trainAutoRec(args, train_list, val_list, n_user, n_item, user_based, device)
    elif( args.model_name == 'BiasMF'):
        trainBiasMF(args, train_list, val_list, n_user, n_item, device)

    end = datetime.now()
    print ("Total time: %s" % str(end-start))

