
import datasets as ds
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


user_based = False

def cli():
    """ Handle argument parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.2,
                        help='learning_rate')
    parser.add_argument('--epoch_num', default=400,
                        help='epoch number')
    parser.add_argument('--batch_size', default=128,
                        help='batch_size')
    parser.add_argument('--hidden_num', default='[512]',
                        help='hidden layers number')
    parser.add_argument('--latent_dim', default=20,
                        help='MF latent dime')
    parser.add_argument('--model_name', default='AutoRec',
                        help='model name')
    parser.add_argument('--act_type', default='relu',
                        help='activation type')
    parser.add_argument('--dropout', default=0.1,
                        help='dropout value')
    parser.add_argument('--momentum', default=0.9,
                        help='momentum value')
    parser.add_argument('--weight_decay', default=1e-4,
                        help='weight decay value')
    parser.add_argument('--dataset', default='ml-1m',
                        help='dataset type')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')

    args = parser.parse_args()

    return args

def trainOurAutoRec(args, train_list, val_list, n_user, n_item, user_based, device):
    trainset = ds.Dataset(train_list, n_user, n_item, user_based)
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
                      use_wandb = args.use_wandb)

    RMSE = setmod.run(trainset, val_list, num_epoch = args.epoch_num, use_wandb = args.use_wandb)

def trainAutoRec(args, train_list, val_list, n_user, n_item, user_based, device):
    trainset = ds.Dataset(train_list, n_user, n_item, user_based)
    if user_based:
        h = n_item
    else:
        h = n_user

    hidden_num = json.loads(userArgs.hidden_num)

    setmod = model.Model(hidden=[h, *hidden_num],
                      learning_rate=args.learning_rate,
                      batch_size=args.batch_size,
                      dropout= args.dropout,
                      device = device,
                      use_wandb = args.use_wandb)

    RMSE = setmod.run(trainset, val_list, num_epoch = args.epoch_num, use_wandb = args.use_wandb)


def trainBiasMF(args, train_list, val_list, n_user, n_item, device):

    # Train
    user_tensor = torch.LongTensor([val[0] for val in train_list]).to(device = device)
    item_tensor = torch.LongTensor([val[1] for val in train_list]).to(device = device)
    rating_tensor = torch.FloatTensor([val[2] for val in train_list]).to(device = device)
    dataset = ds.RateDataset(user_tensor, item_tensor, rating_tensor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Test
    user_tensor = torch.LongTensor([val[0] for val in val_list]).to(device = device)
    item_tensor = torch.LongTensor([val[1] for val in val_list]).to(device = device)
    rating_tensor = torch.FloatTensor([val[2] for val in val_list]).to(device = device)
    val_dataset = ds.RateDataset(user_tensor, item_tensor, rating_tensor)

    params = {'num_users': n_user, 
            'num_items': n_item,
            'global_mean': 3, 
            'latent_dim': args.latent_dim,
            'device': device,
            'lr' : args.learning_rate
            }

    model = BiasMF(params)
    model = model.to(device=device)

    if(args.use_wandb):
            wandb.watch(model)

    model.fit(train_loader=train_loader, val_dataset=val_dataset, num_epoch = args.epoch_num, use_wandb = args.use_wandb)


if __name__ == '__main__':
    start = datetime.now()
    userArgs = cli()
    if(userArgs.use_wandb):
            wandb.init(project="EnterYourWandbProjectName", entity="EnterYourWandbUserName", config=vars(userArgs))
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

