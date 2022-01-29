#main.py
#Modified by ImKe on 2019/9/5.
#Copyright © 2019 ImKe. All rights reserved.

import setdata as sd
import model as model
from datetime import datetime
from dataloader import load_data
import argparse
import wandb


display_step = 10
plotbool = False
user_based = False
global use_logs
use_logs = False

def cli():
    """ Handle argument parsing
    """
    parser = argparse.ArgumentParser(
        prog='python3 -m',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--learning_rate', default=0.2,
                        help='learning_rate')
    parser.add_argument('--epoch_num', default=500,
                        help='epoch number')
    parser.add_argument('--batch_size', default=128,
                        help='batch_size')
    parser.add_argument('--hidden_num', default=500,
                        help='hidden layers number')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start = datetime.now()
    args = cli()
    if(use_logs):
        config = {
                "learning_rate": args.learning_rate,
                "epochs": args.epoch_num,
                "batch_size": args.batch_size,
                "hidden_num": args.hidden_num,
                }
        wandb.init(project="RecFinalProject", entity="tomerkoren", config=config)
    train_list, test_list, n_user, n_item = load_data('ratings', 0.9)
    trainset = sd.Dataset(train_list, n_user, n_item, user_based)
    if user_based:
        h = n_item
    else:
        h = n_user

    setmod = model.Model(hidden=[h, args.hidden_num],
                      learning_rate=args.learning_rate,
                      batch_size=args.batch_size)

    RMSE = setmod.run(trainset, test_list, num_epoch = args.epoch_num, display_step = display_step, plot = plotbool)

    end = datetime.now()
    print ("Total time: %s" % str(end-start))


