import argparse
import json
import os
from tqdm import tqdm
import random as rn
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from models import emo_disc
from datagen_aug import Dataset

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--in-path", type=str, help="Input folder containing train data", default=None, required=True)
    # parser.add_argument("-v", "--val-path", type=str, help="Input folder containing validation data", default=None, required=True)
    parser.add_argument("-o", "--out-path", type=str, help="output folder", default='../models/def', required=True)

    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument('--lr_emo', type=float, default=1e-06)

    parser.add_argument("--gpu-no", type=str, help="select gpu", default='1')
    parser.add_argument('--seed', type=int, default=9)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    args.batch_size = args.batch_size * max(int(torch.cuda.device_count()), 1)
    args.steplr = 200

    args.filters = [64, 128, 256, 512, 512]
    #-----------------------------------------#
    #           Reproducible results          #
    #-----------------------------------------#
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    rn.seed(args.seed)
    torch.manual_seed(args.seed)
    #-----------------------------------------#
   
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    else:
        shutil.rmtree(args.out_path)
        os.mkdir(args.out_path)

    with open(os.path.join(args.out_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args.cuda = torch.cuda.is_available() 
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu") 
    args.kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    return args

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

def enableGrad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)