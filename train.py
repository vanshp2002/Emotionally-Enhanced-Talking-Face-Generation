from gc import freeze
from os.path import dirname, join, basename, isfile, isdir, splitext
from tqdm.auto import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
from models import emo_disc
import audio

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
import albumentations as A
import utils
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)
parser.add_argument('--emotion_disc_path', help='Load the pre-trained emotion discriminator', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
args = parser.parse_args()


global_step = 0
global_epoch = 0
os.environ['CUDA_VISIBLE_DEVICES']='3'
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

def to_categorical(y, num_classes=None, dtype='float32'):

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

emotion_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
intensity_dict = {'XX':0, 'LO':1, 'MD':2, 'HI':3}
emonet_T = 5

