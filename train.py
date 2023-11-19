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

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model')
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


logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def freezeNet(network):
    for p in network.parameters():
        p.requires_grad = False

def unfreezeNet(network):
    for p in network.parameters():
        p.requires_grad = True

device = torch.device("cuda" if use_cuda else "cpu")
device_ids = list(range(torch.cuda.device_count()))

syncnet = SyncNet().to(device)
# syncnet = nn.DataParallel(syncnet, device_ids)
freezeNet(syncnet)

disc_emo = emo_disc.DISCEMO().to(device)
# disc_emo = nn.DataParallel(disc_emo, device_ids)
disc_emo.load_state_dict(torch.load(args.emotion_disc_path))
emo_loss_disc = nn.CrossEntropyLoss()

perceptual_loss = utils.perceptionLoss(device)
recon_loss = nn.L1Loss()

def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def train(device, model, train_data_loader, test_data_loader, optimizer, nepochs=None):
    print(f'num_batches:{len(train_data_loader)}')
    global global_step, global_epoch
    resumed_step = global_step
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        running_ploss, running_loss_de_c = 0., 0.
        running_loss_fake_c, running_loss_real_c = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt, emotion) in prog_bar:
            model.train()
            disc_emo.train()
            freezeNet(disc_emo)
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device) 
            emotion = emotion.to(device)
            

            #### training generator/Wav2lip model
            g = model(indiv_mels, x, emotion) 

            # emo_label is obtained from audio_encoding (not required for out model)

            emotion_ = emotion.unsqueeze(1).repeat(1, 5, 1)
            emotion_ = torch.cat([emotion_[:, i] for i in range(emotion_.size(1))], dim=0)

            de_c = disc_emo.forward(g)
            loss_de_c = emo_loss_disc(de_c, torch.argmax(emotion, dim=1))
            
            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            l1loss = recon_loss(g, gt)
            ploss =  perceptual_loss.calculatePerceptionLoss(g,gt)

            loss = hparams.syncnet_wt * sync_loss + hparams.pl_wt * ploss + hparams.emo_wt * loss_de_c  
            loss += (1 - hparams.syncnet_wt - hparams.emo_wt - hparams.pl_wt) * l1loss

            loss.backward()
            optimizer.step()

            unfreezeNet(disc_emo)

            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.
            running_l1_loss += l1loss.item()
            running_ploss += ploss.item()
            running_loss_de_c += loss_de_c.item()
            
            #### training emotion_disc model
            disc_emo.opt.zero_grad()
            g = g.detach()
            class_real = disc_emo(gt) # for ground-truth
        
            loss_real_c = emo_loss_disc(class_real, torch.argmax(emotion, dim=1))
            loss_real_c.backward()
            disc_emo.opt.step()

            running_loss_real_c += loss_real_c.item()

            global_step += 1
            cur_session_steps = global_step - resumed_step

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

                    if average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.03) # without image GAN a lesser weight is sufficient

            prog_bar.set_description('L1: {:.4f}, Ploss: {:.4f}, Sync Loss: {:.4f}, de_c_loss: {:.4f} | loss_real_c: {:.4f}'.format(running_l1_loss / (step + 1),
                                                                    running_ploss / (step + 1),
                                                                    running_sync_loss / (step + 1),
                                                                    running_loss_de_c / (step + 1),
                                                                    running_loss_real_c / (step + 1)))
        global_epoch += 1

if __name__ == "__main__":

    full_dataset = Dataset('train')
    train_size = int(0.95 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = Wav2Lip().to(device)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5,0.999))


    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer,
              nepochs=hparams.nepochs)