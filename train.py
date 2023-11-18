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

class Dataset(object):
    def __init__(self, split, val=False):
        #self.all_videos = get_image_list(args.data_root, split)
        # self.all_videos = [join(args.data_root, f) for f in os.listdir(args.data_root) if isdir(join(args.data_root, f))]
        self.filelist = []
        self.all_videos = [f for f in os.listdir(args.data_root) if isdir(join(args.data_root, f))]

        for filename in self.all_videos:
            #print(splitext(filename))
            labels = splitext(filename)[0].split('_')
            emotion = emotion_dict[labels[2]]
            
            emotion_intensity = intensity_dict[labels[3]]
            if val:
                if emotion_intensity != 3:
                    continue
            
            self.filelist.append((filename, emotion, emotion_intensity))

        self.filelist = np.array(self.filelist)
        print('Num files: ', len(self.filelist))

        # to apply same augmentation for all the 10 frames (5 reference and 5 ground truth)
        target = {}
        for i in range(1, 2*emonet_T):
            target['image' + str(i)] = 'image'
        
        self.augments = A.Compose([
                        A.RandomBrightnessContrast(p=0.4),    
                        A.RandomGamma(p=0.4),    
                        A.CLAHE(p=0.4),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.4),  
                        A.ChannelShuffle(p=0.4), 
                        A.RGBShift(p=0.4),
                        A.RandomBrightness(p=0.4),
                        A.RandomContrast(p=0.4),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                    ], additional_targets=target, p=0.8)
    
    def augmentVideo(self, video):
        args = {}
        args['image'] = video[0, :, :, :]
        for i in range(1, 2*emonet_T):
            args['image' + str(i)] = video[i, :, :, :]
        result = self.augments(**args)
        video[0, :, :, :] = result['image']
        for i in range(1, 2*emonet_T):
            video[i, :, :, :] = result['image' + str(i)]
        return video

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.filelist) - 1)
            filename = self.filelist[idx]
            vidname = filename[0]
            emotion = int(filename[1])
            emotion = to_categorical(emotion, num_classes=6)

            img_names = list(glob(join(args.data_root, vidname, '*.jpg')))

            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(args.data_root, vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = np.asarray(window)
            y = window.copy()
            # window[:, :, window.shape[2]//2:] = 0.
            # we need to generate whole face as we are incorporating emotion
            window[:, :, :] = 0.

            wrong_window = np.asarray(wrong_window)
            conact_for_aug = np.concatenate([y, wrong_window], axis=0)

            aug_results = self.augmentVideo(conact_for_aug)
            y, wrong_window = np.split(aug_results, 2, axis=0)

            y = np.transpose(y, (3, 0, 1, 2)) / 255
            window = np.transpose(window, (3, 0, 1, 2))
            wrong_window = np.transpose(wrong_window, (3, 0, 1, 2)) / 255

            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y, emotion

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

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


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
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

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

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