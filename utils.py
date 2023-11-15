import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

class perceptionLoss():
    def __init__(self, device):
        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.eval()
        self.features = vgg.features.to(device)
        self.feature_layers = ['4', '9', '18', '27', '36']
        self.mse_loss = nn.MSELoss()

    def getfeatures(self, x):
        feature_list = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.feature_layers:
                feature_list.append(x)
        return feature_list