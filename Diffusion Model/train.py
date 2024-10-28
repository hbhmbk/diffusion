import torch
from torch import nn
from dataset import MyDatasset
from loss_func import loss_fun,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,time_steps
from utils import *
from torchvision import transforms
from tqdm import *
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR
import matplotlib.pyplot as plt
from utils.poly_lr_decay import PolynomialLRDecay
# root='E:\\vqvae\diffusion\\anime-faces'
root="/root/autodl-tmp/anime-faces"

import os
import json
import pickle
import random


def get_img(datapath):
    images = [os.path.join(datapath, i) for i in os.listdir(datapath)
              ]
    return images
imgs=get_img(root)
device='cuda'
batch_size=128
from u_net import *


class AverageMeter:
    """
    Compute and stores the average and current value
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




# print(model)
# model=Unet(24,with_time_emb=True,use_convnext=False).to('cuda')
# print(model)
model=torch.load("checkpoint/state_model epoch 495.pt").to('cuda')

transform=transforms.Compose([transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                             ])
dataset=MyDatasset(imgs,transform=transform)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)

optimizer=torch.optim.AdamW(model.parameters(),lr=2.5e-4,weight_decay=0.1)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
scheduler=PolynomialLRDecay(optimizer,
                                  max_decay_steps=2000,
                                  end_learning_rate=1e-6,
                                  power=1.0)


for epoch in range(496,1000):
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    loss_meter = AverageMeter()
    print('start training {}/{}'.format(epoch, 2000))
    for idx ,x in loop:


        x=x.to(device)
        optimizer.zero_grad()
        loss=loss_fun(model,device,x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,time_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        loss_meter.update(loss.item(), x.size(0))

        # print(loss_meter.val,loss_meter.avg)
        loop.set_postfix(loss=loss.item(),mean_loss=loss_meter.avg)
    scheduler.step()

        #epoch_loss+=loss.item()
    #

    if epoch%5==0:
        torch.save(model,"checkpoint/state_model epoch {}.pt".format(epoch))
    #     print(loss)

























