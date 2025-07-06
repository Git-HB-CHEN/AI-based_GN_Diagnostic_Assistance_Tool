#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chnn

by Pytorch
"""
import sys
sys.path.append('..')

import os
import torch
import copy
import timm
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from timm.data.mixup import Mixup
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.step_lr import StepLRScheduler

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS" #设备排序
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #设置第一块GPU可见
device = torch.device("cuda")
print(device)

path = 'Glomerulus.Category2/'
Tpth = os.path.join(path,'Train')
Vpth = os.path.join(path,'Valid')

os.makedirs('modelbase/models_cate2/', exist_ok=True)
batchsize = 96
maxepochs = 300
lr = 1e-4

writer = SummaryWriter('Log/models_cate2/')
datapath = {'train':Tpth,'valid':Vpth}

data_transforms = {
    'train':transforms.Compose([
        transforms.Resize((384,384)),
        transforms.RandomResizedCrop((384,384),scale=(0.85,1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.638, 0.533, 0.575], [0.262, 0.280, 0.276])
        ]),
    'valid':transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.638, 0.533, 0.575], [0.262, 0.280, 0.276])
        ])
}

image_datasets = {x:ImageFolder(datapath[x],data_transforms[x]) for x in ['train','valid']}

labeldict = dict(Counter(image_datasets['train'].targets))
labellist = torch.tensor([v for k,v in labeldict.items()])
weight = torch.max(labellist) / labellist.float()
samples_weight = np.array([weight[t] for t in image_datasets['train'].targets])
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

dataloaders ={
    'train':DataLoader(image_datasets['train'],
                       batch_size = batchsize,
                       sampler = sampler,
                       num_workers = 24,
                       pin_memory=True,
                       shuffle = False,
                       drop_last=True),
    'valid': DataLoader(image_datasets['valid'],
                       batch_size = batchsize,
                       num_workers = 24,
                       pin_memory=True,
                       shuffle = False)}

datasets_sizes = {x:len(image_datasets[x]) for x in ['train','valid']}
class_names = image_datasets['train'].class_to_idx

print('datasets_sizes:',datasets_sizes)
print('class_names:   ',class_names)

model = timm.create_model('swin_base_patch4_window12_384', pretrained=True)
model.head = nn.Sequential(nn.Linear(model.head.in_features, 224),
                            nn.ReLU(True),
                            nn.Linear(224, 28),
                            nn.ReLU(True),
                            nn.Linear(28,3))

nn.init.kaiming_normal_(model.head[0].weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(model.head[2].weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(model.head[4].weight, mode='fan_out', nonlinearity='relu')

model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=weight).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max = maxepochs//3, eta_min=1e-9
)

print("Optimizer: ", optimizer.__class__.__name__)

best_model_wts = copy.deepcopy(model.state_dict())
best_score_valid = 0.0

for epoch in range(1,maxepochs + 1):
    print('Epoch {}/{}'.format(epoch,maxepochs))
    print('-' * 60)
    running_loss = 0.0
    running_corrects = 0.0
    model.train()
    for imgs,lbls in tqdm(dataloaders['train']):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outs = model(imgs)
            _,preds = torch.max(outs,1)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == lbls)
    scheduler.step()
    epoch_loss = running_loss / datasets_sizes['train']
    epoch_acc  = running_corrects.double() /  datasets_sizes['train']
    print('[{}] >>> Loss: {:.4f} Acc: {:.4f}'.format('Train',epoch_loss,epoch_acc))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/acc', epoch_acc, epoch)
    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)

    model.eval()
    real_vd = []
    pred_vd = []
    for imgs,lbls in tqdm(dataloaders['valid']):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outs = model(imgs)
            _,preds = torch.max(outs,1)
        real_vd += lbls.tolist()
        pred_vd += preds.tolist()

    valid_acc  = accuracy_score(real_vd,pred_vd)
    valid_f1_score = f1_score(real_vd,pred_vd,pos_label=0)
    print('[{}] >>> f1_score:{:.4f} acc:{:.4f}'.format('Valid',valid_f1_score,valid_acc))

    writer.add_scalar('valid/acc',      valid_acc,      epoch)
    writer.add_scalar('valid/f1_score', valid_f1_score, epoch)


    if  valid_f1_score >= best_score_valid:
        best_score_valid = valid_f1_score
        torch.save(model, 'modelbase/models_cate2/best_model.pkl')

torch.save(model, 'modelbase/models_cate2/last_model.pkl')
writer.close()


