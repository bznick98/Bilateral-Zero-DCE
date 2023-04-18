import os
import sys
from test import test

import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch.optim import SGD, Adam, RAdam, RMSprop
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import adabound
from datetime import date

from datasets.dataset import HDRDataset
from datasets.SICE import SICE_Dataset
from metrics import psnr
from model import HDRPointwiseNN, L2LOSS
from utils import load_image, save_params, get_latest_ckpt, load_params, eval


def train(params=None):
    os.makedirs(params['ckpt_path'], exist_ok=True)

    device = torch.device("cuda")

    # train_dataset = HDRDataset(params['dataset'], params=params, suffix=params['dataset_suffix'])
    # train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    train_dataset = SICE_Dataset(image_dir=params['dataset'] + "Dataset_Part1/")
    train_dataset, val_dataset = random_split(train_dataset, [0.8015, 0.1985])
    test_dataset = SICE_Dataset(image_dir=params['dataset'] + "Dataset_Part2/", under_expose_only=True, resize=(900, 1200))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    model = HDRPointwiseNN(params=params)
    ckpt = get_latest_ckpt(params['ckpt_path'])
    if ckpt:
        print('Loading previous state:', ckpt)
        state_dict = torch.load(ckpt)
        state_dict,_ = load_params(state_dict)
        model.load_state_dict(state_dict)
    model.to(device)

    mseloss = torch.nn.SmoothL1Loss()#L2LOSS()#torch.nn.MSELoss()#torch.nn.SmoothL1Loss()#
    optimizer = Adam(model.parameters(), params['lr'], eps=1e-7)#, weight_decay=1e-5)
    # optimizer = SGD(model.parameters(), params['lr'], momentum=0.9)
    # optimizer = adabound.AdaBound(model.parameters(), lr=params['lr'], final_lr=0.1)

    count = 0
    for e in range(params['epochs']):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (low, full, target) in enumerate(tepoch):
                optimizer.zero_grad()

                low = low.to(device)
                full = full.to(device)
                t = target.to(device)
                res = model(low, full)
                
                total_loss = mseloss(t,res)
                total_loss.backward()

                # if (count+1) % params['log_interval'] == 0:
                #     _psnr = psnr(res,t).item()
                #     loss = total_loss.item()
                #     print(e, count, loss, _psnr)
                
                optimizer.step()

                count += 1
        
        # save model every epoch
        ckpt_model_filename = "ckpt_" + str(e) + ".pth"
        ckpt_model_path = os.path.join(params['ckpt_path'], ckpt_model_filename)
        state = save_params(model.state_dict(), params)
        torch.save(state, ckpt_model_path)

        # run testset each epoch
        cur_psnr, cur_ssim, cur_mae = eval(model, test_dataset, device, out_dir=f"./ds/{date.today()}/epoch-{e}-viz/")
        print(f"Testing model: {params['ckpt_path']} \nAverage PSNR = {cur_psnr} | SSIM = {cur_ssim} | MAE = {cur_mae}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--ckpt-path', type=str, default='./ch', help='Model checkpoint path')
    parser.add_argument('--test-image', type=str, dest="test_image", help='Test image path')
    parser.add_argument('--test-out', type=str, default='out.png', dest="test_out", help='Output test image path')

    parser.add_argument('--luma-bins', type=int, default=8)
    parser.add_argument('--channel-multiplier', default=1, type=int)
    parser.add_argument('--spatial-bin', type=int, default=16)
    parser.add_argument('--guide-complexity', type=int, default=16)
    parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
    parser.add_argument('--net-input-size', type=int, default=256, help='Size of low-res input')
    parser.add_argument('--net-output-size', type=int, default=512, help='Size of full-res input/output')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--ckpt-interval', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='/home/ppnk-wsl/capstone/Dataset/', help='Dataset path with input/output dirs')
    parser.add_argument('--dataset-suffix', type=str, default='', help='Add suffix to input/output dirs. Useful when train on different dataset image sizes')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)

    train(params=params)
