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
from datetime import date

from datasets.dataset import HDRDataset
from datasets.SICE import SICE_Dataset
from metrics import psnr
from model import HDRPointwiseNN, L2LOSS
from utils import load_image, save_params, get_latest_ckpt, load_params, eval, get_path_name, get_timecode


def train(args=None):
	os.makedirs(args['ckpt_dir'], exist_ok=True)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Using Device: {device}")

	train_dir = os.path.join(args['data_dir'], "Dataset_Part1/")
	test_dir = os.path.join(args['data_dir'], "Dataset_Part2/")

	train_dataset = SICE_Dataset(train_dir)
	train_dataset, val_dataset = random_split(train_dataset, [0.8015, 0.1985])
	test_dataset = SICE_Dataset(test_dir, under_expose_only=True, resize=(900, 1200))
	train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

	# model initiation
	model = HDRPointwiseNN(params=args)
	if args['resume_from']:
		resume_from_path = args['resume_from']
		print('Loading previous state:', resume_from_path)
		state_dict = torch.load(resume_from_path)
		state_dict,_ = load_params(state_dict)
		model.load_state_dict(state_dict)
	model.to(device)

	# loss function
	mseloss = torch.nn.SmoothL1Loss()#L2LOSS()#torch.nn.MSELoss()#torch.nn.SmoothL1Loss()#

	# optimizer
	optimizer = Adam(model.parameters(), args['lr'], eps=1e-7)#, weight_decay=1e-5)
	# optimizer = SGD(model.parameters(), params['lr'], momentum=0.9)
	# optimizer = adabound.AdaBound(model.parameters(), lr=params['lr'], final_lr=0.1)

	# training
	count = 0
	for e in range(args['epochs']):
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
		save_filename = "ckpt_" + str(e) + ".pth"
		save_path = os.path.join(args['ckpt_dir'], save_filename)
		state = save_params(model.state_dict(), args)
		torch.save(state, save_path)

		# run testset every epoch
		cur_psnr, cur_ssim, cur_mae = eval(model, test_dataset, device, out_dir=f"./visualization/{get_path_name(args['ckpt_dir'])}/epoch-{e}-visualization/")
		print(f"Testing model: {e+1}/{args['epochs']} in {args['ckpt_dir']} \nAverage PSNR = {cur_psnr} | SSIM = {cur_ssim} | MAE = {cur_mae}")

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='HDRNet Inference')
	parser.add_argument('--ckpt_dir', type=str, default=f'./checkpoints/hdrnet_{get_timecode()}/', help='Save checkpoints to this dir')
	parser.add_argument('--data_dir', type=str, default="/home/ppnk-wsl/capstone/Dataset/")
	parser.add_argument('--resume_from', type=str, default='', help='resume from this checkpoint')

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

	args = vars(parser.parse_args())

	print('PARAMS:')
	print(args)

	train(args=args)
