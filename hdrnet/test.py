import os
import sys
import cv2
import numpy as np
import skimage.exposure
import torch
from torchvision import transforms

from datetime import date
from model import HDRPointwiseNN
from datasets.dataset import HDRDataset
from datasets.SICE import SICE_Dataset
from utils import load_image, resize, load_params, eval, get_path_name, get_timecode
import matplotlib.pyplot as plt

def test(args):
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Using Device: {device}")

	# load model states and model params
	state_dict = torch.load(args['ckpt_path'])
	state_dict, model_params = load_params(state_dict)

	# dataset
	test_set = SICE_Dataset(args['data_dir'] + "Dataset_Part2/", under_expose_only=True, resize=(900, 1200))
	
	# model init
	model = HDRPointwiseNN(params=model_params)
	model.load_state_dict(state_dict)
	model.to(device)

	# evaluation
	psnr, ssim, mae = eval(model, test_set, device, out_dir=f"./visualization/{get_path_name(args['ckpt_path'])}-visualization/")
	print(f"Testing model: {args['ckpt_path']} \nAverage PSNR = {psnr:.3f} | SSIM = {ssim:.3f} | MAE = {mae:.3f}")

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='HDRNet Inference')
	parser.add_argument('--ckpt_path', type=str, default=f'./checkpoints/hdrnet_{get_timecode()}/', help='model state path')
	parser.add_argument('--data_dir', type=str, default="/home/ppnk-wsl/capstone/Dataset/", help='dataset directory')

	args = vars(parser.parse_args())

	test(args)