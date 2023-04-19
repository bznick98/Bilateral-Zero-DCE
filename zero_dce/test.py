import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import model
import loss as MyLoss
import numpy as np
from datetime import date
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from datasets.SICE import SICE_Dataset
from utils import display_tensors, eval, get_path_name, get_timecode


def test(config):
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Using Device: {device}")
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = config.scale_factor
	
    # init model
	model = model.enhance_net_nopool(scale_factor).to(device)

	# load pre-trained weight if specified
	if config.model_path:
		model.load_state_dict(torch.load(config.model_path))
		print(f"Pre-trained weight loaded from {config.model_path}")
	
	# load dataset
	test_dir = config.data_dir + "Dataset_Part2/"
	test_set = SICE_Dataset(test_dir, under_expose_only=True, resize=(1200, 900))

	# test
	psnr, ssim, mae = eval(model, test_set, device, out_dir=f"./visualization/{get_path_name(config.ckpt_dir)}-visualization/")
	print(f"Testing model: {config.ckpt_path} \nAverage PSNR = {psnr:.3f} | SSIM = {ssim:.3f} | MAE = {mae:.3f}")
	

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--data_dir', type=str, default="/home/ppnk-wsl/capstone/Dataset/")
	parser.add_argument('--ckpt_path', type=str, default=f"./checkpoints/zerodce_{get_timecode()}/")

	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--scale_factor', type=int, default=1)
	

	config = parser.parse_args()

	test(config)