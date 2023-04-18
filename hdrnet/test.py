import os
import sys
import cv2
import numpy as np
import skimage.exposure
import torch
from torchvision import transforms

from model import HDRPointwiseNN
from datasets.dataset import HDRDataset
from datasets.SICE import SICE_Dataset
from utils import load_image, resize, load_params, eval
import matplotlib.pyplot as plt

def test(ckpt, args={}):
	state_dict = torch.load(ckpt)
	state_dict, params = load_params(state_dict)
	params.update(args)

	device = torch.device("cuda")
	tensor = transforms.Compose([
		transforms.ToTensor(),
	])
	# low = tensor(resize(load_image(params['test_image']),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
	# full = tensor(load_image(params['test_image']).astype(np.float32)).repeat(1,1,1,1)/255

	test_set = SICE_Dataset(image_dir=params['dataset'] + "Dataset_Part2/", under_expose_only=True, resize=(900, 1200))
	
	model = HDRPointwiseNN(params=params)
	model.load_state_dict(state_dict)

	psnr, ssim, mae = eval(model, test_set, device, tb_writer=0, out_dir=f"./ds/{ckpt.split('/')[-1].split('.')[0]}/")
	print(f"Testing model: {ckpt} \nAverage PSNR = {psnr} | SSIM = {ssim} | MAE = {mae}")

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='HDRNet Inference')
	parser.add_argument('--checkpoint', type=str, help='model state path')
	# parser.add_argument('--input', type=str, dest="test_image", help='image path')
	# parser.add_argument('--output', type=str, dest="test_out", help='output image path')

	args = vars(parser.parse_args())

	test(args['checkpoint'], args)