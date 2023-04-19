import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import random
import loss as MyLoss
import numpy as np
from datetime import date
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from loss import ZeroReferenceLoss
from model import enhance_net_nopool
from datasets.SICE import SICE_Dataset
from utils import display_tensors, eval, get_path_name, get_timecode
# from loss import L_color, L_exp, L_spa, L_TV, Sa_Loss, perception_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
	# save dir
	if not os.path.exists(config.ckpt_dir):
		os.makedirs(config.ckpt_dir)
	
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Using Device: {device}")

	# visualize training process on tensorboard
	config_str = f"epochs = {config.epochs} | lr={config.lr} | batch_size={config.train_batch_size} | train from start = {not config.resume_from} | saved to {config.data_dir}"
	tb = SummaryWriter(comment=config_str)
	
    # init model
	model = enhance_net_nopool(config.scale_factor).to(device)

	# load pre-trained weight if specified
	if config.resume_from:
		model.load_state_dict(torch.load(config.resume_from))
		print(f"Pre-trained weight loaded from {config.resume_from}")
	
	# load dataset
	train_dir = config.data_dir + "Dataset_Part1/"
	test_dir = config.data_dir + "Dataset_Part2/"
	train_set = SICE_Dataset(train_dir)
	train_set, val_set = random_split(train_set, [0.8015, 0.1985])
	test_set = SICE_Dataset(test_dir, under_expose_only=True, resize=(1200, 900))

	train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	# loss function
	ZeroRefLoss = ZeroReferenceLoss(patch_size=16, TV_loss_weight=1, E=0.6)

	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	# start training
	for e in range(config.epochs):
		model.train()
		with tqdm(train_loader, unit="batch") as tepoch:
			for batch in tepoch:

				img, ref = batch
				img, ref = img.to(device), ref.to(device)

				enhanced_image, A = model(img)

				loss = ZeroRefLoss(img, enhanced_image, A)

				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm(model.parameters(),config.grad_clip_norm)
				optimizer.step()

				tepoch.set_postfix(epoch="{}/{}".format(e+1, config.epochs), loss=loss.cpu().detach().numpy())

		# save weight each epoch
		save_filename = "ckpt_" + str(e) + '.pth'
		save_path = os.path.join(config.ckpt_dir, save_filename)
		torch.save(model.state_dict(), save_path)

		# validate
		psnr, ssim, mae = eval(model, test_set, device, out_dir=f"./visualization/{get_path_name(config.ckpt_dir)}/epoch-{e}-visualization/")
		print(f"Testing model: {e+1}/{config.epochs} in {config.ckpt_dir} \nAverage PSNR = {psnr:.3f} | SSIM = {ssim:.3f} | MAE = {mae:.3f}")

		# add info on tensorboard
		tb.add_scalar("Loss", loss.cpu().detach().numpy(), e)
		tb.add_scalar("PSNR", psnr, e)
		tb.add_scalar("SSIM", ssim, e)
		tb.add_scalar("MAE", mae, e)

	tb.close()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--data_dir', type=str, default="/home/ppnk-wsl/capstone/Dataset/")
	parser.add_argument('--ckpt_dir', type=str, default=f"./checkpoints/zerodce_{get_timecode()}/")
	parser.add_argument('--resume_from', type=str, default='', help='resume from this checkpoint')

	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--scale_factor', type=int, default=1)

	config = parser.parse_args()

	train(config)

