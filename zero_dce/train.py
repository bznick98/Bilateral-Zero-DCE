import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import model
import random
import loss as MyLoss
import numpy as np
from datetime import date
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from datasets.SICE import SICE_Dataset
from utils import display_tensors, eval
# from loss import L_color, L_exp, L_spa, L_TV, Sa_Loss, perception_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Using Device: {device}")
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = config.scale_factor

	# visualize training process on tensorboard
	config_str = f"epochs = {config.num_epochs} | lr={config.lr} | batch_size={config.train_batch_size} | train from start = {not config.load_pretrain} | saved to {config.snapshots_folder}"
	tb = SummaryWriter(comment=config_str)
	
    # init model
	DCE_net = model.enhance_net_nopool(scale_factor).to(device)

	# load pre-trained weight if specified
	if config.load_pretrain:
		DCE_net.load_state_dict(torch.load(config.pretrain_dir))
		print(f"Pre-trained weight loaded from {config.pretrain_dir}")
	
	# load dataset
	train_dir = config.data_dir + "Dataset_Part1/"
	test_dir = config.data_dir + "Dataset_Part2/"
	train_set = SICE_Dataset(train_dir)
	train_set, val_set = random_split(train_set, [0.8015, 0.1985])
	test_set = SICE_Dataset(test_dir, under_expose_only=True, resize=(1200, 900))

	train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	# loss function
	L_color = MyLoss.L_color()
	L_spa = MyLoss.L_spa()
	L_exp = MyLoss.L_exp(16)
	# L_exp = MyLoss.L_exp(16,0.6)
	L_TV = MyLoss.L_TV()

	# optimizer
	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	# start training
	for epoch in range(config.num_epochs):
		DCE_net.train()
		with tqdm(train_loader, unit="batch") as tepoch:
			for batch in tepoch:

				img, ref = batch
				img, ref = img.to(device), ref.to(device)

				E = 0.6

				enhanced_image, A = DCE_net(img)

				Loss_TV = 1600*L_TV(A)
				# Loss_TV = 200*L_TV(A)			
				loss_spa = torch.mean(L_spa(enhanced_image, img))
				loss_col = 5*torch.mean(L_color(enhanced_image))
				loss_exp = 10*torch.mean(L_exp(enhanced_image,E))
				
				# best_loss
				loss =  Loss_TV + loss_spa + loss_col + loss_exp
				
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
				optimizer.step()

				tepoch.set_postfix(epoch="{}/{}".format(epoch+1, config.num_epochs), loss=loss.cpu().detach().numpy())

		# validate
		psnr, ssim, mae = eval(DCE_net, test_set, device, tb_writer=tb, out_dir=f"./result/{date.today()}/epoch-{epoch}-viz/")
		print(f"Epoch {epoch}'s PSNR = {psnr} | SSIM = {ssim} | MAE = {mae}")

		# add info on tensorboard
		tb.add_scalar("Loss", loss.cpu().detach().numpy(), epoch)
		tb.add_scalar("PSNR", psnr, epoch)
		tb.add_scalar("SSIM", ssim, epoch)
		tb.add_scalar("MAE", mae, epoch)

		# save weight each epoch
		torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

	tb.close()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--data_dir', type=str, default="/home/ppnk-wsl/capstone/Dataset/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--scale_factor', type=int, default=1)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots_Zero_DCE++/")
	parser.add_argument('--load_pretrain', type=bool, default=False) # TODO: merge this flag with directory
	parser.add_argument('--pretrain_dir', type=str, default="snapshots_Zero_DCE++/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	train(config)








	
