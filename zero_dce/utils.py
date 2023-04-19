import numpy as np
import torch
import time
import cv2
import os
import random
import skimage.exposure

from datetime import datetime
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanAbsoluteError

def display_PILs(*args):
	"""
	display multiple PIL images using hstack
	"""
	Image.fromarray(np.hstack(args)).show()

def display_tensors(*args):
	"""
	display multiple torch tensors using hstack
	"""
	ToPIL = transforms.ToPILImage()
	imgs = []
	for arg in args:
		imgs.append(ToPIL(arg))

	imgs = tuple(imgs)
	return Image.fromarray(np.hstack(imgs)).show()

def hstack_tensors(*args):
	"""
	hstack multiple torch CxHxW tensors into CxHxnW
	"""
	imgs = []
	for arg in args:
		imgs.append(arg.permute(1, 2, 0))

	imgs = tuple(imgs)
	imgs = torch.hstack(imgs)
	return imgs.permute(2, 0, 1)

def save_tensor(tensor, dir):
	tensor = (tensor.cpu().detach().numpy()).transpose(1, 2, 0)
	tensor = skimage.exposure.rescale_intensity(tensor, out_range=(0.0,255.0)).astype(np.uint8)
	cv2.imwrite(dir, tensor[...,::-1])

def set_seed(seed=46):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

def check_and_rotate(img1, img2):
	"""
	rotate the first image (PIL Image) if two images have unmatched orientations
		e.g. img1 is in landscape orientation, img2 is in portrait orientation
		rotate img1 to have same orientation to img2
	"""
	if img1.size != img2.size:
		return img1.rotate(-90, expand=True)
	return img1

def get_path_name(path):
    return os.path.basename(os.path.normpath(path))

def get_timecode():
      return datetime.now().strftime("%y%m%d_%H%M")

def eval(model, dataset, device, tb_writer=None, out_dir=None):
	"""
	given a model, evaluate the model performance (PSNR, SSIM) on the dataset
	return:
		psnr: average PSNR score across the dataset
		ssim: average SSIM score across the dataset
	"""
	model.eval()

	set_seed(1143)
	dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)

	psnr_list = []
	ssim_list = []
	mae_list = []

	# visualize batch # on tensorboard is tensorboard provided
	visualize_idx = [0,1,5,10,20,50,100,300,500]
	visualize_imgs = []

	psnr = PeakSignalNoiseRatio(reduction=None)
	ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
	mae = MeanAbsoluteError() # different from MAE in Zero-DCE++ paper

	# time measure
	total_time = 0

	with torch.no_grad():
		for i, (img, ref) in enumerate(tqdm(dataloader)):
			img, ref = img.to(device), ref.to(device)

			start = time.time()
			enhanced, A = model(img)
			total_time += time.time() - start
			
			psnr_list.append(psnr(enhanced.cpu(), ref.cpu()))
			ssim_list.append(ssim(enhanced.cpu(), ref.cpu()))
			mae_list.append(mae(enhanced.cpu(), ref.cpu()))

			if tb_writer and i in visualize_idx:
				visualize_imgs.append(hstack_tensors(img[0], enhanced[0], ref[0]))
				# display_tensors(img[0], enhanced[0], ref[0])
			
			if out_dir and i in visualize_idx:
				out_dir_i = os.path.join(out_dir, str(i), '')
				if not os.path.exists(out_dir_i):
					os.makedirs(out_dir_i)
				# save input, enhanced and ref image to disk
				save_tensor(img[0], os.path.join(out_dir_i, "input.jpg"))
				save_tensor(enhanced[0], os.path.join(out_dir_i, "output.jpg"))
				save_tensor(ref[0], os.path.join(out_dir_i, "ref.jpg"))
			
			# if i == 2:
			# 	display_tensors(img[0], enhanced[0], ref[0])
			# 	while True: pass
	
	if tb_writer:
		tb_writer.add_image("Testset Visualization: Input | Enhanced | Ref", make_grid(visualize_imgs, nrow=1))


	# report time
	print(f"Total time used = {total_time} s inferencing {len(dataset)} images on [{device}] ({len(dataset) / total_time} FPS)")

	return np.mean(psnr_list), np.mean(ssim_list), np.mean(mae_list)