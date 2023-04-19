import numpy as np
import cv2
import os
import glob
import torch
import random
import time
import skimage

from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanAbsoluteError
from torchvision.utils import make_grid

def resize(img, size=512, strict=False):
    short = min(img.shape[:2])
    scale = size/short
    if not strict:
        img = cv2.resize(img, (round(
            img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_NEAREST)
    return img


def crop(img, size=512):
    try:
        y, x = random.randint(
            0, img.shape[0]-size), random.randint(0, img.shape[1]-size)
    except Exception as e:
        y, x = 0, 0
    return img[y:y+size, x:x+size, :]


def load_image(filename, size=None, use_crop=False):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = resize(img, size=size)
    if use_crop:
        img = crop(img, size)
    return img

def get_latest_ckpt(path):
    try:
        list_of_files = glob.glob(os.path.join(path,'*')) 
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except ValueError:
        return None

def save_params(state, params):
    state['model_params'] = params
    return state

def load_params(state):
    params = state['model_params']
    del state['model_params']
    return state, params


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

def eval(model, dataset, device, out_dir=None):
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

	# visualize batch number if on this list
	visualize_idx = [0,1,5,10,20,50,100,300,500]

	psnr = PeakSignalNoiseRatio(reduction=None)
	ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
	mae = MeanAbsoluteError() # different from MAE in Zero-DCE++ paper

	# time measure
	total_time = 0

	with torch.no_grad():
		for i, (img_low, img_full, ref) in enumerate(tqdm(dataloader)):
			img_low, img_full, ref = img_low.to(device), img_full.to(device), ref.to(device)

			start = time.time()
			enhanced = model(img_low, img_full)
			total_time += time.time() - start
			
			psnr_list.append(psnr(enhanced.cpu(), ref.cpu()))
			ssim_list.append(ssim(enhanced.cpu(), ref.cpu()))
			mae_list.append(mae(enhanced.cpu(), ref.cpu()))
			
			if out_dir and i in visualize_idx:
				out_dir_i = os.path.join(out_dir, str(i), '')
				if not os.path.exists(out_dir_i):
					os.makedirs(out_dir_i)
				# save input, enhanced and ref image to disk
				save_tensor(img_full[0], os.path.join(out_dir_i, "input.jpg"))
				save_tensor(enhanced[0], os.path.join(out_dir_i, "output.jpg"))
				save_tensor(ref[0], os.path.join(out_dir_i, "ref.jpg"))
			
			# if i == 2:
			# 	display_tensors(img[0], enhanced[0], ref[0])
			# 	while True: pass


	# report time
	print(f"Total time used = {total_time:.3f} s inferencing {len(dataset)} images on [{device}] ({len(dataset) / total_time:.3f} FPS)")

	return np.mean(psnr_list), np.mean(ssim_list), np.mean(mae_list)