import numpy as np
import torch
import sys
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import random
from natsort import natsorted, ns
from utils import display_tensors, check_and_rotate


class SICE_Dataset(Dataset):
	def __init__(self, image_dir, under_expose_only=False, resize=(512, 512)):
		# test set uses under exposed images only
		self.data_list = self.extract_image_pairs(image_dir, under_expose_only=under_expose_only)
		self.resize = resize
		print(f"Total dataset samples = {len(self.data_list)}")

		ls = (256, 256)
		fs = resize
		
		self.low = transforms.Compose([
			transforms.Resize(ls, Image.BICUBIC),
			transforms.ToTensor()
		])
		self.correction = transforms.Compose([
			transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0),
		])
		self.out = transforms.Compose([
			transforms.Resize(fs, Image.BICUBIC),
			transforms.ToTensor()
		])
		self.full = transforms.Compose([
			transforms.Resize(fs, Image.BICUBIC),
			transforms.ToTensor()
		])

		
	def __getitem__(self, index):
		image_path, label_path = self.data_list[index]
		image = Image.open(image_path)
		label = Image.open(label_path)

		# check and rotate image if orientation unmatched
		image = check_and_rotate(image, label)
		
		# if self.resize:
		#     image = image.resize(self.resize, Image.ANTIALIAS)
		#     label = label.resize(self.resize, Image.ANTIALIAS)

		image_low = self.low(image)
		image_full = self.full(image)
		image_out = self.out(label)

		# image = (np.asarray(image)/255.0) 
		# image = torch.from_numpy(image).float()
		
		# label = (np.asarray(label)/255.0) 
		# label = torch.from_numpy(label).float()

		return image_low, image_full, image_out

	def __len__(self):
		return len(self.data_list)

	def extract_image_pairs(self, dataset_dir, under_expose_only=False):
		"""
		extract images paired with corresponding reference images (under Label/)
		- under_expose_only: if enabled, only select under-exposed images for each scene,
			e.g. if there are 7 images, select only the first 3 images;
				if there are 9 images, select only the first 4 images.
		return:
			list of tuple paths,
				e.g. [(some-dir/1/1.jpg, some-dir/label/1.jpg), (some-dir/1/2.jpg, some-dir/label/1.jpg), ...]
		"""
		label_list = glob.glob(dataset_dir + "Label/**.**", recursive=True)
		label_list = natsorted(label_list)
		data_list = []
		for i, label_path in enumerate(label_list):
			jpgs = glob.glob(dataset_dir + str(i+1) + "/**.**")
			jpgs = natsorted(jpgs)
			if under_expose_only:
				# remove over-exposed images
				jpgs = jpgs[:len(jpgs)//2]
			# add (image, reference) pair to data list
			for jpg in jpgs:
				data_list.append((jpg, label_path))
		
		return data_list
	

if __name__ == "__main__":
	print("Sanity check train set:")
	train_dir = "/home/ppnk/ucla_capstone/Dataset/Dataset_Part1/"
	test_dir = "/home/ppnk/ucla_capstone/Dataset/Dataset_Part2/"

	train_set = SICE_Dataset(train_dir)
	test_set = SICE_Dataset(test_dir, under_expose_only=True)

	print(len(train_set))
	print(len(test_set))

	sys.path.insert(0, "..")
	from utils import display_tensors

	# display_tensors(*test_set[1], *test_set[2])

	

	# for image, label in extract_image_pairs(test_dir, under_expose_only=True):
	# 	print(image)
	# 	print(label)
	# 	print()

