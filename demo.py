from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model.networks import Generator
import numpy as np
import os
from dataset import MUG_test
from torchvision import transforms
import argparse
import cv2


def save_videos(path, vids, n, cat):
	"""Save video using OpenCV as basic method"""
	for i in range(n):
		try:
			# Convert the tensor to a numpy array and change the order of the channels
			vid = vids[i].permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
			vid = (vid * 255).clip(0, 255).astype(np.uint8)

			os.makedirs(path, exist_ok=True)

			# Save via OpenCV
			output_path = os.path.join(path, f"{i}_{cat}.mp4")
			h, w = vid.shape[1:3]
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			out = cv2.VideoWriter(output_path, fourcc, 24.0, (w, h))

			for frame in vid:
				out.write(frame[..., ::-1])  # Convert RGB -> BGR
			out.release()

		except Exception as e:
			print(f"Error saving video {i}: {str(e)}")
			continue

	return


def main(args):

	# write into tensorboard
	log_path = os.path.join('demos', args.dataset, 'log')
	vid_path = os.path.join('demos', args.dataset, 'vids')

	os.makedirs(log_path, exist_ok=True)
	os.makedirs(vid_path, exist_ok=True)
	writer = SummaryWriter(log_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Model init
	G = Generator(args.dim_z, args.dim_a, args.nclasses, args.ch).to(device)
	G = nn.DataParallel(G)

	try:
		G.load_state_dict(torch.load(args.model_path, map_location=device))
		print("Model loaded successfully")
	except Exception as e:
		print(f"Error loading model: {str(e)}")
		return

	# Transforms
	transform = transforms.Compose([
		transforms.Resize((args.img_size, args.img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	# Load data
	try:
		dataset = MUG_test(args.data_path, transform=transform)
		dataloader = torch.utils.data.DataLoader(
			dataset=dataset,
			batch_size=args.batch_size,
			num_workers=min(args.num_workers, 4),
			shuffle=False,
			pin_memory=True
		)
	except Exception as e:
		print(f"Error loading dataset: {str(e)}")
		return

	# Video generation
	with torch.no_grad():
		G.eval()

		try:
			img = next(iter(dataloader)).to(device)
			bs = img.size(0)
			nclasses = args.nclasses

			z = torch.randn(bs, args.dim_z).to(device)

			for i in range(nclasses):
				y = torch.zeros(bs, nclasses).to(device)
				y[:, i] = 1.0
				vid_gen = G(img, z, y)

				vid_gen = vid_gen.transpose(2, 1)
				vid_gen = ((vid_gen - vid_gen.min()) / (vid_gen.max() - vid_gen.min())).data

				writer.add_video(tag=f'vid_cat_{i}', vid_tensor=vid_gen)
				writer.flush()

				print(f'==> saving videos for category {i}')
				save_videos(vid_path, vid_gen, bs, i)

		except Exception as e:
			print(f"Error during generation: {str(e)}")


if __name__ == '__main__':

	parser = argparse.ArgumentParser('imaginator demo config')

	parser.add_argument('--dataset', type=str, default='mug')
	parser.add_argument('--data_path', type=str, default='./mug/data64')
	parser.add_argument('--img_size', type=int, default=64)
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--dim_z', type=int, default=512)
	parser.add_argument('--dim_a', type=int, default=100)
	parser.add_argument('--nclasses', type=int, default=6)
	parser.add_argument('--ch', type=int, default=64)
	parser.add_argument('--model_path', type=str, required=True)
	parser.add_argument('--random_seed', type=int, default=12345)

	args = parser.parse_args()

	# Setting up seed for reproducibility
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	main(args)
