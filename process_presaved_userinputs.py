
import numpy as np
import cv2
import scipy.misc
from skimage import color
import torch
from IPython import embed
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

import model_rec as model
import model_rec_noupreshape as model2

# ***** CONSTANTS *****
l_norm = 100.
l_cent = 50.
ab_norm = 110.
mask_cent = 0.
use_gpu = False

def np2tens(in_np,use_gpu=True):
	# numpy HxWxC ==> Torch tensor 1xCxHxW
	out_tens = torch.Tensor(in_np.transpose((2,0,1)))[None,:,:,:]
	return out_tens.cuda() if(use_gpu) else out_tens.cpu()

def tens2np(in_tens,use_gpu=True):
	# Torch tensor 1xCxHxW ==> numpy HxWxC
	return in_tens.cpu().numpy().transpose((2,3,1,0))[:,:,:,0] if(use_gpu) else in_tens.numpy().transpose((2,3,1,0))[:,:,:,0]

# ***** LOAD MODEL *****
colorizer = model.SIGGRAPHGenerator()
colorizer.load_state_dict(torch.load('./models/caffemodel_mask01_rec.pth'))
colorizer.cuda() if(use_gpu) else colorizer.cpu()
colorizer.eval()

input_dirs = ['cup_with_dist_190417_013901',
	'cup_with_dist_190417_013831',
	'cup_with_dist_190417_013901',
	'image013_with_dist_190417_015719',
	'image013_with_dist_190417_020208',
	'image015_with_dist_190417_020225',
	'image015_with_dist_190417_020612',
	'image017_with_dist_190417_020632',
	'image023_with_dist_190417_020723',
	'image023_with_dist_190417_020853',]

for input_dir in input_dirs:
	# ***** LOAD INPUTS *****
	in_l = np.load('./imgs/%s/im_l.npy'%input_dir).transpose((1,2,0)) # [0, 100] of shape 256x256x1
	in_ab = np.load('./imgs/%s/im_ab.npy'%input_dir).transpose((1,2,0)) # [-110, +110] of shape 256x256x2
	in_mask = 1.*np.load('./imgs/%s/im_mask.npy'%input_dir).transpose((1,2,0)) # [0, 1] of shape 256x256x2

	# normalize & center input ab, input mask
	img_rs_l_norm = (in_l-l_cent)/l_norm # [-.5, .5]
	in_ab_norm = in_ab/ab_norm # [-1, 1]
	in_mask_norm = in_mask - mask_cent # [0, 1]

	# ***** RUN MODEL *****
	out_class, out_reg = colorizer.forward(np2tens(img_rs_l_norm,use_gpu=use_gpu),
										   np2tens(in_ab_norm,use_gpu=use_gpu),
										   np2tens(in_mask_norm,use_gpu=use_gpu))
	out_reg = out_reg*ab_norm

	out_ab = out_reg[0,...].data.numpy().transpose((1,2,0))
	np.save('./imgs/%s/out_ab.npy'%input_dir, out_ab)

	# ***** GENERATE IMAGE *****
	out_lab = np.concatenate((in_l, out_ab), axis=2)
	out_rgb = np.clip(255*color.lab2rgb(out_lab), 0, 255).astype('uint8')
	cv2.imwrite('./imgs/%s/out.png'%input_dir, out_rgb[:,:,::-1])
