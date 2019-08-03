
import numpy as np
from skimage import color
import torch

def np2tens(in_np,use_gpu=True):
	# numpy HxWxC ==> Torch tensor 1xCxHxW
	out_tens = torch.Tensor(in_np.transpose((2,0,1)))[None,:,:,:]
	if(use_gpu):
		out_tens = out_tens.cuda()
	return out_tens

def tens2np(in_tens):
	# Torch tensor 1xCxHxW ==> numpy HxWxC
	return in_tens.cpu().numpy().transpose((2,3,1,0))[:,:,:,0]

def lab2rgb_clip(in_lab):
	return np.clip(color.lab2rgb(in_lab),0,1)


def add_color_patch(in_ab,in_mask,ab=[0,0],hw=[128,128],P=5):
	# add a color patch
	in_ab[hw[0]:hw[0]+P,hw[1]:hw[1]+P,0] = ab[0]
	in_ab[hw[0]:hw[0]+P,hw[1]:hw[1]+P,1] = ab[1]
	in_mask[hw[0]:hw[0]+P,hw[1]:hw[1]+P,:] = 1.
	return (in_ab,in_mask)

