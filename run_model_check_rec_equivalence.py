
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
mask_cent = 0.5
use_gpu = False
H_proc, W_proc = (256,256) # resolution to process, needs to be multiple of 8

# ***** HELPER FUNCTIONS *****
def add_color_patch(in_ab,in_mask,ab=[0,0],hw=[128,128],P=5):
	# add a color patch
	in_ab[hw[0]:hw[0]+P,hw[1]:hw[1]+P,0] = ab[0]
	in_ab[hw[0]:hw[0]+P,hw[1]:hw[1]+P,1] = ab[1]
	in_mask[hw[0]:hw[0]+P,hw[1]:hw[1]+P,:] = 1.
	return (in_ab,in_mask)

def np2tens(in_np,use_gpu=True):
	# numpy HxWxC ==> Torch tensor 1xCxHxW
	out_tens = torch.Tensor(in_np.transpose((2,0,1)))[None,:,:,:]
	if(use_gpu):
		out_tens = out_tens.cuda()
	else:
		out_tens = out_tens.cpu()
	return out_tens

def tens2np(in_tens,use_gpu=True):
	# Torch tensor 1xCxHxW ==> numpy HxWxC
	if(use_gpu):
		return in_tens.cpu().numpy().transpose((2,3,1,0))[:,:,:,0]
	else:
		return in_tens.numpy().transpose((2,3,1,0))[:,:,:,0]

# ***** LOAD MODEL *****
colorizer = model.SIGGRAPHGenerator()
colorizer.load_state_dict(torch.load('./models/net_G_19_03_04_trained1ep.pth'))
colorizer2 = model2.SIGGRAPHGenerator()
colorizer2.load_state_dict(torch.load('./models/net_G_19_03_04_trained1ep.pth'))
if(use_gpu):
	colorizer.cuda()
	colorizer2.cuda()
else:
	colorizer.cpu()
	colorizer2.cpu()
colorizer.eval()
colorizer2.eval()

# ***** LOAD IMAGE, PREPARE DATA *****
img_orig = cv2.imread('./imgs/migrant_mother.jpg')[:,:,::-1]
(H_orig,W_orig) = img_orig.shape[:2]
print('[%ix%i] Original resolution'%(H_orig,W_orig))
print('[%ix%i] Processed resolution'%(H_proc,W_proc))

# take L channel at fullres
img_orig_lab = color.rgb2lab(img_orig)
img_orig_l = img_orig_lab[:,:,[0]]

# resize to processing size, take L channel for input
img_rs = cv2.resize(img_orig, (W_proc, H_proc), interpolation=cv2.INTER_CUBIC)
img_rs_lab = color.rgb2lab(img_rs)
img_rs_l_norm = (img_rs_lab[:,:,[0]]-l_cent)/l_norm # normalized


# ***** COLOR INPUT POINTS *****
# initialize blank ab input, mask
in_ab = np.zeros((H_proc, W_proc, 2))
in_mask = np.zeros((H_proc, W_proc, 1))

# normalize & center input ab, input mask
in_ab_norm = in_ab/ab_norm
in_mask_norm = in_mask - mask_cent

# ***** RUN MODEL *****
out_class, out_reg = colorizer.forward(np2tens(img_rs_l_norm,use_gpu=use_gpu), np2tens(in_ab_norm,use_gpu=use_gpu), np2tens(in_mask_norm,use_gpu=use_gpu))
out_class2, out_reg2 = colorizer2.forward(np2tens(img_rs_l_norm,use_gpu=use_gpu), np2tens(in_ab_norm,use_gpu=use_gpu), np2tens(in_mask_norm,use_gpu=use_gpu))

print('Norm of classifiaction tensor',torch.norm(out_class))
print('Norm of discrepancy', torch.norm(colorizer.upsample4(out_class2) - out_class))


