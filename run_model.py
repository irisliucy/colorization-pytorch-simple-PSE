
import numpy as np
import cv2
import scipy.misc
from skimage import color
import torch
from IPython import embed
import argparse
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

import util

parser = argparse.ArgumentParser()

parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')

# model parameters
parser.add_argument('--arch', type=str, default='siggraph', help='[siggraph] or [drn_d_22]')
parser.add_argument('--img_path', type=str, default='./imgs/migrant_mother.jpg', help='image to process')
parser.add_argument('--model_path', type=str, default='./weights/caffemodel_mask01_rec.pth', help='model weights')
parser.add_argument('--hint_ab_path', type=str, default=None, help='hints saved off as [2 x Hproc x Wproc]')
parser.add_argument('--hint_mask_path', type=str, default=None, help='hint masked saved off as [2 x Hproc x Wproc]')

# classification point
parser.add_argument('--hw_class', type=float, default=(50,120), nargs='+', help='point to look at for predicted distribution')
# h,w = 50, 120 # mother's face
# h,w = 30, 45 # point in background

# constants
parser.add_argument('--l_norm', type=float, default=100., help='L normalization')
parser.add_argument('--l_cent', type=float, default=50., help='L center')
parser.add_argument('--ab_norm', type=float, default=110., help='ab normalization')
parser.add_argument('--mask_cent', type=float, default=0., help='hint mask centering')
parser.add_argument('--HW_proc', type=float, default=(256,256), nargs='+', help='dimension to process image')
parser.add_argument('--A', type=int, default=23, help='number of bins')
parser.add_argument('--ab_step', type=int, default=10., help='ab increments when discretizing')


opt = parser.parse_args()

# ***** COLOR INPUT POINTS *****
H_proc, W_proc = opt.HW_proc
if(opt.hint_ab_path is None):
	in_ab = np.zeros((H_proc, W_proc, 2))
else:
	in_ab = np.load(opt.hint_ab_path).transpose((1,2,0)) # ./imgs/migrant_mother/im_ab.npy

if(opt.hint_mask_path is None):
	in_mask = np.zeros((H_proc, W_proc, 1))
else:
	in_mask = 1.*np.load(opt.hint_mask_path).transpose((1,2,0)) # ./imgs/migrant_mother/im_mask.npy

if(opt.arch=='siggraph'):
	import models.siggraph
	colorizer = models.siggraph.SIGGRAPHGenerator()
elif(opt.arch=='drn_d_22'):
	import models.drnseg
	colorizer = models.drnseg.DRNSeg(model_name='drn_d_22')

print('Loading from [%s]'%opt.model_path)
# embed()
a = torch.load(opt.model_path)
keys_colorizer = colorizer.state_dict().keys()
keys_load = a.keys()
print('Non-matching keys', np.setdiff1d(keys_colorizer, keys_load), np.setdiff1d(keys_load, keys_colorizer))
print('**If above list is exessively long, there is probably an error**')
colorizer.load_state_dict(a,strict=False)

if(opt.use_gpu):
	colorizer.cuda()
colorizer.eval()

# ***** LOAD IMAGE, PREPARE DATA *****
img_orig = cv2.imread(opt.img_path)[:,:,::-1]
(H_orig,W_orig) = img_orig.shape[:2]
print('[%ix%i] Original resolution'%(H_orig, W_orig))
print('[%ix%i] Processed resolution'%(H_proc,W_proc))

# resize to processing size, take L channel for input
img_rs = cv2.resize(img_orig, (W_proc, H_proc), interpolation=cv2.INTER_CUBIC)
img_rs_lab = color.rgb2lab(img_rs)
img_rs_l = img_rs_lab[:,:,[0]]
img_rs_l_norm = (img_rs_l-opt.l_cent)/opt.l_norm # normalized

# normalize & center input ab, input mask
in_ab_norm = in_ab/opt.ab_norm
in_mask_norm = in_mask - opt.mask_cent

# ***** RUN MODEL *****
out_class, out_reg = colorizer.forward(util.np2tens(img_rs_l_norm,use_gpu=opt.use_gpu), 
	util.np2tens(in_ab_norm,use_gpu=opt.use_gpu), 
	util.np2tens(in_mask_norm,use_gpu=opt.use_gpu))
out_class = out_class.data # 1 x AB x H_proc x W_proc, probability distribution at every spatial location (h,w) of possible colors
out_reg = out_reg.data # 1 x 2 x H_proc x W_proc

out_ab_norm = util.tens2np(out_reg)
out_ab = out_ab_norm*opt.ab_norm # un-normalize

# ***** CONCATENATE WITH INPUT *****
# concatenate with L channel, convert to RGB, save
out_lab = np.concatenate((img_rs_l,out_ab),axis=2)
out_rgb = util.lab2rgb_clip(out_lab)

# ***** COMPUTE UNCERTAINTY *****
out_entropy = -torch.sum(out_class*torch.log(out_class),dim=1,keepdim=True)

# for visualization
in_ab_lab_flat = np.concatenate((in_mask*50, in_ab), axis=2)
in_ab_rgb_flat = util.lab2rgb_clip(in_ab_lab_flat)

in_ab_lab_img = np.concatenate((img_rs_l, in_ab), axis=2)
in_ab_rgb_img = util.lab2rgb_clip(in_ab_lab_img)



plt.figure(figsize=(18,6))
plt.subplot(1,4,1)
plt.imshow(in_mask[:,:,0],clim=(0,1),cmap='gray')
plt.title('Input hint mask')
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(in_ab_rgb_flat)
plt.title('Input hints')
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(in_ab_rgb_img)
plt.title('Input (grayscale + hints)')
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(out_rgb)
plt.title('Output')
plt.axis('off')

# plt.show()




# ***** COLOR DISTRIBUTION PREDICTION *****
# color bins for color classification and output recommendations
A = opt.A # the ab colormap is divided up into 23x23 color bins
AB = A**2 # there are 23*23=529 color bins in total
ab_step = opt.ab_step # spacing between discretized color bins
ab_edge = opt.ab_norm + ab_step/2
a_range = np.arange(-opt.ab_norm, opt.ab_norm+ab_step, step=ab_step)
bbs, aas = np.meshgrid(a_range, a_range)
# abs_norm = abs/ab_norm # 529x2, bin centers normalized from [-1, +1]
ab_labs = np.concatenate((50+np.zeros((A,A,1)),aas[:,:,None],bbs[:,:,None]),axis=2)
ab_rgbs = util.lab2rgb_clip(ab_labs)

# capture the probability distribution at a single point
h,w = opt.hw_class
out_class_point = out_class[0,:,h,w] # 529, probability distribution over discretized space

# PLOT RESULTS
# show queried point on original image
plt.figure(figsize=(18,6))
plt.subplot(1,4,1)
plt.imshow(out_entropy.cpu().numpy()[0,0,:,:],clim=(0,5),cmap='hot')
plt.plot(w,h,'wo', markersize=14)
# plt.colorbar()
plt.title('Prediction Entropy (White O)')
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(in_ab_rgb_img)
plt.plot(w,h,'wo', markersize=14)
plt.title('Input + Queried point (White O)')
plt.axis('off')

# plot probability distribution over ab colorspace
plt.subplot(1,4,3)
plt.imshow(out_class_point.cpu().reshape(A,A).numpy(), 
	extent=[-ab_edge,ab_edge,ab_edge,-ab_edge], cmap='hot')
plt.xlabel('b')
plt.ylabel('a')
plt.title('Predicted distribution')

plt.subplot(1,4,4)
plt.imshow(ab_rgbs,  
	extent=[-ab_edge,ab_edge,ab_edge,-ab_edge])
plt.xlabel('b')
plt.ylabel('a')
plt.title('ab colors (L=50)')
plt.show()

