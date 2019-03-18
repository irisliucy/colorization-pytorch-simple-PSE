
import numpy as np
import cv2
import scipy.misc
from skimage import color
import torch
from IPython import embed
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

import model_rec as model

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
if(use_gpu):
	colorizer.cuda()
else:
	colorizer.cpu()
colorizer.eval()

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
# in_ab = np.load('./imgs/migrant_mother/im_ab.npy').transpose((1,2,0))
# in_mask = 1.*np.load('./imgs/migrant_mother/im_mask.npy').transpose((1,2,0))
in_ab = np.zeros((H_proc, W_proc, 2))
in_mask = np.zeros((H_proc, W_proc, 1))

# normalize & center input ab, input mask
in_ab_norm = in_ab/ab_norm
in_mask_norm = in_mask - mask_cent

# ***** RUN MODEL *****
out_class, out_reg = colorizer.forward(np2tens(img_rs_l_norm,use_gpu=use_gpu), np2tens(in_ab_norm,use_gpu=use_gpu), np2tens(in_mask_norm,use_gpu=use_gpu))
out_class = out_class.data # 1 x AB x H_proc x W_proc, probability distribution at every spatial location (h,w) of possible colors
out_reg = out_reg.data # 1 x 2 x H_proc x W_proc

out_ab_norm = tens2np(out_reg)
out_ab = out_ab_norm*ab_norm # un-normalize

# ***** CONCATENATE WITH INPUT *****
# resize output to original resolution
out_orig_ab = zoom(out_ab, (1.*H_orig/H_proc, 1.*W_orig/W_proc, 1) )

# concatenate with L channel, convert to RGB, save
out_orig_lab = np.concatenate((img_orig_l,out_orig_ab),axis=2)
out_orig_rgb = np.uint8(np.clip(color.lab2rgb(out_orig_lab),0,1)*255)
cv2.imwrite('./imgs/migrant_mother/output_fullres.png',out_orig_rgb[:,:,::-1])

# ***** COMPUTE UNCERTAINTY *****

out_entropy = -torch.sum(out_class*torch.log(out_class),dim=1,keepdim=True)


# ***** COLOR RECOMMENDATION CODE *****

# color bins for color classification and output recommendations
A = 23 # the ab colormap is divided up into 23x23 color bins
AB = A**2 # there are 23*23=529 color bins in total
ab_step = 10 # spacing between discretized color bins
ab_edge = ab_norm + ab_step/2
a_range = np.arange(-ab_norm, ab_norm+ab_step, step=ab_step)
bbs, aas = np.meshgrid(a_range, a_range)
aas = aas.flatten()
bbs = bbs.flatten()
abs = np.concatenate((aas[:,None],bbs[:,None]),axis=1) # 529x2, list of discretized bin centers
abs_norm = abs/ab_norm # 529x2, bin centers normalized from [-1, +1]

K = 7 # number of recommendations from kmeans network

# initialize network which converts probability distribution ==> K discrete recommendations
recommender = model.KMeansGenerator(AB, K)
recommender.load_state_dict(torch.load('./models/net_K_03_07.pth'))
recommender.cuda() if use_gpu else recommender.cpu()
recommender.eval()

# capture the probability distribution at a single point
h,w = 50, 120 # mother's face
# h,w = 30, 45 # point in background
out_class_point = out_class[0,:,h,w] # 529, probability distribution over discretized space
in_point = img_rs_lab[h,w,0] # grayscale value

# use recommender network to get 7 discrete recommendations
reccs_ab = ab_norm*recommender(out_class_point[None,:,None,None]).reshape(K,2).data.cpu().numpy() # 7x2 list of colors in ab space

# convert recommended colors to RGB and display
reccs_lab = np.concatenate((in_point+np.zeros((K,1)),reccs_ab),axis=1)
reccs_rgb = color.lab2rgb(reccs_lab[None,:,:])

# recommender produces unordered points, compute an ordering based on probability distribution
dists = np.sum((reccs_ab[None,:,:] - abs[:,None,:])**2,axis=2) # for each of the 529 bins, compute distance to each of the recommended colors
inds = np.argmin(dists,axis=1) # for each of the 529 bins, figure out which recommended color is the closest
reccs_probs = np.array([np.sum(out_class_point.data.cpu().numpy()[inds==kk]) for kk in range(K)]) # probability of each of the recommended colors, should sum to 1
reccs_sorted_inds = np.argsort(reccs_probs)[::-1]


# PLOT RESULTS

# show queried point on original image
plt.figure()
plt.imshow(img_rs_lab[:,:,0], cmap='gray')
plt.plot(w,h,'rx')
plt.savefig('./imgs/migrant_mother/predicted_point_%03d_%03d.png'%(h,w))
# plt.show()

# plot probability distribution over ab colorspace
plt.figure()
plt.imshow(out_class_point.cpu().reshape(A,A).numpy(), extent=[-ab_edge,ab_edge,ab_edge,-ab_edge])
plt.plot(reccs_ab[:,1],reccs_ab[:,0],'wx')
plt.xlabel('b')
plt.ylabel('a')
plt.savefig('./imgs/migrant_mother/predicted_dist_%03d_%03d.png'%(h,w))
# plt.show()

plt.figure()
plt.imshow(reccs_rgb)
plt.savefig('./imgs/migrant_mother/recc_colors_%03d_%03d.png'%(h,w))
# plt.show()

plt.figure()
plt.imshow(reccs_rgb[:,reccs_sorted_inds,:])
plt.savefig('./imgs/migrant_mother/recc_colors_sorted_%03d_%03d.png'%(h,w))
