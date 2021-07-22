import gradio as gr
import numpy as np
import cv2
from skimage import color
import torch

import models.siggraph
import util


#TODO modify as user input
hint_ab_path = './imgs/ppr10k_1019/meanshift_im_ab.npy'
hint_mask_path = './imgs/ppr10k_1019/meanshift_im_mask.npy'
model_path = './weights/caffemodel_mask01_rec.pth'

use_gpu = False
l_cent = 50
l_norm =  100
ab_norm = 110
mask_cent = 0
H_proc, W_proc = 224, 224

colorizer = models.siggraph.SIGGRAPHGenerator()  # load the model
a = torch.load(model_path)
colorizer.load_state_dict(a,strict=False)

def color_sampling():
    """
    Return mask and ab channel mask in
    """
    raise NotImplementedError
    

def run_color_matching(source_image, reference_image):
    # convert it to npy
    if(hint_ab_path is None):
        in_ab = np.zeros((H_proc, W_proc, 2))
    else:
        # in_ab = np.load(hint_ab_path).transpose((1,2,0)) # ./imgs/migrant_mother/im_ab.npy
        in_ab = np.load(hint_ab_path)

    if(hint_mask_path is None):
        in_mask = np.zeros((H_proc, W_proc, 1))
    else:
        # in_mask = 1.*np.load(hint_mask_path).transpose((1,2,0)) # ./imgs/migrant_mother/im_mask.npy
        in_mask = 1.*np.load(hint_mask_path)

    print('[INFO] Initializing colorization model...')
    a = torch.load(model_path)
    colorizer.load_state_dict(a,strict=False)

    if(use_gpu):
        colorizer.cuda()
    colorizer.eval()

    (H_orig,W_orig) = source_image.shape[:2]
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    source_image = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

    # resize to processing size, take L channel for input
    img_rs = cv2.resize(source_image, (W_proc, H_proc), interpolation=cv2.INTER_CUBIC)
    img_rs_lab = color.rgb2lab(img_rs)
    img_rs_l = img_rs_lab[:,:,[0]]
    img_rs_l_norm = (img_rs_l-l_cent)/l_norm # normalized

    # normalize & center input ab, input mask
    in_ab_norm = in_ab/ab_norm
    in_mask_norm = in_mask - mask_cent

    # ***** RUN MODEL *****
    out_class, out_reg = colorizer.forward(util.np2tens(img_rs_l_norm,use_gpu=use_gpu), 
        util.np2tens(in_ab_norm,use_gpu=use_gpu), 
        util.np2tens(in_mask_norm,use_gpu=use_gpu))
    out_class = out_class.data # 1 x AB x H_proc x W_proc, probability distribution at every spatial location (h,w) of possible colors
    out_reg = out_reg.data # 1 x 2 x H_proc x W_proc

    out_ab_norm = util.tens2np(out_reg)
    out_ab = out_ab_norm*ab_norm # un-normalize

    # ***** CONCATENATE WITH INPUT *****
    # concatenate with L channel, convert to RGB, save
    out_lab = np.concatenate((img_rs_l,out_ab),axis=2)
    out_rgb = util.lab2rgb_clip(out_lab)

    print('[INFO] Finish computing the colorization result!')
    return out_rgb

iface = gr.Interface(
    run_color_matching, 
    [gr.inputs.Image(shape=(224, 224)),
    gr.inputs.Image(shape=(2224, 224))],
    "image",
    examples=[
        ["imgs/ppr10k_1019/source.png", "imgs/ppr10k_1019/reference.png"]
    ]
    )

if __name__ == "__main__":
    iface.launch()
