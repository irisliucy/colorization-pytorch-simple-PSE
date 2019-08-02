# colorization-pytorch-simple-PSE

# "Migrant Mother" with automatic colorization
python run_model.py --arch drn_d_22 --model_path weights/drn_d_22_norebal_ep150.pth

# "Migrant Mother" with some pre-saved user interactions
python run_model.py --arch drn_d_22 --model_path weights/drn_d_22_norebal_ep150.pth --hint_ab_path imgs/migrant_mother/im_ab.npy --hint_mask_path imgs/migrant_mother/im_mask.npy
