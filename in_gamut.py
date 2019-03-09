
from skimage import color
import numpy as np

L = 100
A = 110
B = 110

thresh = .1

in_gamuts = np.zeros((L+1,2*A+1,2*B+1),dtype='bool')

# construct a 221x221x3 image with constant L value and every possible (a,b) value
a_vals = np.arange(-A,A+1)
b_vals = np.arange(-B,B+1)
lab_vals = np.concatenate((np.zeros((2*A+1,2*A+1,1)),np.repeat(a_vals[:,None,None],2*A+1,axis=1), np.repeat(b_vals[None,:,None],2*B+1,axis=0)),axis=2)

# loop through L values
for ll in range(L+1):
	print('L: %03d/%03d'%(ll,L))
	lab_vals[:,:,0] = ll

	# convert to rgb and back
	lab_mods = color.rgb2lab(np.clip(color.lab2rgb(lab_vals),0,1))

	# calculate the difference
	diffs = np.sqrt(np.sum((lab_vals-lab_mods)**2,axis=2))

	in_gamuts[ll,:,:] = diffs < thresh

np.save('in_gamuts',in_gamuts)


import matplotlib.pyplot as plt
plt.imshow(in_gamuts[80,:,:],extent=[-110,110,110,-110])
plt.ylabel('a')
plt.xlabel('b')
plt.show()
