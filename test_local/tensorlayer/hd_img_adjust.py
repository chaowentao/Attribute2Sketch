#!env/bin/python

import os
import numpy as np
import scipy
from scipy.misc import imsave,imresize
img_dir = 'original_sketch/'
new_img_dir = 'adjusted_sketch/'
img_file_list = []

img_file = open('original_list.txt','r')

for line in img_file:
    line = line.strip()
    img_file_list.append(line)

for img in img_file_list:
    try:
        img_raw = scipy.misc.imread(os.path.join(img_dir,img))
        img_raw = np.array(img_raw)
        img_raw = img_raw[121:581,201:571]#new img size is [460,370]
        img_raw = imresize(img_raw,[256,256])
        imsave(os.path.join(new_img_dir,img),img_raw)
    except:
        print('%s does not exist' % img)
