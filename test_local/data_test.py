#!usr/bin/python
import scipy
import os
import re
import time
import nltk
import re
import string
import tensorlayer as tl
from utils import *
import numpy as np
dataset = 'face'

need_256 = True
path_real = "/home/cwt/Text2Face_ng/test_local/low_real_image" #文件夹目录
path_hd_real = "/home/cwt/Text2Face_ng/test_local/hd_real_image" #文件夹目录 
path_fake = "/home/cwt/Text2Face_ng/test_local/low_fake_image" #文件夹目录 
path_hd_fake = "/home/cwt/Text2Face_ng/test_local/hd_fake_image" #文件夹目录 
images_real=[]
images_hd_real=[]
images_fake=[]
images_hd_fake=[]

img_real_list = os.listdir(path_real) #得到文件夹下的所有文件名称
img_hd_real_list = os.listdir(path_hd_real) #得到文件夹下的所有文件名称
img_fake_list = os.listdir(path_fake) #得到文件夹下的所有文件名称
img_hd_fake_list = os.listdir(path_hd_fake) #得到文件夹下的所有文件名称

import pickle

def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)
print(" * %d images found, start loading..." % len(img_real_list))
s=time.time()
for name in img_real_list:
	img_raw = scipy.misc.imread(os.path.join(path_real, name),mode='L')
	img = np.array(scipy.misc.imresize(img_raw,[64,64]))
	img = img.astype(np.float32)
	images_real.append(img) 
print(" * loading and resizing took %ss" % (time.time()-s))

print(" * %d images found, start loading..." % len(img_hd_real_list))
s=time.time()
for name in img_hd_real_list:
	img_raw = scipy.misc.imread(os.path.join(path_hd_real, name),mode='L')
	img = np.array(scipy.misc.imresize(img_raw,[256,256]))
	img = img.astype(np.float32)
	images_hd_real.append(img) 
print(" * loading and resizing took %ss" % (time.time()-s))

print(" * %d images found, start loading..." % len(img_fake_list))
s=time.time()
for name in img_fake_list:
	img_raw = scipy.misc.imread(os.path.join(path_fake, name),mode='L')
	img = np.array(scipy.misc.imresize(img_raw,[64,64]))
	img = img.astype(np.float32)
	images_fake.append(img)
print(" * loading and resizing took %ss" % (time.time()-s))

print(" * %d images found, start loading..." % len(img_hd_fake_list))
s=time.time()
for name in img_hd_fake_list:
	img_raw = scipy.misc.imread(os.path.join(path_hd_fake, name),mode='L')
	img = np.array(scipy.misc.imresize(img_raw,[256,256]))
	img = img.astype(np.float32)
	images_hd_fake.append(img)
print(" * loading and resizing took %ss" % (time.time()-s))

save_all((images_hd_real, images_real), '/home/cwt/Text2Face_ng/test_local/_image_real_test_local.pickle')
save_all((images_hd_fake, images_fake), '/home/cwt/Text2Face_ng/test_local/_image_fake_test_local.pickle')
