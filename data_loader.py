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
from numpy import *;
import numpy as np
import chardet   #需要导入这个模块，检测编码格式
dataset = 'face'

need_256 = True

#cwd = os.getcwd() #Gets the current file path.
cwd = 'E:/Attribute2Sketch/'
#print(cwd)
hd_img_dir = os.path.join(cwd,'data/sketch_blending_1') # directory of hd images
img_dir = os.path.join(cwd,'data/sketch_blending_1')
#img_dir = os.path.join(cwd,'data/face/new_img2/') 
caption_file1 = open('E:/Attribute2Sketch/data/capt_blending_nl.txt','rb')
VOC_FIR = cwd + 'data/vocab2.txt'

captions_dict = {}

processed_capts = []

#¶ÁÈ¡ÓïÁÏ£¬Í¬Ê±Îª±êÇ©±àÐ´×Öµä
key = 1
for line in caption_file1:
    line = line.strip('\n'.encode(encoding = "utf-8"))
    encode_type = chardet.detect(line)  
    line = line.decode(encode_type['encoding']) #进行相应解码，赋给原标识符（变量）
    line = line.lower() #È«²¿Ð¡Ð´
    line = preprocess_caption(line)
    processed_capts.append(tl.nlp.process_sentence(line,start_word="<S>",end_word="</S>"))
    captions_dict[key] = line #°´Ë³Ðò±àÐ´×Öµä
    key = key+1	
print(" * %d x %d captions found" % (len(captions_dict), len(line)))

if not os.path.isfile(VOC_FIR):
	_ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
else:
	print("WARNING: vocab.txt already exists")
vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")
captions_ids = []

try:
	tmp = captions_dict.items()
except:
	tmp = captions_dict.iteritems()
# ´´½¨±êÇ©ÏòÁ¿ÁÐ±í
for key, v in tmp:
    captions_ids.append([vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])

captions_ids = np.asarray(captions_ids)
print (" * tokenized %d captions" % len(captions_ids))

#²âÊÔtext to id ºÍid to text
img_capt = captions_dict[1]
print("img_capt: %s" % img_capt)
print("nltk.tokenized sentence: %s" % nltk.tokenize.word_tokenize(img_capt))

img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
print(np.shape(img_capt_ids))
print("img_capt_ids: %s" % img_capt_ids)
print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])
#load images
img_title_list = [] #ÓÃÓÚ´æ·ÅÍ¼Æ¬Â·¾¶
# img_list_file = open('E:/Attribute2Sketch/data/added_img_and_caption/caption_files2/joint_wm_titles.txt','rb')
filenames = os.listdir(hd_img_dir)
filenames.sort()
for filename in filenames:              #listdir的参数是文件夹的路径
        #print (filename)                                  #此时的filename是文件夹中文件的名称
        img_title_list.append(filename)

# for line in img_list_file:
#     # line = line.strip('\n'.encode(encoding = "utf-8"))
#     line = line.strip()
#     encode_type = chardet.detect(line)  
#     line = line.decode(encode_type['encoding']) #进行相应解码，赋给原标识符（变量）
#     img_title_list.append(line)

print(" * %d images found, start loading..." % len(img_title_list))
print(" img_title_list[0]",img_title_list[0])
s=time.time()

images = []
images_256 = []
#loading 64*64 images from img_dir and 256*256 images from hd_img_dir

for name in img_title_list:

        img_raw = scipy.misc.imread(os.path.join(img_dir, name),mode = 'L') #ºÏ²¢Â·¾¶
        # img_raw = scipy.misc.imread(os.path.join(img_dir, name))#3通道，用于测试
        img = np.array(scipy.misc.imresize(img_raw,[64,64]))
        img = img.astype(np.float32)
        img_rot = np.fliplr(img)
        images.append(img) #ÒÀ´ÎÏòÍ¼Æ¬ÁÐ±íÖÐÌí¼ÓÍ¼Æ¬

        # images.append(img_rot)		
        if need_256:
            img_raw = scipy.misc.imread(os.path.join(hd_img_dir,name),mode = 'L') #ÕæÊµhd_real_imageÊÇRGB 3Í¨µÀµÄÂð  
            # img_raw = scipy.misc.imread(os.path.join(hd_img_dir,name))# 3通道，用于测试
            img = np.array(scipy.misc.imresize(img_raw,[256,256]))
            img = img.astype(np.float32)
            img_rot = np.fliplr(img)
            images_256.append(img) #ÒÀ´ÎÏòÍ¼Æ¬ÁÐ±íÖÐÌí¼ÓÍ¼Æ¬
            
            # images_256.append(img_rot) #ÒÀ´ÎÏòÍ¼Æ¬ÁÐ±íÖÐÌí¼ÓË®Æ½·­×ª
print(" * loading and resizing took %ss" % (time.time()-s))

n_images = len(images)
n_captions = len(captions_ids)

print("n_captions: %d n_images: %d" % (n_captions,n_images))
#ÉèÖÃÇ°1000ÕÅ×÷ÎªÑµÁ·¼¯£¬1100£ºÎª²âÊÔ¼¯ 0~999 1000~1099

captions_ids_train , captions_ids_test = captions_ids[:20000],captions_ids[20000:]
images_train, images_test = images[:20000], images[20000:]
# captions_ids_train, captions_ids_test = captions_ids[200:2200], captions_ids[:200]
# images_train, images_test = images[200:2200], images[:200]
#indexs_train,indexs_test = indexs[:1100],indexs[1100:]

if need_256:
        images_train_256, images_test_256 = images_256[:20000], images_256[20000:]
n_images_train = len(images_train)
n_images_test = len(images_test)
n_captions_train = len(captions_ids_train)
n_captions_test = len(captions_ids_test)
#n_indexs_train = len(indexs_train)
#n_indexss_test = len(indexs_test)
print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))
import pickle

def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)

save_all(vocab, 'E:/Attribute2Sketch/data/output_pickle/_vocab2.pickle')
save_all((images_train_256, images_train), 'E:/Attribute2Sketch/data/output_pickle/_image_train2.pickle')
save_all((images_test_256, images_test), 'E:/Attribute2Sketch/data/output_pickle/_image_test2.pickle')
# save_all((images_train, images_test), 'E:/Attribute2Sketch/data/output_pickle/_images5.pickle')
save_all((n_captions_train, n_captions_test, n_images_train, n_images_test), 'E:/Attribute2Sketch/data/output_pickle/_n2.pickle')
save_all((captions_ids_train, captions_ids_test), 'E:/Attribute2Sketch/data/output_pickle/_caption2.pickle')
