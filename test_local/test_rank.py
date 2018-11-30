#! /usr/bin/python
# -*- coding: utf8 -*-

""" GAN-CLS """
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk

from utils import *
from model import *
import model

###======================== PREPARE DATA ====================================###
print("Loading data from pickle ...")
import pickle
with open("/home/cwt/Text2Face_ng/data/output_pickle/_vocab2.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("/home/cwt/Text2Face_ng/data/output_pickle/_image_train2.pickle", 'rb') as f:
    images_train_hd, images_train = pickle.load(f)
with open("/home/cwt/Text2Face_ng/data/output_pickle/_image_test2.pickle", 'rb') as f:
    images_test_hd, images_test = pickle.load(f)
with open("/home/cwt/Text2Face_ng/data/output_pickle/_n2.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_images_train, n_images_test = pickle.load(f)
with open("/home/cwt/Text2Face_ng/data/output_pickle/_caption2.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)

images_train = np.array(images_train)
images_test = np.array(images_test)
images_train_hd = np.array(images_train_hd)
images_test_hd = np.array(images_test_hd)

batch_size = 25
z_dim = 100 #38*2
t_dim = 128
image_size = 64
image_hd_size = 256
ni = int(np.ceil(np.sqrt(batch_size)))


tl.files.exists_or_mkdir("/home/cwt/Text2Face_ng/test_local/hd_real_image")
tl.files.exists_or_mkdir("/home/cwt/Text2Face_ng/test_local/hd_fake_image")
tl.files.exists_or_mkdir("/home/cwt/Text2Face_ng/test_local/low_fake_image")
tl.files.exists_or_mkdir("/home/cwt/Text2Face_ng/test_local/low_real_image")
save_dir = "/home/cwt/Text2Face_ng/output_local/checkpoint" # 模型加载路径

def main_test():
    ###======================== DEFIINE MODEL ===================================###
    # stageI G input:t_real_image,t_real_caption,t_z
    # stageI D input:t_real_image,t_wrong_caption,t_real_caption,t_z
    # stageII G input:t_hd_real_image,t_real_caption,t_hd_fake_image
    # stageII D input:t_hd_real_image,t_wrong_caption,t_real_caption,t_hd_fake_image    
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size,1], name = 'real_image')
    t_real_caption = tf.placeholder(dtype=tf.float32, shape=[batch_size, t_dim], name='real_caption_input_f')
    t_wrong_caption = tf.placeholder(dtype=tf.float32, shape=[batch_size, t_dim], name='wrong_caption_input_f')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')
    t_hd_real_image = tf.placeholder('float32',[batch_size,image_hd_size,image_hd_size,1],name='hd_real_image')
    t_hd_fake_image = tf.placeholder('float32',[batch_size,image_size,image_size,1],name='hd_fake_image') #stageII input 
    
    
    """ training interface for txt to face """
    generator = model.generator
    discriminator = model.discriminator
    hd_generator = model.hd_generator
    hd_discriminator = model.hd_discriminator

    net_capt = capt_embed(t_real_caption)
    
    net_fake_image,_ = generator(t_z,net_capt.outputs,is_train=True,reuse=False,batch_size=batch_size) 
    net_d,disc_fake_image_logits = discriminator(net_fake_image.outputs,net_capt.outputs,is_train=True,reuse=False)
    _,disc_real_image_logits = discriminator(t_real_image,net_capt.outputs,is_train=True,reuse=True)
    _,disc_mismatch_logits = discriminator(t_real_image,capt_embed(t_wrong_caption).outputs,is_train=True,reuse=True)
    #不匹配的情况是否应该包含两种情况？（假描述，真照片）（假描述，假照片）（真描述，假照片）（真描述，真照片）
    net_g,_ = generator(t_z,capt_embed(t_real_caption).outputs,is_train=False,reuse=True,batch_size=batch_size)

    ## net structures for hd models
    net_hd_fake_image,_ = hd_generator(t_hd_fake_image,net_capt.outputs,is_train=True,reuse=False,batch_size=batch_size)
    
    net_hd_d,disc_hd_fake_image_logits = hd_discriminator(net_hd_fake_image.outputs,net_capt.outputs,is_train=True,reuse=False)
    _,disc_hd_real_image_logits = hd_discriminator(t_hd_real_image,net_capt.outputs,is_train=True,reuse=True)
    _,disc_hd_mismatch_logits = hd_discriminator(t_hd_real_image,capt_embed(t_wrong_caption).outputs,is_train=True,reuse=True)

    net_hd_g,_ = hd_generator(t_hd_fake_image,capt_embed(t_real_caption).outputs,is_train=False,reuse=True,batch_size=batch_size)

    ###============================ TRAINING ====================================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)

    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')
    net_hg_name = os.path.join(save_dir, 'net_hd_g.npz')
    net_hd_name = os.path.join(save_dir, 'net_hd_d.npz')
    print('Loading npzs...............')
    load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)
    load_and_assign_npz(sess=sess, name=net_d_name, model=net_d)
    load_and_assign_npz(sess=sess, name=net_hg_name, model=net_hd_g)
    load_and_assign_npz(sess=sess, name=net_hd_name, model=net_hd_d)
   
    n_epoch = 0
    test_freq = 5
    print_freq = 50
    n_batch_epoch = int(n_images_test/batch_size)
    print(n_batch_epoch)
    index_file_train = open("/home/cwt/Text2Face_ng/data/index_file_train.txt",'w')
    index_file_test = open("/home/cwt/Text2Face_ng/data/index_file_test.txt",'w')
    for epoch in range(0, n_epoch+1):
        start_time = time.time()
        for step in range(0,n_batch_epoch,2):
            step_time = time.time()
            ## get matched text
            # 
            print(step)
            idexs = [i for i in range(batch_size*step,batch_size*step+batch_size*2,2)]
            real_idexs = idexs
            print(idexs)
            b_real_caption = captions_ids_test[idexs]
            #print(np.shape(captions_ids_train[20]))
            #print(np.shape(b_real_caption))
            b_real_caption = tl.prepro.pad_sequences(b_real_caption,maxlen=t_dim, padding='post')
            #print(np.shape(b_real_caption))
            ## get real image
            b_real_images = images_test[np.floor(np.asarray(idexs).astype('float')/1).astype('int')]
            #print(np.shape(b_real_images))
            ## get hd real image
            b_hd_real_images = images_test_hd[np.floor(np.asarray(idexs).astype('float')/1).astype('int')]
            ## get noise
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)
            b_real_images = np.array(b_real_images).reshape(batch_size,64,64,1)
            b_hd_real_images = np.array(b_hd_real_images).reshape(batch_size,256,256,1)

            b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
            b_hd_real_images = threading_data(b_hd_real_images, prepro_img, mode='train')

 
            fake_img_gen = sess.run([net_g.outputs], feed_dict={
                                    t_real_caption : b_real_caption,
                                    t_z : b_z}) 
            fake_img_gen = np.array(fake_img_gen).reshape(batch_size,64,64,1)
            img_gen = sess.run([net_hd_g.outputs], feed_dict={
                                    t_real_caption : b_real_caption,
                                    t_hd_fake_image: fake_img_gen})
            img_gen = np.array(img_gen).reshape(batch_size,256,256,1)
            save_images(fake_img_gen,[ni,ni], '/home/cwt/Text2Face_ng/test_local/low_fake_image/test_fake_{:02d}.png'.format(step+epoch*n_batch_epoch))
            save_images(b_real_images,[ni,ni],'/home/cwt/Text2Face_ng/test_local/low_real_image/test_real_{:02d}.png'.format(step+epoch*n_batch_epoch))
            save_images(img_gen,[ni,ni], '/home/cwt/Text2Face_ng/test_local/hd_fake_image/test_fake_{:02d}.png'.format(step+epoch*n_batch_epoch)) 
            save_images(b_hd_real_images,[ni,ni],'/home/cwt/Text2Face_ng/test_local/hd_real_image/test_real_{:02d}.png'.format(step+epoch*n_batch_epoch))         
            if (step) % print_freq == 0:                                  
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs"\
                % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time))
            index_file_test.write(str(real_idexs)+'\n')
           

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="test",
                       help='train, test')

    args = parser.parse_args()

    if args.mode == "test":
        main_test()

#! /usr/bin/python
# -*- coding: utf8 -*-
