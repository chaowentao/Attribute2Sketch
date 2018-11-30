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
with open("E:/Attribute2Sketch/data/output_pickle/_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("E:/Attribute2Sketch/data/output_pickle/_image_train.pickle", 'rb') as f:
    images_train_hd, images_train = pickle.load(f)
with open("E:/Attribute2Sketch/data/output_pickle/_image_test.pickle", 'rb') as f:
    images_test_hd, images_test = pickle.load(f)
with open("E:/Attribute2Sketch/data/output_pickle/_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_images_train, n_images_test = pickle.load(f)
with open("E:/Attribute2Sketch/data/output_pickle/_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test  = pickle.load(f)
# images_train = images_train[:6000]
# images_train_hd = images_train_hd[:6000]
# captions_ids_train = captions_ids_train[:6000]
images_train = np.array(images_train)
images_test = np.array(images_test)
images_train_hd = np.array(images_train_hd)
images_test_hd = np.array(images_test_hd)
captions_ids_train = np.array(captions_ids_train)
captions_ids_test = np.array(captions_ids_test)
print("captions_ids_train:",np.shape(captions_ids_train))
print("images_train_hd:",np.shape(images_train_hd))
print("images_test_hd:",np.shape(images_test_hd))
# n_captions_train = 6000
# n_images_train = 6000



batch_size = 16
z_dim = 100 #38*2
t_dim = 64
image_size = 64
image_hd_size = 256
ni = int(np.ceil(np.sqrt(batch_size)))

tl.files.exists_or_mkdir("E:/Attribute2Sketch/output_div/step1_gan-cls")
tl.files.exists_or_mkdir("E:/Attribute2Sketch/output_div/step1_gan-real")
tl.files.exists_or_mkdir("E:/Attribute2Sketch/output_div/checkpoint")
tl.files.exists_or_mkdir("E:/Attribute2Sketch/output_div/low_def_real")
tl.files.exists_or_mkdir("E:/Attribute2Sketch/output_div/low_def_fake")

tl.files.exists_or_mkdir("E:/Attribute2Sketch/test_div/step1_gan-cls")
tl.files.exists_or_mkdir("E:/Attribute2Sketch/test_div/step1_gan-real")
tl.files.exists_or_mkdir("E:/Attribute2Sketch/test_div/low_def_real")
tl.files.exists_or_mkdir("E:/Attribute2Sketch/test_div/low_def_fake")
save_dir = "E:/Attribute2Sketch/output_div/checkpoint"

def main_train():
    ###======================== DEFIINE model ===================================###
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
    hd_discriminator_local = model.hd_discriminator_local

    net_capt = capt_embed(t_real_caption)
    
    net_fake_image,_ = generator(t_z,net_capt.outputs,is_train=True,reuse=False,batch_size=batch_size) 
    net_d,disc_fake_image_logits_1 = discriminator(net_fake_image.outputs[:,:,:,0:1],net_capt.outputs,is_train=True,reuse=False)
    _,disc_fake_image_logits_2 = discriminator(net_fake_image.outputs[:,:,:,1:2],net_capt.outputs,is_train=True,reuse=True)
    _,disc_fake_image_logits_3 = discriminator(net_fake_image.outputs[:,:,:,2:3],net_capt.outputs,is_train=True,reuse=True)
    _,disc_fake_image_logits_4 = discriminator(net_fake_image.outputs[:,:,:,3:4],net_capt.outputs,is_train=True,reuse=True)

    disc_fake_image_logits = (disc_fake_image_logits_1+disc_fake_image_logits_2+disc_fake_image_logits_3+disc_fake_image_logits_4)/4
    
    _,disc_real_image_logits = discriminator(t_real_image,net_capt.outputs,is_train=True,reuse=True)
    _,disc_mismatch_logits = discriminator(t_real_image,capt_embed(t_wrong_caption).outputs,is_train=True,reuse=True)
    #不匹配的情况是否应该包含两种情况？（假描述，真照片）（假描述，假照片）（真描述，假照片）（真描述，真照片）
    net_g,_ = generator(t_z,capt_embed(t_real_caption).outputs,is_train=False,reuse=True,batch_size=batch_size)

    d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits, tf.zeros_like(disc_mismatch_logits),name='d2')
    d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')


    # cwt chenged 5/10
    d_loss = d_loss1 + 0.5*(d_loss3+d_loss2)
    # g_loss = tl.cost.mean_squared_error(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), is_mean=False)
    g_loss1 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g1')
    global_loss1 = tl.cost.absolute_difference_error(net_fake_image.outputs[:,:,:,0:1], t_real_image, is_mean=True) #L1 Loss
    global_loss2 = tl.cost.absolute_difference_error(net_fake_image.outputs[:,:,:,1:2], t_real_image, is_mean=True) #L1 Loss
    global_loss3 = tl.cost.absolute_difference_error(net_fake_image.outputs[:,:,:,2:3], t_real_image, is_mean=True) #L1 Loss
    global_loss4 = tl.cost.absolute_difference_error(net_fake_image.outputs[:,:,:,3:4], t_real_image, is_mean=True) #L1 Loss
    g_loss = g_loss1 + 1 * (tf.minimum(tf.minimum(tf.minimum(global_loss1,global_loss2),global_loss3),global_loss4))
    
    
    ## net structures for hd models
    net_hd_fake_image,_ = hd_generator(t_hd_fake_image,net_capt.outputs,is_train=True,reuse=False,batch_size=batch_size)
    
    net_hd_d,disc_hd_fake_image_logits = hd_discriminator_local(net_hd_fake_image.outputs,net_capt.outputs,is_train=True,reuse=False)
    _,disc_hd_real_image_logits = hd_discriminator_local(t_hd_real_image,net_capt.outputs,is_train=True,reuse=True)
    _,disc_hd_mismatch_logits = hd_discriminator_local(t_hd_real_image,capt_embed(t_wrong_caption).outputs,is_train=True,reuse=True)
    
    net_hd_g,_ = hd_generator(t_hd_fake_image,capt_embed(t_real_caption).outputs,is_train=False,reuse=True,batch_size=batch_size)

    hd_loss1 = tl.cost.sigmoid_cross_entropy(disc_hd_real_image_logits, tf.ones_like(disc_hd_real_image_logits), name = 'hd1')
    hd_loss2 = tl.cost.sigmoid_cross_entropy(disc_hd_fake_image_logits, tf.zeros_like(disc_hd_fake_image_logits), name = 'hd2')
    hd_loss3 = tl.cost.sigmoid_cross_entropy(disc_hd_mismatch_logits, tf.zeros_like(disc_hd_mismatch_logits), name = 'hd3')

    # cwt chenged 5/10
    hd_loss = hd_loss1 + 0.5*(hd_loss2+hd_loss3)

    hg_loss1 = tl.cost.sigmoid_cross_entropy(disc_hd_fake_image_logits, tf.ones_like(disc_hd_fake_image_logits), name = 'hg1')
    hg_loss2 = tl.cost.absolute_difference_error(net_hd_fake_image.outputs, t_hd_real_image, is_mean=True) #L1 loss
    
    hg_local_loss1 = tl.cost.absolute_difference_error(net_hd_fake_image.outputs[0:batch_size,48:112,90:154,0:1],  t_hd_real_image[0:batch_size,48:112,90:154,0:1], is_mean=True) #L1 loss 眼睛
    hg_local_loss2 = tl.cost.absolute_difference_error(net_hd_fake_image.outputs[0:batch_size,144:208,90:154,0:1], t_hd_real_image[0:batch_size,144:208,90:154,0:1], is_mean=True) #L1 loss 眼睛
    hg_local_loss3 = tl.cost.absolute_difference_error(net_hd_fake_image.outputs[0:batch_size,96:160,120:184,0:1], t_hd_real_image[0:batch_size,96:160,120:184,0:1], is_mean=True) #L1 loss 鼻子
    hg_local_loss4 = tl.cost.absolute_difference_error(net_hd_fake_image.outputs[0:batch_size,96:160,174:238,0:1], t_hd_real_image[0:batch_size,96:160,174:238,0:1], is_mean=True) #L1 loss 嘴巴
    # hg_local_loss5 = tl.cost.absolute_difference_error(net_hd_fake_image.outputs[0:batch_size,0:64,134:198,0:1], t_hd_real_image[0:batch_size,0:64,134:198,0:1], is_mean=True) #L1 loss 耳朵
    # hg_local_loss6 = tl.cost.absolute_difference_error(net_hd_fake_image.outputs[0:batch_size,192:256,134:198,0:1], t_hd_real_image[0:batch_size,192:256,134:198,0:1], is_mean=True) #L1 loss 耳朵

    hg_loss = hg_loss1 + 10 * (hg_loss2) + 10*(hg_local_loss1+hg_local_loss2+hg_local_loss3+hg_local_loss4)/4
    ####======================== DEFINE TRAIN OPTS ==============================###
    lr = 0.0001
    lr_decay = 0.8
    decay_every = 50
    beta1 = 0.5

    d_vars = tl.layers.get_variables_with_name('discriminator',True,True)
    g_vars = tl.layers.get_variables_with_name('generator',True,True)
    hd_vars = tl.layers.get_variables_with_name('hddis',True,True)
    hg_vars = tl.layers.get_variables_with_name('hdgen',True,True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr,trainable=False)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars )
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars )
    hd_optim = tf.train.AdamOptimizer(4*lr_v, beta1=beta1).minimize(hd_loss, var_list=hd_vars )
    hg_optim = tf.train.AdamOptimizer(4*lr_v, beta1=beta1).minimize(hg_loss, var_list=hg_vars )
    print('FINISHED WITH OPTIMIZERS')
    
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
   
    n_epoch = 300
    test_freq = 3
    print_freq = 100
    n_batch_epoch = int(n_images_train/batch_size)
    # n_batch_epoch = 200
    index_file_train = open("E:/Attribute2Sketch/data/index_file_train.txt",'w')
    index_file_test = open("E:/Attribute2Sketch/data/index_file_test.txt",'w')
    for epoch in range(0, n_epoch+1):
        start_time = time.time()

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)
        for step in range(n_batch_epoch):
	        step_time = time.time()
	    	#     idexs = [step]
	        # get matched text
	        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size) ## 每个batch图片和描述对应的序号
	        real_idexs = idexs
	        b_real_caption = captions_ids_train[idexs]
	        b_real_caption = tl.prepro.pad_sequences(b_real_caption,maxlen=t_dim, padding='post')
	        ## get real image
	        b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/1).astype('int')]
	        ## get hd real image
	        b_hd_real_images = images_train_hd[np.floor(np.asarray(idexs).astype('float')/1).astype('int')]
	        ## get wrong caption
	        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
	        b_wrong_caption = captions_ids_train[idexs]
	        b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption,maxlen=t_dim, padding='post')
	        ## get wrong image
	        idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)
	        while 1: ##防止idexs和idexs2有相同的元素
	            list_c = [] 
	            for i in range(batch_size):
	                if idexs[i]==idexs2[i]:
	                    list_c.append(idexs[i])
	            #print(list_c)
	            if len(list_c)>0:
	                idexs2 = get_random_int(min=0, max=n_captions_train-1, number=batch_size) ## 每个batch图片和描述对应的序号
	            else:
	                break
	        wrong_idexs = idexs2
	        b_wrong_images = images_train[idexs2]
	        ## get hd wrong image
	        b_hd_wrong_images = images_train_hd[idexs2]
	        ## get noise
	        b_z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)
	        b_real_images = np.array(b_real_images).reshape(batch_size,64,64,1)
	        b_wrong_images = np.array(b_wrong_images).reshape(batch_size,64,64,1)
	        b_hd_real_images = np.array(b_hd_real_images).reshape(batch_size,256,256,1)
	        b_hd_wrong_images = np.array(b_hd_wrong_images).reshape(batch_size,256,256,1)
	        
	        b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
	        b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train')
	        b_hd_real_images = threading_data(b_hd_real_images, prepro_img, mode='train')
	        b_hd_wrong_images = threading_data(b_hd_wrong_images, prepro_img, mode='train')

	        if epoch <= 200:
	            ##update D
	            errD,_ = sess.run([d_loss,d_optim],feed_dict={
	                              t_real_image : b_real_images,
	                              t_wrong_caption : b_wrong_caption,
	                              t_real_caption : b_real_caption,
	                              t_z : b_z})    
	            ## update G
	            errG, _ = sess.run([g_loss, g_optim], feed_dict={
	                                t_real_caption : b_real_caption,
	                                t_real_image : b_real_images,
	                                t_z : b_z})
	            if (step) % print_freq == 0:
	                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %4.4f, g_loss: %4.4f "\
	                            % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD , errG ))
        if epoch > 0:
            fake_img_gen = sess.run([net_g.outputs], feed_dict={
                                    t_real_caption : b_real_caption,
                                    t_z : b_z}) 
            fake_img_gen = np.array(fake_img_gen).reshape(batch_size,64,64,4)
            # ##update HD
            # errHD,_ = sess.run([hd_loss,hd_optim],feed_dict={
            #                   t_hd_real_image : b_hd_real_images,
            #                   t_wrong_caption : b_wrong_caption,
            #                   t_real_caption : b_real_caption,
            #                   t_hd_fake_image : fake_img_gen[:,:,:,0:1]})
            # ## update HG
            # errHG, _ = sess.run([hg_loss, hg_optim], feed_dict={
            #                   t_real_caption : b_real_caption,
            #                   t_hd_real_image : b_hd_real_images,
            #                   t_hd_fake_image : fake_img_gen[:,:,:,0:1]})
            # if (step) % print_freq == 0:                                
            #     print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, hd_loss: %4.4f, hg_loss: %4.4f "\
            #             % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errHD, errHG))
	       #  fake_img_gen = sess.run([net_g.outputs], feed_dict={
	       #                                  t_real_caption : b_real_caption,
	       #                                  t_z : b_z}) 
        # fake_img_gen = np.array(fake_img_gen).reshape(batch_size,64,64,4)
            save_images(fake_img_gen[:,:,:,0:1],[ni,ni], 'E:/Attribute2Sketch/output_div/low_def_fake/train_{:02d}_1.png'.format(epoch))
            save_images(fake_img_gen[:,:,:,1:2],[ni,ni], 'E:/Attribute2Sketch/output_div/low_def_fake/train_{:02d}_2.png'.format(epoch))
            save_images(fake_img_gen[:,:,:,2:3],[ni,ni], 'E:/Attribute2Sketch/output_div/low_def_fake/train_{:02d}_3.png'.format(epoch))
            save_images(fake_img_gen[:,:,:,3:4],[ni,ni], 'E:/Attribute2Sketch/output_div/low_def_fake/train_{:02d}_4.png'.format(epoch))
            save_images(b_real_images,[ni,ni],'E:/Attribute2Sketch/output_div/low_def_real/train_{:02d}.png'.format(epoch))
        if epoch > 200:
            img_gen1 = sess.run([net_hd_g.outputs], feed_dict={
                                    t_real_caption : b_real_caption,
                                    t_hd_fake_image: fake_img_gen[:,:,:,0:1]})
            img_gen1 = np.array(img_gen1).reshape(batch_size,256,256,1)
            img_gen2 = sess.run([net_hd_g.outputs], feed_dict={
                                    t_real_caption : b_real_caption,
                                    t_hd_fake_image: fake_img_gen[:,:,:,1:2]})
            img_gen2 = np.array(img_gen2).reshape(batch_size,256,256,1)
            img_gen3 = sess.run([net_hd_g.outputs], feed_dict={
                                    t_real_caption : b_real_caption,
                                    t_hd_fake_image: fake_img_gen[:,:,:,2:3]})
            img_gen3 = np.array(img_gen3).reshape(batch_size,256,256,1)
            img_gen4 = sess.run([net_hd_g.outputs], feed_dict={
                                    t_real_caption : b_real_caption,
                                    t_hd_fake_image: fake_img_gen[:,:,:,3:4]})
            img_gen4 = np.array(img_gen4).reshape(batch_size,256,256,1)            
            save_images(img_gen1,[ni,ni], 'E:/Attribute2Sketch/output_div/step1_gan-cls/train_{:02d}_1.png'.format(step))
            save_images(img_gen2,[ni,ni], 'E:/Attribute2Sketch/output_div/step1_gan-cls/train_{:02d}_2.png'.format(step)) 
            save_images(img_gen3,[ni,ni], 'E:/Attribute2Sketch/output_div/step1_gan-cls/train_{:02d}_3.png'.format(step)) 
            save_images(img_gen4,[ni,ni], 'E:/Attribute2Sketch/output_div/step1_gan-cls/train_{:02d}_4.png'.format(step)) 
            save_images(b_hd_real_images,[ni,ni],'E:/Attribute2Sketch/output_div/step1_gan-real/train_{:02d}.png'.format(step))           
        # index_file_train.write(str(real_idexs)+'\n')

        print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))

        # if epoch>100 and (epoch + 1) % test_freq == 0:
        #     batch_size1 = 64
        #     ni1 = int(np.ceil(np.sqrt(batch_size1)))
        #     idexs = get_random_int(min=0, max=n_captions_test-1, number=batch_size1) ## 每个test图片和描述对应的序号
        #     print(idexs)
        #     sample_caption = captions_ids_test[idexs]
        #     sample_caption = tl.prepro.pad_sequences(sample_caption,maxlen=t_dim,padding='post')
        #     sample_real_images = images_test[np.floor(np.asarray(idexs).astype('float')/1).astype('int')]
        #     sample_real_images = np.array(sample_real_images).reshape(batch_size1,64,64,1)
        #     sample_hd_real_images = images_test_hd[np.floor(np.asarray(idexs).astype('float')/1).astype('int')]
        #     sample_hd_real_images = np.array(sample_hd_real_images).reshape(batch_size1,256,256,1)
        #     sample_z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size1, z_dim)).astype(np.float32)
        #     ## 
        #     fake_img_gen1 = sess.run([net_g.outputs], feed_dict={
        #                             t_real_caption : sample_caption,
        #                             t_z : sample_z})
        #     fake_img_gen1 = np.array(fake_img_gen1).reshape(batch_size1,64,64,4)
        #     save_images(fake_img_gen1[:,:,:,0:1],[ni,ni], 'E:/Attribute2Sketch/test_div/low_def_fake/test_{:02d}_1.png'.format(epoch))
        #     save_images(fake_img_gen1[:,:,:,1:2],[ni,ni], 'E:/Attribute2Sketch/test_div/low_def_fake/test_{:02d}_2.png'.format(epoch))
        #     save_images(fake_img_gen1[:,:,:,2:3],[ni,ni], 'E:/Attribute2Sketch/test_div/low_def_fake/test_{:02d}_3.png'.format(epoch))
        #     save_images(fake_img_gen1[:,:,:,3:4],[ni,ni], 'E:/Attribute2Sketch/test_div/low_def_fake/test_{:02d}_4.png'.format(epoch))

            
        #     save_images(sample_real_images,[ni1,ni1],'E:/Attribute2Sketch/test_div/low_def_real/test_{:02d}.png'.format(epoch))
        #     img_gen1 = sess.run([net_hd_g.outputs], feed_dict={
        #                                 t_real_caption : sample_caption,
        #                                 t_hd_fake_image: fake_img_gen[:,:,:,0:1]})
        #     img_gen1 = np.array(img_gen1).reshape(batch_size1,256,256,1)
        #     img_gen2 = sess.run([net_hd_g.outputs], feed_dict={
        #                                 t_real_caption : sample_caption,
        #                                 t_hd_fake_image: fake_img_gen[:,:,:,1:2]})
        #     img_gen2 = np.array(img_gen2).reshape(batch_size1,256,256,1)
        #     img_gen3 = sess.run([net_hd_g.outputs], feed_dict={
        #                                 t_real_caption : sample_caption,
        #                                 t_hd_fake_image: fake_img_gen[:,:,:,2:3]})
        #     img_gen3 = np.array(img_gen3).reshape(batch_size1,256,256,1)
        #     img_gen4 = sess.run([net_hd_g.outputs], feed_dict={
        #                                 t_real_caption : sample_caption,
        #                                 t_hd_fake_image: fake_img_gen[:,:,:,3:4]})
        #     img_gen4 = np.array(img_gen4).reshape(batch_size1,256,256,1)          
        #     save_images(img_gen1,[ni1,ni1], 'E:/Attribute2Sketch/test_div/step1_gan-cls/test_{:02d}_1.png'.format(epoch))
        #     save_images(img_gen2,[ni1,ni1], 'E:/Attribute2Sketch/test_div/step1_gan-cls/test_{:02d}_2.png'.format(epoch))
        #     save_images(img_gen3,[ni1,ni1], 'E:/Attribute2Sketch/test_div/step1_gan-cls/test_{:02d}_3.png'.format(epoch))
        #     save_images(img_gen4,[ni1,ni1], 'E:/Attribute2Sketch/test_div/step1_gan-cls/test_{:02d}_4.png'.format(epoch))
        #     save_images(sample_hd_real_images,[ni1,ni1],'E:/Attribute2Sketch/test_div/step1_gan-real/test_{:02d}.png'.format(epoch))
        #     #         index_file_test.write(str(idexs)+'\n')
        # ## save model
        if (epoch != 0) and (epoch % 10) == 0:
            tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
            tl.files.save_npz(net_hd_g.all_params, name=net_hg_name, sess=sess)
            tl.files.save_npz(net_hd_d.all_params, name=net_hd_name, sess=sess)
            print("[*] Save checkpoints SUCCESS!")

        if (epoch != 0) and (epoch % 100) == 0:
            tl.files.save_npz(net_g.all_params, name=net_g_name+str(epoch), sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_d_name+str(epoch), sess=sess)
            tl.files.save_npz(net_hd_g.all_params, name=net_hg_name+str(epoch), sess=sess)
            tl.files.save_npz(net_hd_d.all_params, name=net_hd_name+str(epoch), sess=sess)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train",
                       help='train, test_div')

    args = parser.parse_args()

    if args.mode == "train":
        main_train()

#! /usr/bin/python
# -*- coding: utf8 -*-