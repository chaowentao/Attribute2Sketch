#! /usr/bin/python

# -*- coding:utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os
import numpy as np

batch_size = 16
z_dim = 100 # 38*264
image_size = 64
c_dim = 1

t_dim = 64 #45

image_size = 64

s2,s4,s8,s16 = int(image_size/2),int(image_size/4),int(image_size/8),int(image_size/16)
    
w_init = tf.random_normal_initializer(stddev=0.02)
    
gamma_init = tf.random_normal_initializer(1.,0.02)

lrelu = lambda x:tl.act.lrelu(x,0.2)

def capt_embed(input_seqs):
    """ the size of input_seqs is (128,1) """
    net_capt = InputLayer(input_seqs,name='net_capt/input')
    return net_capt  # 


def generator(input_z,input_txt,is_train=True,reuse=False,batch_size=batch_size):
    gf_dim = 128
    #lrelu = lambda x:tl.act.lrelu(x,0.2)
    with tf.variable_scope("generator",reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_z,name='g_input_z')
        net_txt = InputLayer(input_txt,name='g_input_txt')
        net_txt = DenseLayer(net_txt, n_units=t_dim,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init = w_init, b_init=None, name='g_reduce_text/dense')
        net_in = ConcatLayer([net_in,net_txt],concat_dim=1,name='g_concat')
        
        net_h0 = DenseLayer(net_in,n_units=gf_dim*8*s16*s16,W_init=w_init,act=tf.identity,name='g_h0/dense')
        net_h0 = ReshapeLayer(net_h0,shape=[-1,s16,s16,gf_dim*8],name='g_h0/reshape')
        net_h0 = BatchNormLayer(net_h0,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h0/batch_norm')
        ## net h0: 1 deconv/1 conv
        net_h0 = DeConv2d(net_h0,gf_dim*8,(3,3),out_size=(s16,s16),strides=(1,1),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='g_h0/deconv2d')
        net_h0 = BatchNormLayer(net_h0,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h0/batch_norm2')
        net_h0 = Conv2d(net_h0,gf_dim*4,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='g_h0/conv2d')
        net_h0 = BatchNormLayer(net_h0,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h0/batch_norm3')
        ##net h1: 1 deconv/2 conv
        net_h1 = DeConv2d(net_h0,gf_dim*4,(3,3),out_size=(s8,s8),strides=(2,2),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='g_h1/deconv')
        net_h1 = BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h1/batch_norm')
        net_h1 = Conv2d(net_h1,gf_dim*2,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='g_h1/deconv2')
        net_h1 = BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h1/batch_norm2')
        net_h1 = Conv2d(net_h1,gf_dim*2,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='g_h1/deconv3')
        net_h1 = BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h1/batch_norm3')
        ##net h2: 1 deconv/1 conv/1 resnet
        net_h2 = DeConv2d(net_h1,gf_dim,(3,3),out_size=(s4,s4),strides=(2,2),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='g_h2/deconv')
        net_h2 = BatchNormLayer(net_h2,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h2/batch_norm')
        net_h2 = Conv2d(net_h2,gf_dim,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='g_h2/deconv2')
        net_h2 = BatchNormLayer(net_h2,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h2/batch_norm2')
        net = Conv2d(net_h2,gf_dim,(5,5),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='g_h2/deconv3')
        net = BatchNormLayer(net,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h2/batch_norm3')
        net_h2 = ConcatLayer([net_h2,net],concat_dim=3,name='g_h2/concat')
        #print(np.shape(net_h2.outputs))

        ##net h3: 1 deconv/1 conv
        net_h3 = DeConv2d(net_h2,gf_dim,(3,3),out_size=(s2,s2),strides=(2,2),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='g_h3/deconv')
        net_h3 = BatchNormLayer(net_h3,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h3/batch_norm')
        #print(np.shape(net_h3.outputs))
        net_h3 = Conv2d(net_h3,gf_dim,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='g_h3/deconv2')
        print(np.shape(net_h3.outputs))
        net_h3 = BatchNormLayer(net_h3,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='g_h3/batch_norm2')
        ## net h4: 1 deconv
        net_h4 = DeConv2d(net_h3,c_dim,(5,5),out_size=(image_size,image_size),strides=(2,2),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='g_h4/deconv')
        logits = net_h4.outputs
        print(np.shape(logits))
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4,logits


def discriminator(input_image,input_txt=None,is_train=True,reuse=False):
    df_dim = 64
    lrelu = lambda x:tl.act.lrelu(x,0.2)

    with tf.variable_scope("discriminator",reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_image,name='d_in')
        net_h0 = Conv2d(net_in, df_dim, (5,5),(2,2),act=lrelu,padding='SAME',W_init=w_init,name='d_h0/conv2d')
        
        net_h1 = Conv2d(net_h0,df_dim,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='d_h1/batch_norm')

        net_h2 = Conv2d(net_h1,df_dim*2,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='d_h2/batch_norm')

        net_h3 = Conv2d(net_h2,df_dim*2,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='d_h3/batch_norm')
        if input_txt is not None:
            net_txt = InputLayer(input_txt,name='d_txt')
            net_txt = DenseLayer(net_txt, n_units=64,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init = w_init, b_init=None, name='d_reduce_text/dense')
            net_txt = ExpandDimsLayer(net_txt,1,name='d_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt,1,name='d_txt/expanddim2')
            net_txt = TileLayer(net_txt,[1,4,4,1],name='d_txt/tile')
            net_h4 = ConcatLayer([net_h3,net_txt],concat_dim=3,name='d_h4/concat')
            net_h4 = Conv2d(net_h4,df_dim*4,(1,1),(1,1),padding='VALID',W_init=w_init,name='d_h4/conv2d')
            net_h3 = BatchNormLayer(net_h4,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='d_h4/batch_norm')
        net_ho = Conv2d(net_h3,1,(s16,s16),(s16,s16),padding='VALID',W_init=w_init,name='d_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho,logits

### generator and discriminator for hd images ###
"""
I will first try generating without captions. If it doesn't work, I'll try adding captions
"""

def hd_generator(input_img,input_txt=None, is_train=True,reuse=False,batch_size=batch_size):
    gf_dim = 32
    lrelu = tf.nn.relu
    
    with tf.variable_scope("hdgen",reuse=reuse): #Check later whether it's still "generator"
        tl.layers.set_name_reuse(reuse)
        
        net_txt = InputLayer(input_txt,name='hg_input_txt')
        net_in = InputLayer(input_img,name='hg_input_img')
        # 64*64*1
        net_in = Conv2d(net_in,int(gf_dim/4),(3,3),strides=(1,1),padding='SAME',act=lrelu,W_init=w_init,name='hg_input/conv2d1')
        net_in = BatchNormLayer(net_in,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_input/batch_norm1')
        net_in = Conv2d(net_in,int(gf_dim/2),(3,3),strides=(2,2),padding='SAME',act=None,W_init=w_init,name='hg_input/conv2d2')
        net_in = BatchNormLayer(net_in,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_input/batch_norm2')
        net_in = Conv2d(net_in,gf_dim,(3,3),strides=(2,2),padding='SAME',act=None,W_init=w_init,name='hg_input/conv2d3')
        net_in = BatchNormLayer(net_in,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_input/batch_norm3')
        print(np.shape(net_in.outputs))
        # 32*32*gf_dim
        net_res = Conv2d(net_in,gf_dim,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='hg_input/res1')
        net_res = BatchNormLayer(net_res,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_input/res_batch_norm1')
        net_res = Conv2d(net_res,gf_dim,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='hg_input/res2')
        net_res = BatchNormLayer(net_res,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_input/res_batch_norm2')
        print(np.shape(net_res.outputs))
        net_in = ElementwiseLayer(layer=[net_res,net_in],combine_fn=tf.add,name='hg_input/add')
        # 16*16*(gf_dim*2)
        #net_in = MaxPool2d(net_in,filter_size=(2, 2),strides=None,padding='SAME',name='hg_input/maxpool1')
        #net_in = MaxPool2d(net_in,filter_size=(2, 2),strides=None,padding='SAME',name='hg_input/maxpool2')
        #net_in = Conv2d(net_in,1,(5,5),strides=(4,4),padding='SAME',act=None,W_init=w_init,name='hg_input/conv2d')
        #net_in = BatchNormLayer(net_in,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_input/batch_norm')
        #net_in = FlattenLayer(net_in,name='hg_input/flatten')
        net_txt = DenseLayer(net_txt, n_units=t_dim,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init = w_init, b_init=None, name='hg_reduce_text/dense')
        net_txt = ExpandDimsLayer(net_txt,1,name='hg_txt/expanddim1')
        net_txt = ExpandDimsLayer(net_txt,1,name='hg_txt/expanddim2')
        net_txt = TileLayer(net_txt,[1,16,16,1],name='hg_txt/tile')

        net_in = ConcatLayer([net_in,net_txt],concat_dim=3,name='hg_concat')
        
        
        #net_h0 = DenseLayer(net_in,n_units = gf_dim*16*16*4,W_init=w_init,act=tf.identity,name='hg_h0/dense')#got question about n_units
        #net_h0 = ReshapeLayer(net_h0,shape=[-1,16,16,gf_dim*4],name='hg_h0/reshape')
        #net_h0 = BatchNormLayer(net_h0,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_h0/batch_norm')

        ## net_h0: 1 denconv/ 2 conv
        net_h1 = DeConv2d(net_in,gf_dim*4,(3,3),out_size=(32,32),strides=(2,2),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='hg_h1/deconv2d')
        net_h1 = Conv2d(net_h1,gf_dim*4,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='hg_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_h1/batch_norm')
        
        net_h2 = DeConv2d(net_h1,gf_dim*4,(3,3),out_size=(64,64),strides=(2,2),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='hg_h2/deconv2d')
        net_h2_1 = Conv2d(net_h2,gf_dim*2,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='hg_h2_1/conv2d')
        net_h2_1 = BatchNormLayer(net_h2_1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_h2_1/batch_norm')
        net_h2_2 = Conv2d(net_h2,gf_dim*2,(5,5),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='hg_h2_2/conv2d')
        net_h2_2 = BatchNormLayer(net_h2_2,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_h2_2/batch_norm')
        net_h2 = ConcatLayer([net_h2_1,net_h2_2],concat_dim=3,name='hg_h2/concat')

        net_h3 = DeConv2d(net_h2,gf_dim*2,(3,3),out_size=(128,128),strides=(2,2),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='hg_h3/deconv2d')
        net_h3 = Conv2d(net_h3,gf_dim,(3,3),strides=(1,1),padding='SAME',act=None,W_init=w_init,name='hg_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hg_h3/batch_norm')
        ## net_h1: 1 deconv
        net_ho = DeConv2d(net_h3,c_dim,(3,3),out_size=(256,256),strides=(2,2),padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='hg_ho/deconv2d')
        print(np.shape(net_ho.outputs))
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
        print(np.shape(net_ho.outputs))
        print(np.shape(logits))
    return net_ho,logits

def hd_discriminator(input_image,input_txt,is_train=True,reuse=False):
    df_dim = 8
    lrelu = lambda x:tl.act.lrelu(x,0.2)

    with tf.variable_scope("hddis",reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # 256*256
        print(np.shape(input_image))
        net_in = InputLayer(input_image,name='hd_in')
        print(np.shape(net_in.outputs))
        net_h0 = Conv2d(net_in, df_dim*2,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h0/conv2d')
        print(np.shape(net_h0.outputs))
        net_h0 = BatchNormLayer(net_h0,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h0/batch_norm')
        # 128*128
        net_h1 = Conv2d(net_h0,df_dim*2, (3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h1/conv2d')
        print(np.shape(net_h0.outputs))
        net_h1 = BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h1/batch_norm')
        # 64*64
        net_h2 = Conv2d(net_h1, df_dim*2, (3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h2/batch_norm')
        net_h2_1 = Conv2d(net_h1, df_dim*2, (5,5),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h2_1/conv2d')
        net_h2_1 = BatchNormLayer(net_h2_1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h2_1/batch_norm')
        net_h2 = ConcatLayer([net_h2_1,net_h2],concat_dim=3,name='hd_h2/concat')
        # 32*32
        net_h3 = Conv2d(net_h2,df_dim*4,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h3/batch_norm')
        # 16*16
        net_h4 = Conv2d(net_h3,df_dim*8,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h4/batch_norm')
        # 8*8
        net_h5 = Conv2d(net_h4,df_dim*8,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h5/batch_norm')
        ## resnet for discriminator
        net_res = Conv2d(net_h5,df_dim*8,(3,3),(1,1),act=None,padding='SAME',W_init=w_init,name='hd_h5/res_conv2d1')
        net_res = BatchNormLayer(net_res,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h5/res_batch_norm1')
        net_res = Conv2d(net_res,df_dim*8,(3,3),(1,1),act=None,padding='SAME',W_init=w_init,name='hd_h5/res_conv2d2')
        net_res = BatchNormLayer(net_res,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h5/res_batch_norm2')
        net_h5 = ElementwiseLayer(layer=[net_h5,net_res],combine_fn=tf.add,name='hd_h5/add')
        if input_txt is not None:
            net_txt = InputLayer(input_txt,name='hd_txt')
            net_txt = DenseLayer(net_txt, n_units=64,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init = w_init, b_init=None, name='hg_reduce_text/dense')
            net_txt = ExpandDimsLayer(net_txt,1,name='hd_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt,1,name='hd_txt/expanddim2')
            net_txt = TileLayer(net_txt,[1,4,4,1],name='hd_txt/tile')
            net_h6 = ConcatLayer([net_h5,net_txt],concat_dim=3,name='hd_h6/concat')
            net_h6 = Conv2d(net_h6,df_dim*2,(3,3),(1,1),padding='VALID',W_init=w_init,name='hd_h6/conv2d')
            #print(np.shape(net_h6.outputs))
            net_h6 = BatchNormLayer(net_h6,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h6/batch_norm')
        net_ho = Conv2d(net_h6,1,(2,2),(2,2),padding='VALID',W_init=w_init,name='hd_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho,logits
    
def hd_discriminator_local(input_image,input_txt,is_train=True,reuse=False):
    df_dim = 8
    lrelu = lambda x:tl.act.lrelu(x,0.2)

    with tf.variable_scope("hddis",reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # 256*256
        print(np.shape(input_image))
        net_in = InputLayer(input_image,name='hd_in')
        print(np.shape(net_in.outputs))
        net_h0 = Conv2d(net_in, df_dim*2,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h0/conv2d')
        print(np.shape(net_h0.outputs))
        net_h0 = BatchNormLayer(net_h0,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h0/batch_norm')
        # 128*128
        net_h1 = Conv2d(net_h0,df_dim*2, (3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h1/conv2d')
        print(np.shape(net_h0.outputs))
        net_h1 = BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h1/batch_norm')
        # 64*64
        net_h2 = Conv2d(net_h1, df_dim*2, (3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h2/batch_norm')
        net_h2_1 = Conv2d(net_h1, df_dim*2, (5,5),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h2_1/conv2d')
        net_h2_1 = BatchNormLayer(net_h2_1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h2_1/batch_norm')
        net_h2 = ConcatLayer([net_h2_1,net_h2],concat_dim=3,name='hd_h2/concat')
        # 32*32
        net_h3 = Conv2d(net_h2,df_dim*4,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h3/batch_norm')
        # 16*16
        net_h4 = Conv2d(net_h3,df_dim*8,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h4/batch_norm')
        # 8*8
        net_h5 = Conv2d(net_h4,df_dim*8,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='hd_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h5/batch_norm')
        ## resnet for discriminator
        net_res = Conv2d(net_h5,df_dim*8,(3,3),(1,1),act=None,padding='SAME',W_init=w_init,name='hd_h5/res_conv2d1')
        net_res = BatchNormLayer(net_res,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h5/res_batch_norm1')
        net_res = Conv2d(net_res,df_dim*8,(3,3),(1,1),act=None,padding='SAME',W_init=w_init,name='hd_h5/res_conv2d2')
        net_res = BatchNormLayer(net_res,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h5/res_batch_norm2')
        net_h5 = ElementwiseLayer(layer=[net_h5,net_res],combine_fn=tf.add,name='hd_h5/add')
        if input_txt is not None:
            net_txt = InputLayer(input_txt,name='hd_txt')
            net_txt = DenseLayer(net_txt, n_units=64,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init = w_init, b_init=None, name='hg_reduce_text/dense')
            net_txt = ExpandDimsLayer(net_txt,1,name='hd_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt,1,name='hd_txt/expanddim2')
            net_txt = TileLayer(net_txt,[1,4,4,1],name='hd_txt/tile')
            net_h6 = ConcatLayer([net_h5,net_txt],concat_dim=3,name='hd_h6/concat')
            net_h6 = Conv2d(net_h6,df_dim*2,(3,3),(1,1),padding='VALID',W_init=w_init,name='hd_h6/conv2d')
            #print(np.shape(net_h6.outputs))
            net_h6 = BatchNormLayer(net_h6,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='hd_h6/batch_norm')
        net_ho = Conv2d(net_h6,1,(2,2),(2,2),padding='VALID',W_init=w_init,name='hd_ho/conv2d')
        logits = net_ho.outputs
        net_local1,disc_hd_local_logits_1 = local_discriminator1(input_image[0:batch_size, 48:112,90:154,0:1],input_txt[0:batch_size,10:18],is_train=True,reuse=reuse) #眼睛
        _,disc_hd_local_logits_2 = local_discriminator1(input_image[0:batch_size,144:208,90:154,0:1],input_txt[0:batch_size,10:18],is_train=True,reuse=True) #眼睛
        # local nose loss
        _,disc_hd_local_logits_3 = local_discriminator1(input_image[0:batch_size,96:160,120:184,0:1],input_txt[0:batch_size,28:36],is_train=True,reuse=True) #鼻子
        # local mouth loss
        _,disc_hd_local_logits_4 = local_discriminator1(input_image[0:batch_size,96:160,174:238,0:1],input_txt[0:batch_size,22:30],is_train=True,reuse=True) #嘴巴
        #  local ears loss
        # _,disc_hd_local_logits_5 = local_discriminator1(input_image[0:batch_size,0:64,134:198,0:1],input_txt[0:batch_size,33:41],is_train=True,reuse=True) #耳朵
        # _,disc_hd_local_logits_6 = local_discriminator1(input_image[0:batch_size,192:256,134:198,0:1],input_txt[0:batch_size,33:41],is_train=True,reuse=True) #耳朵
        # logits_all = 0.6*logits + 0.4*(disc_hd_local_logits_1 + disc_hd_local_logits_2 + disc_hd_local_logits_3 + disc_hd_local_logits_4 + disc_hd_local_logits_5+disc_hd_local_logits_6)/6
        logits_all = 0.7*logits + 0.3*(disc_hd_local_logits_1 + disc_hd_local_logits_2 + disc_hd_local_logits_3 + disc_hd_local_logits_4)/4
        net_ho.outputs = tf.nn.tanh(logits_all)
    return net_ho,logits_all


def local_discriminator1(input_image,input_txt=None,is_train=True,reuse=False):
    t_dim = 30
    df_dim = 64
    lrelu = lambda x:tl.act.lrelu(x,0.2)
    w_init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope("local_discriminator1",reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_image,name='local_in')
        net_h0 = Conv2d(net_in, df_dim, (5,5),(2,2),act=lrelu,padding='SAME',W_init=w_init,name='local_h0/conv2d')
        
        net_h1 = Conv2d(net_h0,df_dim,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='local_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='local_h1/batch_norm')

        net_h2 = Conv2d(net_h1,df_dim*2,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='local_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='local_h2/batch_norm')

        net_h3 = Conv2d(net_h2,df_dim*2,(3,3),(2,2),act=None,padding='SAME',W_init=w_init,name='local_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='local_h3/batch_norm')
        if input_txt is not None:
            net_txt = InputLayer(input_txt,name='local_txt')
            net_txt = DenseLayer(net_txt, n_units=30,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init = w_init, b_init=None, name='local_reduce_text/dense')
            net_txt = ExpandDimsLayer(net_txt,1,name='local_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt,1,name='local_txt/expanddim2')
            net_txt = TileLayer(net_txt,[1,4,4,1],name='local_txt/tile')
            net_h4 = ConcatLayer([net_h3,net_txt],concat_dim=3,name='local_h4/concat')
            net_h4 = Conv2d(net_h4,df_dim*4,(1,1),(1,1),padding='VALID',W_init=w_init,name='local_h4/conv2d')
            net_h3 = BatchNormLayer(net_h4,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='local_h4/batch_norm')
        net_ho = Conv2d(net_h3,1,(s16,s16),(s16,s16),padding='VALID',W_init=w_init,name='local_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho,logits