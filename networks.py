# -*- coding: UTF-8 -*-
from ops import *
from config import *
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

#非补全网络问题
class generator:       
    def __call__(self, inputs_img, reuse):
        with tf.variable_scope("G", reuse=reuse) as vs:
            inputs = slim.conv2d(inputs_img, 64, 5, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = slim.conv2d(inputs, 128, 3, 2, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = slim.conv2d(inputs, 128, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = slim.conv2d(inputs, 256, 3, 2, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)        
            inputs = slim.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)           
            inputs = slim.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            
            inputs = tf.contrib.layers.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity,rate=2)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)          
            inputs = tf.contrib.layers.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity,rate=4)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = tf.contrib.layers.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity,rate=8)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = tf.contrib.layers.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity,rate=16)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            
            inputs = slim.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)      
            inputs = slim.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            
            inputs = slim.conv2d_transpose(inputs, 128, 4, 2, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)  
            inputs = slim.conv2d(inputs, 128, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)            
            inputs = slim.conv2d_transpose(inputs, 64, 4, 2, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)  
            inputs = slim.conv2d(inputs, 32, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs) 
            out = slim.conv2d(inputs, 3, 3, 1, activation_fn=tf.nn.tanh)
            
        
        G_var = tf.contrib.framework.get_variables('G')
        return out,G_var
        

class discriminator:
    def __call__(self, inputs, inputs_local, train_phase):
        reuse = len([t for t in tf.global_variables() if t.name.startswith('D')]) > 0
#        这句代码是指如果是第一次运行discriminator则reuse=False
        with tf.variable_scope('D',reuse=reuse):
            local = slim.conv2d(inputs_local, 64, 5, 2, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            local = leaky_relu(local)
            local = slim.conv2d(local, 128, 5, 2, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            local = leaky_relu(local)
            local = slim.conv2d(local, 256, 5, 2, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            local = leaky_relu(local)            
            local = slim.conv2d(local, 512, 5, 2, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            local = leaky_relu(local)
            local = tf.reshape(local, [-1, np.prod([4, 4, 512])])
            local = slim.fully_connected(local, 1024, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            output_l = leaky_relu(local)

            image_g = slim.conv2d(inputs, 64, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)
            image_g = slim.conv2d(image_g, 128, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)
            image_g = slim.conv2d(image_g, 256, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)            
            image_g = slim.conv2d(image_g, 512, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)
            image_g = slim.conv2d(image_g, 512, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)            
            image_g = tf.reshape(image_g, [-1, np.prod([4, 4, 512])])
            image_g = slim.fully_connected(image_g, 1024, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            output_g = leaky_relu(image_g)
            
            output = tf.concat([output_g,output_l],axis=1)
            output = slim.fully_connected(output, num_outputs=1, activation_fn=None)
            output = tf.squeeze(output, -1) 
            
        D_var = tf.contrib.framework.get_variables('D')
        return output,D_var