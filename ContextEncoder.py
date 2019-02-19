# -*- coding: UTF-8 -*-
import tensorflow as tf
from networks import generator, discriminator
from utils import *
from operate_data import *
import numpy as np
from PIL import Image
import scipy.misc as misc
import os
    
class ContextEncoder:
    def __init__(self,data):
        # Paper: Context Encoders: Feature Learning by Inpainting
        self.batch_size=BATCH_SIZE 
        self.sess = tf.Session()
        self.data = data  #inputs真实图像
        self.step = tf.Variable(0, name='global_step', trainable=False)
        
        coord, threads = self.queue_context()
        self.build_model()
        
    def queue_context(self):
        # thread coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        return coord, threads
    
    def build_model(self):
        self.local_batch = tf.placeholder(tf.float32, [BATCH_SIZE, MASK_H, MASK_W, IMG_C])
        self.train_phase = tf.placeholder(tf.bool)    
        
        self.inputs_raw=self.data
        self.inputs=norm_img(self.inputs_raw)

        G = generator()
        D = discriminator()
        
        points_batch, mask = get_points()
                
        input_batch = self.inputs*(1-mask)
        imitation , self.G_var = G(input_batch, reuse=False) #imitation是补全后的图像
        completion = imitation * mask + input_batch#completion是真正的补全后的图像
        
        self.completion=denorm_img(completion)
        self.imitation=denorm_img(imitation)
        self.input_batch=denorm_img(input_batch)
        self.input_img=denorm_img(self.inputs)

        self.patch=[]
        self.local=[]
        for k in range(self.batch_size):
            x1, y1, x2, y2 = points_batch[k]
            patch_batch = tf.image.resize_images(tf.image.crop_to_bounding_box(self.inputs[k],y1,x1,64,64),[64,64])
            local_batch = tf.image.resize_images(tf.image.crop_to_bounding_box(completion[k],y1,x1,64,64),[64,64])
            self.patch.append(patch_batch)
            self.local.append(local_batch)
        self.patch = tf.convert_to_tensor(self.patch)
        self.local = tf.convert_to_tensor(self.local)        
#        之前是将获取蒙版部分在训练时再处理，现在将其放入模型构造中,并且更改了学习率以及超参数

        self.real_logits , self.D_var = D(self.inputs,self.patch,self.train_phase)
        self.fake_logits , _ = D(completion,self.local,self.train_phase)
        
        label_real = tf.ones(self.batch_size)
        label_fake = tf.zeros(self.batch_size)

        self.G_loss_mse = tf.reduce_mean(tf.square(self.inputs - imitation))#mse均方误差
        self.G_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=label_real))                   
        self.G_loss_all = self.G_loss_mse + 0.0004*self.G_loss_gan
        
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits, labels=label_real))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=label_fake))
        self.D_loss = self.D_loss_fake + self.D_loss_real
                
        opt = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.5)        
        self.D_Opt = opt.minimize(self.D_loss, var_list=self.D_var)
        self.G_Opt_mse = opt.minimize(self.G_loss_mse, global_step=self.step,var_list=self.G_var)
        self.G_Opt = opt.minimize(self.G_loss_all,global_step=self.step, var_list=self.G_var)
        
        self.sess.run(tf.global_variables_initializer())

    def train(self):                               
        saver = tf.train.Saver()               
        for i in range(20001):            
            #只训练补全网络
            if i <= Tc:
                self.sess.run(self.G_Opt_mse, feed_dict={self.train_phase: True})
                if i % 100 == 0:
                    G_loss_mse=self.sess.run(self.G_loss_mse,feed_dict={self.train_phase: False})    
                    G_loss_gan=self.sess.run(self.G_loss_gan,feed_dict={self.train_phase: False})
                    D_loss=self.sess.run(self.D_loss,feed_dict={self.train_phase: False})
                    print("Epoch: [%d], G_loss_mse:  [%.4f],G_loss_gan:  [%.4f],D_loss:  [%.4f]" % (i,G_loss_mse,G_loss_gan,D_loss))
                if i % 2000 == 0:
                    com_img,imi_img,input_img = self.sess.run([self.completion,self.imitation,self.input_batch], feed_dict={self.train_phase: False})
                    path = os.path.join('Results', '{}completion.png'.format(i))
                    save_image(com_img, path)
                    path = os.path.join('Results', '{}imitation.png'.format(i))
                    save_image(imi_img, path)
                    path = os.path.join('Results', '{}input.png'.format(i))
                    save_image(input_img, path)

            #更新鉴别器
            else:
                self.sess.run(self.D_Opt, feed_dict={self.train_phase: True})                    
                if i % 100 == 0:
                    D_loss=self.sess.run(self.D_loss, feed_dict={self.train_phase: False})
                    print("Epoch: [%d] , D_loss:  [%.4f]" % (i,D_loss))                        
                if i > Tc+Td:
                    self.sess.run(self.G_Opt, feed_dict={self.train_phase: True})
                    if i % 100 == 0:
                        G_loss_mse=self.sess.run(self.G_loss_mse,feed_dict={ self.train_phase: False})    
                        G_loss_gan=self.sess.run(self.G_loss_gan,feed_dict={self.train_phase: False})
                        D_loss=self.sess.run(self.D_loss,feed_dict={self.train_phase: False})
                        G_loss_all=self.sess.run(self.G_loss_all,feed_dict={self.train_phase: False})
                        print("Epoch: [%d], G_loss_all:  [%.4f] ,  G_loss_mse:  [%.4f] , G_loss_gan:  [%.4f] , D_loss:  [%.4f]" % (i,G_loss_all,G_loss_mse,G_loss_gan,D_loss))               
                    if i % 2000 == 0:
                        com_img,in_img,input_img,imitation = self.sess.run([self.completion,self.input_img,self.input_batch,self.imitation], feed_dict={self.train_phase: False})
                        path = os.path.join('Results', '{}completion.png'.format(i))
                        save_image(com_img, path)
                        path = os.path.join('Results', '{}real.png'.format(i))
                        save_image(in_img, path)
                        path = os.path.join('Results', '{}input.png'.format(i))
                        save_image(input_img, path)
                        path = os.path.join('Results', '{}imitation.png'.format(i))
                        save_image(imitation, path)
        print("Done.")

if __name__ == "__main__":
    data = get_loader(BATCH_SIZE,split='train')
    CE = ContextEncoder(data)
    CE.train()
