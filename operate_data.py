# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:34:22 2019

@author: Beauty
"""
import tensorflow as tf
import os
import numpy as np
import scipy.misc as misc
from glob import glob 
from utils import read_img_and_crop
from config import *
 
def norm_img(image):
    image = image/127.5-1
    return image

def denorm_img(norm):
    return tf.clip_by_value((norm + 1)*127.5, 0, 255)

#用队列的方式读取数据
def get_loader(batch_size, split=None, is_grayscale=False, seed=None):
    dataset_path = "./cats_bigger_than_128x128//"
    images = []
    paths = glob(os.path.join(dataset_path, '*.jpg'))
#    tf.train.string_input_producer（）输出字符串到一个输入管道队列。
    filename_queue = tf.train.string_input_producer(list(paths), shuffle=True, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf.image.decode_jpeg(data, channels=3)
    image = tf.image.resize_images(image, [218, 178], method=tf.image.ResizeMethod.AREA)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
	
    queue = tf.train.shuffle_batch(
            [image], batch_size=batch_size,
            num_threads=64, capacity=capacity,
            min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
    
    queue = tf.image.resize_images(queue, [178, 178], method=tf.image.ResizeMethod.AREA)
    q = tf.image.crop_to_bounding_box(queue, 25, 25, 128, 128)
    return tf.to_float(q)