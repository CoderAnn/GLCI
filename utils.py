# -*- coding: UTF-8 -*-
from config import *
import numpy as np
from PIL import Image
import math

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.ones([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)


def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h
        
        patch = np.ones([(q2+1-q1), (p2+1-p1),3])
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE,3), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1,:] = patch
        mask.append(m)
    return np.array(points), np.array(mask)

def get_patchs(batch, X, Y):
    return batch[:, X:X+MASK_H, Y:Y+MASK_W, :]

def read_img_and_crop(path):
    img = np.array(Image.open(path))
    if np.shape(img).__len__() < 3:
        img = np.dstack((img, img, img))
    h = img.shape[0]
    w = img.shape[1]
    if h > w:
        diff = h - w
        random_y = np.random.randint(0, diff)
        return img[random_y:random_y+w, :]
    elif h < w:
        diff = w - h
        random_x = np.random.randint(0, diff)
        return img[:, random_x:random_x+h]
    else:
        return img
