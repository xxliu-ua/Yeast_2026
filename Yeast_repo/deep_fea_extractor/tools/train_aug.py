import cv2
import random
import numpy as np
import torch
import os
import glob
import math
from PIL import Image
from tools.testAug import rotate

# data augmentation technique
# generate 5 augmented images by rotation 90, 180, 270, 360 flip horizontally and vertically
# the original image name is XXXXX.png, the resize and augmented images are named as
# XXXXX_0.png, XXXXX_1.png, ..., XXXXX_5.png
# however, one should be careful in generating images since these generated images may not make sense
# we did not use data augmentation for the previous papers



img_dir = '/media/lina/OS/ISBI2017/orig_data/orig_{}_data'
save_dir = '/media/lina/OS/ISBI2017/orig_data/aug_train_data'
train_img_dir = img_dir.format('train')
img_list = glob.glob(train_img_dir + '/*.jpg')

N = len(img_list)
img_ids = [img_list[i][-16:-4] for i in range(N)]
target_size = (224, 224)

for i in range(N):
    orig_img = cv2.imread(img_list[i])
    orig_img = cv2.resize(orig_img, target_size)
    cv2.imwrite(save_dir + '/' + img_ids[i] + '_0.png', orig_img)
    img, _ = rotate(orig_img, None, 90, color_mode='RGB')
    cv2.imwrite(save_dir + '/' + img_ids[i] + '_1.png', img)
    img, _ = rotate(orig_img, None, 180, color_mode='RGB')
    cv2.imwrite(save_dir + '/' + img_ids[i] + '_2.png', img)
    img, _ = rotate(orig_img, None, 270, color_mode='RGB')
    cv2.imwrite(save_dir + '/' + img_ids[i] + '_3.png', img)
    img = cv2.flip(orig_img, 1)
    cv2.imwrite(save_dir + '/' + img_ids[i] + '_4.png', img)
    img = cv2.flip(orig_img, 0)
    cv2.imwrite(save_dir + '/' + img_ids[i] + '_5.png', img)
