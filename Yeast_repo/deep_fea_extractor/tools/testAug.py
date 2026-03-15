import cv2
import random
import numpy as np
import torch
import os
import glob
import math
from PIL import Image

""" rotate for multi-color space image
    img:        an image    Uint8
    mask:       a mask      Uint8

"""

def rotate(img, mask, angle, color_mode='other'):
    if color_mode == 'RGB':
        img_h, img_w, _ = img.shape
        M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
        new_img = cv2.warpAffine(img, M_rotate, (img_w, img_h))
        new_mask = []
        if mask is not None:
            new_mask = cv2.warpAffine(mask, M_rotate, (img_w, img_h))

    else:
        N, img_h, img_w = img.shape
        M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
        new_img = []
        new_mask = []
        for i in range(N):
            new_img.append(cv2.warpAffine(img[i, :, :], M_rotate, (img_w, img_h)))
            if mask is not None:
                new_mask.append(cv2.warpAffine(mask[i, :, :], M_rotate, (img_w, img_h)))
        new_img = np.array(new_img)
        new_mask = np.array(new_mask)
    return new_img, new_mask


"""
img:    an PIL image    
mask:   a mask      
"""
def testaug(img, mask=None,transform=None,test_aug=False):
    numpy_img = np.array(img)
    numpy_img = np.squeeze(numpy_img)
    if mask is not None:
        mask = mask.numpy()
        mask = np.squeeze(mask)
    #  if PIL image, Channel is the last

    H, W, C = numpy_img.shape
    test_img = []

    # the orig image
    if transform is not None:

        new_img = transform(img)
        new_img = new_img.numpy()

    test_img.append(new_img)

    if test_aug is True:
        # rotate by 90
        new, _ = rotate(new_img, mask, 90)
        test_img.append(new)

        # rotate by 180
        new, _ = rotate(new_img, mask, 180)
        test_img.append(new)

        #rotate by 270
        new, _ = rotate(new_img, mask, 270)
        test_img.append(new)
        # flip
        temp = []
        for i in range(C):
            new = cv2.flip(np.squeeze(new_img[i,:,:]), 1)
            temp.append(new)
        temp = np.array(temp)
        test_img.append(temp)

        # flip
        temp = []
        for i in range(C):
            new = cv2.flip(np.squeeze(new_img[i,:,:]), 0)
            temp.append(new)
        temp = np.array(temp)
        test_img.append(temp)

    test_img = np.array(test_img)
    #test_img = np.squeeze(test_img)
    test_img = torch.from_numpy(test_img)

    return test_img



"""
test_results: N*H*W
"""

def reverse_seg_testaug(test_results):
    test_results = np.squeeze(test_results)
    N, H, W = test_results.shape
    ensemble_results = test_results
    num_ensemble = 6

    for i in range(num_ensemble):
        index_i = np.arange(i, N, num_ensemble, dtype=np.int32)
        data_i = test_results[index_i, :, :]
        if i==1:
            ensemble_results[index_i, :, :], _ = rotate(data_i, None, -90, color_mode='grayscale')
        elif i==2:
            ensemble_results[index_i, :, :], _ = rotate(data_i, None, -180, color_mode='grayscale')
        elif i==3:
            ensemble_results[index_i, :, :], _ = rotate(data_i, None, -270, color_mode='grayscale')
        elif i==4:
            for j in range(len(index_i)):
                ensemble_results[index_i[j], :, :] = cv2.flip(data_i[j, :, :], 1)
        elif i==5:
            for j in range(len(index_i)):
                ensemble_results[index_i[j], :, :] = cv2.flip(data_i[j, :, :], 0)

        # ensemble_results = np.reshape(ensemble_results,[num_ensemble, N/num_ensemble, H, W],order='F')
        # ensemble_results = np.mean(ensemble_results,axis=0)

    return ensemble_results

def reverse_classify_prob(preds, batch_size):
    results = torch.zeros(batch_size, 2)
    for j in range(batch_size):
        index = range(j * 6, (j + 1) * 6)
        index = torch.tensor(index)
        temp = torch.index_select(preds, 0, index)
        temp = temp.mean(dim=0)
        results[j, :] = temp

    return results
