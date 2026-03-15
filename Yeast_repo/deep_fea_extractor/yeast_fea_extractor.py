# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 04:08:00 2023

@author: xxliu
"""
# import python & pytorch packages
import os
import torch
import time
from tools.utils import set_seed
import pandas as pd
from PIL import Image
import cv2

set_seed(1)

from torchvision import transforms
import tracemalloc

# import self-defined functions
from tools.utils import DenseEncoder, compute_img_mean_std
from tools.nnet import initialize_model
from tools.testAug import testaug

import winsound


tracemalloc.start()

# # for single GPU case
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # if there is only one GPU set it '0'; ['0', '1']
gpu_id = 0                                  # if only one gpu is available set it '0'; ['0', '1']


duration = 100  # milliseconds
freq = 440  # Hz

#################### parameter settings #######################################
testAug = False                         # True/False whether use data augmentation
model_name = 'densenet'                 # pretrained CNN used
num_classes = 2                         # number of classes； only use it when training cnn
batch_size = 12                         # batch size for training


img_dir = '/Users/username/.../image_directory/'  # directory of the image dataset of the chosen species
save_name = '/Users/username/.../save_directory/code/fea/yeast/yeast_species_features.csv'   # save name for the features

################################################################################
# calculate the mean and standard deviation values of the images
# to calculate the mean and standard deviation, we need to have image list of the entire training dataset

img_list = os.listdir(img_dir)
img_list = [img_dir + i for i in img_list]

# for calculated means for each class
means, stds = compute_img_mean_std(img_list) # in RGB order
print((means, stds))


# get the pretrained CNN model for feature extraction
model, input_size = initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True)

# for feature extraction, it does not require to calculate the gratitude, and we need to set it False
for param in model.parameters():
    param.requires_grad = False

# if there is a GPU, use cuda for calculation
# otherwise use CPU
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
model.eval()                # for feature extraction, change the model to evaluation mode
model.to(device)

# define the features we will use
fea_extractor = DenseEncoder(model)

# transformation for the dataset
val_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)])


# extract  features
start_time = time.perf_counter()

for i in range(len(img_list)):
    # if str(wavelength) in img_list1[i]:
    img = cv2.imread(img_list[i],cv2.IMREAD_GRAYSCALE) # opencv in BGR order
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    new_img = testaug(img, None, val_transform,test_aug=testAug)
    fea = fea_extractor(new_img.to(device))


    if isinstance(fea, list):
        fea = fea[0]
    fea = fea.to(torch.device('cpu')).numpy().squeeze()

    if testAug is True:
        fea = fea.reshape(6,-1)
    else:
        fea = fea.reshape(1, -1)

    data = pd.DataFrame(fea)
    data.to_csv(save_name, sep='\t', mode='a',index=0, header=0)

# print(i)
stop_time = time.perf_counter()
print(stop_time - start_time)

winsound.Beep(freq, duration)

print(tracemalloc.get_traced_memory())
tracemalloc.stop()