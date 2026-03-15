import torch, cv2
import os
import sys
import numpy as np
import glob
import csv
import pandas as pd
from PIL import Image
from tools.testAug import testaug
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import random


# this dataloader is only for classification task
class ISBI2017Seg(Dataset):
    category_names = ['nevus', 'seborrheic keratosis', 'melanoma']
    root_dir = '/media/lina/OS/ISBI2017/orig_data'

    #root_dir = '/home/lina1/project/dataset/ISBI2017/orig_data'

    debug = False
    def __init__(self,
                 root=root_dir,
                 debug = debug,
                 split='val',
                 task = 'mel',
                 transform=None,
                 use_test_ensemble=False):

        self.root = root
        assert (isinstance(split, str))
        self.split = split
        self.task = task
        self.transform = transform
        self.debug = debug
        self.use_test_ensemble = use_test_ensemble

        if self.split == 'train':
            _img_dir = os.path.join(self.root, 'aug_train_data', '*.png')
        else:
            _img_dir = os.path.join(self.root, 'orig_' + self.split + '_data', '*.jpg')
        #_mask_dir = os.path.join(self.root, 'gt_mask_'+ self.split,'*.png')
        _cat_dir = os.path.join(self.root, 'gt_class_'+self.split+'.csv')

        # build the ids file
        data = pd.read_csv(_cat_dir)
        if debug is True:
            print(data.head(5))
            print(data.columns)
            print(data.shape)
        N, _ = data.shape

        self.img_ids = []
        self.img_list = []
        #self.mask_list = []
        #self.cat_list =[]
        self.label_list = []


        for i in range(N):
            temp = data.loc[i,'image_id']
            self.img_ids.append(temp)

            if self.split == 'train':
                for j in range(6):
                    self.img_list.append(os.path.join(self.root, 'aug_train_data', temp + '_' + str(j) +'.png'))
            else:
                self.img_list.append(os.path.join(self.root, 'orig_'+self.split+'_data', temp+'.jpg'))

            #self.mask_list.append(os.path.join(self.root, 'gt_mask_' + self.split, temp+'_segmentation.png'))
            # cat = data.loc[i,['melanoma','seborrheic_keratosis']]
            # if np.logical_and(cat['melanoma']==0,cat['seborrheic_keratosis']==0):
            #     self.cat_list.append(0)
            # elif cat['seborrheic_keratosis']==1:
            #     self.cat_list.append(1)
            # elif cat['melanoma']==1:
            #     self.cat_list.append(2)
            # else:
            #     raise Exception('invalid category for id: '+ temp)

            if self.task == 'mel':
                label = data.loc[i, 'melanoma']
            elif self.task == 'sk':
                label = data.loc[i, 'seborrheic_keratosis']

            if self.split == 'train':
                for j in range(6):
                    self.label_list.append(label)
            else:
                self.label_list.append(label)

        assert (len(self.label_list)==len(self.img_list))
        print("Number of images: {} for the {} dataset\n".format(len(self.img_ids), self.split))


    def __getitem__(self, index):

         _img = Image.open(self.img_list[index]).convert('RGB')
         # _mask = Image.open(self.mask_list[index])
         # _mask = _mask.resize((_img.shape[1], _img.shape[0]), Image.ANTIALIAS)
         # _mask = np.array(_mask).astype(np.float32)
         # _mask[_mask>127] = 255
         # _mask[_mask<=127] = 0
         _label = self.label_list[index]
         _label = torch.tensor(int(_label))

         if self.use_test_ensemble is True:
             _img = testaug(_img, None, self.transform, self.use_test_ensemble)

         else:
             if self.transform is not None:
                 _img = self.transform(_img )


         sample = {'image':_img, 'label': _label}
         return sample



    def __len__(self):
        return len(self.img_list)


class ScatterDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # image = Image.open(self.filenames[idx]).convert('RGB')
        img = cv2.imread(self.filenames[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(img)
        image = self.transform(image)
        sample = {'image':image, 'label': self.labels[idx]}
        return sample


def split_Train_Val_Data(data_dir, ratio):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)  # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    # print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)
    # print(dataset.samples)
    # shuffle the data

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):  # data为一类图片

        num_sample_train = int(len(data) * ratio[0])

        num_sample_val = len(data) - num_sample_train
        num_val_index = num_sample_train + num_sample_val
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
    print("No of train images: {}".format(len(train_inputs)))
    print("No of val images: {}".format(len(val_inputs)))
    shuffle = True
    if shuffle is True:
        random.seed(0)
        temp = list(zip(train_inputs, train_labels))
        random.shuffle(temp)
        train_inputs, train_labels = zip(*temp)
        random.seed(1)
        temp = list(zip(val_inputs, val_labels))
        random.shuffle(temp)
        val_inputs, val_labels = zip(*temp)

    return train_inputs, train_labels, val_inputs, val_labels

