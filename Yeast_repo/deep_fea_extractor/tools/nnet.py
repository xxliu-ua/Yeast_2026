import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tools.utils import CustomInception3


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True




def initialize_model(model_name, num_classes=2, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        # you need to set the following content if you want to train
        # a CNN classifier for a custom dataset
        # num_ftrs = model_ft.fc.in_features
        # num_inter = 256
        # model_ft.fc = torch.nn.Sequential(nn.Linear(num_ftrs, num_inter),
        #                             nn.ReLU(),
        #                             nn.Linear(num_inter, num_inter),
        #                             nn.ReLU(),
        #                             nn.Linear(num_inter, num_classes)
        #                             )
        # for name, value in model_ft.named_parameters():
        #     if name == 'fc':
        #         value.requires_grad = True
        # train_list = ['fc']
        # for name, child in model_ft.named_children():
        #     if name in train_list:
        #         for para in child.parameters():
        #             para.requires_grad = True
        #     else:
        #         for para in child.parameters():
        #             para.requires_grad = False
        input_size = (224, 224)


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = (224, 224)


    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        # num_inter = 256
        # model_ft.classifier = nn.Sequential(
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(num_ftrs, num_inter),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(num_inter, num_inter),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(num_inter, num_classes)
        # )
        #
        # train_list = ['classifier']
        # for name, child in model_ft.named_children():
        #     if name in train_list:
        #         for para in child.parameters():
        #             para.requires_grad = True

        input_size = (224, 224)

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        # # for classification
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # # Handle the primary net
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs,num_classes)

        input_size = (299, 299)
    elif model_name == "resnext":
        model_ft = models.resnext50_32x4d(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = (224, 224)
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size