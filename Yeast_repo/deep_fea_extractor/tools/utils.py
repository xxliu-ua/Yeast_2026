import cv2
from tqdm import tqdm
import numpy as np
import random
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CustomInception3(nn.Module):
    def __init__(self, inception, num_classes=1000, aux_logits=False, transform_input=False, final_pooling=None):
        super(CustomInception3, self).__init__()
        self.final_pooling = final_pooling
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        ## we turn off auxiliary
        # if self.training and self.aux_logits:
        #    aux = self.AuxLogits(x)

        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)  # size (batch_size, 2048, 1, 1)
        # 1 x 1 x 2048

        ## We'll save average pooling over the last conv output, but turn off the last FC layer

        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)
        # 2048
        # x = self.fc(x)

        ## turn off aux output
        # 1000 (num_classes)
        # if self.training and self.aux_logits:
        #    return x, aux

        if self.final_pooling:
            x = F.avg_pool1d(x.view(x.size(0), 2048, 1).permute(0, 2, 1), kernel_size=self.final_pooling)

        return x

class Encoder(nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        densnet = model
        self.feature = densnet.features
        self.classifier = nn.Sequential(*list(densnet.classifier.children())[:-1])
        pretrained_dict = densnet.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)

    def forward(self, x):
        output = self.feature(x)
        relu = nn.ReLU(inplace=True)
        output = relu(output)
        avg = nn.AvgPool2d(7, stride=1)
        output = avg(output)
        return output

class DenseEncoder(nn.Module):
    def __init__(self, model, fixed_extractor=True):
        super(DenseEncoder, self).__init__()
        original_model = model
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])

        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs



def compute_img_mean_std(image_lists):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    img_list = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_lists))):
        img = cv2.imread(image_lists[i])
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

        # img = cv2.imread(image_lists[i], cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (img_h, img_w))
        # img1 = cv2.GaussianBlur(img, (5, 5), 0)
        # imgx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        # imgy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        # imgx = cv2.convertScaleAbs(imgx)
        # imgy = cv2.convertScaleAbs(imgy)
        # imgxy = cv2.addWeighted(imgx, 0.5, imgy, 0.5, 0)
        # H, W = img.shape
        # new_img = np.zeros([H, W, 3], dtype=np.uint8)
        # new_img[:, :, 0] = img
        # new_img[:, :, 1] = img1
        # new_img[:, :, 2] = imgxy
        # imgs.append(new_img)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    #
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(pixels.mean())
        stdevs.append(pixels.std())


    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    means = torch.Tensor(means)
    stdevs = torch.Tensor(stdevs)
    return means, stdevs


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_



        return tensor

def inverse_freq(label):
    # binary label for 0 or 1
    # weighting each class by inverse frequency
    label = torch.from_numpy(label)
    sum_1 = label.sum()  # weight for class 0
    N = len(label)
    alpha = sum_1/N
    return torch.FloatTensor([alpha, 1-alpha]).cuda()


GLOBAL_SEED = 1
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # not affect much of the precision
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
     global GLOBAL_WORKER_ID
     GLOBAL_WORKER_ID = worker_id
     set_seed(GLOBAL_SEED + worker_id)


def worker_init_fn0(worker_id):
    set_seed(1)
