# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :Vit
# @File     :main
# @Date     :2022/1/18 16:37
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from models import PatchEmbedding

if __name__ == '__main__':
    img = Image.open('./dog.jpg')

    fig = plt.figure()
    # plt.imshow(img)
    # plt.show()

    # resize to imagenet size
    x = ToTensor()(img)
    x = Resize((224, 224))(x)
    x = x.unsqueeze(0)  # add batch dim
    print(x.shape)
    patches = PatchEmbedding()(x)
    print(patches.shape)