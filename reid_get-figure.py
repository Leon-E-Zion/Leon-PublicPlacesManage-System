#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 12:58:46 2022

@author: leonzion
"""
import sys
if sys.platform == 'win32':
    sys.path.append(r'D:\Leon-Coding\Leon_FC')
else:
    sys.path.append(r'/home/leonzion/Leon_Coding/Leon_FC')
from leon_info import *
from leon_os import *


import torch
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import os

leon_path_in(os.path.join('reid','Person_reID_baseline_pytorch-master','model','ft_ResNet50'))
leon_path_in(os.path.join('reid','Person_reID_baseline_pytorch-master'))
def get_fea(path):
    img = cv2.imread(path)
    img = torch.tensor(img.transpose(2, 0, 1))
    img = torch.unsqueeze(img, dim = 0)
    img = torch.tensor(img).to(torch.float32)
    net = torch.load('D:/Leon-Coding/Leon-PublicPlacesManage-System/models/reid-effi_b0.pth_net')
    outputs = net(img) 
    # ---- L2-norm Feature ------
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff

if 1:
    ff = get_fea(r'D:\Leon-Coding\Leon_TestData\human_2.jpg')
    ff1 = get_fea(r'D:\Leon-Coding\Leon_TestData\human_0.jpg')
    print(F.cosine_similarity(ff1,ff,dim=1))