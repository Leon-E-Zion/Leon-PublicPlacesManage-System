#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 12:58:46 2022

@author: leonzion
"""

import torch
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F

def get_fea(path):
    img = cv2.imread(path)
    img = torch.tensor(img.transpose(2, 0, 1))
    img = torch.unsqueeze(img, dim = 0)
    img = torch.tensor(img).to(torch.float32)
    net = torch.load('net_19.pth_net')
    outputs = net(img) 
    # ---- L2-norm Feature ------
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff

ff = get_fea(r'/home/leonzion/图片/Test/human_2.jpg')
ff1 = get_fea(r'/home/leonzion/图片/Test/human_0.jpg')

print(F.cosine_similarity(ff1,ff,dim=1))