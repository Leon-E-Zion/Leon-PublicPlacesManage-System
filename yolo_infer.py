#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 13:22:46 2022

@author: leonzion
"""

from tools.yolo_leon import *
from PIL import Image
import datetime
from torchvision.ops import nms, roi_align, roi_pool
import torch.nn.functional as F



if 1>2 :
    # 图片推理
    yo = Yolo_infer()
    img = Image.open(r'D:\Leon-Coding\Leon_TestData\mans_0.jpg')
    yolo_output = yo.infer(img)
    # print(yolo_output)
