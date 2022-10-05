#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 13:22:46 2022

@author: leonzion
"""

from fc.yolo_leon import *
from PIL import Image
import datetime
from norfair import Detection, Tracker
from torchvision.ops import nms, roi_align, roi_pool
import torch.nn.functional as F





# 获取时间间隔|秒
# starttime = datetime.datetime.now()
# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)

# 图片推理
yo = Yolo_infer()
img = Image.open(r'/home/leonzion/图片/Test/mans_0.jpg')
yolo_output = yo.infer(img)

