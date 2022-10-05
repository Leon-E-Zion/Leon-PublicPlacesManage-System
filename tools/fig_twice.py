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
# 变量初始化
bboxes = []
fea = yolo_output['fea']

# 获取特征

# for id,key in enumerate(yolo_output['crop']):
#     unit = yolo_output['crop'][key]
#     crop_bbox = unit['crop_bbox']
#     # box = torch.tensor([[id,crop_bbox[0],crop_bbox[1],crop_bbox[2],crop_bbox[3]]]).float().cuda()
#     box = torch.tensor([[id,crop_bbox[0],crop_bbox[1],crop_bbox[2],crop_bbox[3]]]).float().cuda()
#     box = torch.tensor(yolo_output['outputs']).float().cuda()
#     fea_es.append(roi_align(fea,[box], [120,120]))
# 获取特征 - 获取成功
# print(fea_es)
box = torch.tensor(yolo_output['outputs']).float().cuda()
fea_es=roi_align(fea,[box], [120,120],)
print(fea_es)
# 指定特征
# tar = fea_es[8]
print(F.cosine_similarity(fea_es[4].view(-1),fea_es[11].view(-1),dim=0))
# for fe in fea_es:
#     print(F.cosine_similarity(tar,fe))