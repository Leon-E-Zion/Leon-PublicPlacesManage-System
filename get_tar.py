#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 15:18:26 2022

@author: leonzion
"""
import sys
if sys.platform == 'win32':
    sys.path.append(r'D:\Leon-Coding')
else:
    sys.path.append(r'/home/leonzion/Leon_Coding/leon_FC')
from Leon_FC import *



# 运行指定文件
path_0 = leon_path_in(r'yolov5_track/track.py')
path_1 = leon_path_in(r'yolov5_track/yolov5/weights/crowdhuman_yolov5m.pt')
path_2 = leon_path_in(r'yolov5_track/strong_sort/configs/strong_sort.yaml')
path_3 = leon_path_in(r'D:\Leon-Coding\Leon_TestData\mans_0.mp4')
com = r'python {}  --yolo-weights {} --config-strongsort {}  --classes 0 --source {}'.format(path_0,path_1,path_2,path_3)
# print(com)
os.system(com)