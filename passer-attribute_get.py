# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 21:05:24 2022

@author: leonz
"""

import os
import json
import torch
from PIL import Image
from torchvision import transforms as T


import sys
if sys.platform == 'win32':
    sys.path.append(r'D:\Leon-Coding\Leon_FC')
else:
    sys.path.append(r'/home/leonzion/Leon_Coding/Leon_FC')
from leon_info import *
from leon_os import *
print(leon_path_in('passer-attribute_get'))
from net import get_model


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default=r'D:\Leon-Coding\Leon-PublicPlacesManage-System\passer-attribute_get\test_sample\test_7.jpg',help='Path to test image')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

######################################################################
# Model and Data
# ---------
def load_network(network):
    save_path = os.path.join('models','passer_attribute.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src


def leon_get_model():
######################################################################
    model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
    num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]
    model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
    model = load_network(model)
    model.eval()
    return model


######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open(leon_path_in(os.path.join(r'passer-attribute_get','doc','label.json')), 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open(leon_path_in(os.path.join(r'passer-attribute_get','doc','attribute.json')), 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))


model = leon_get_model()
src = load_image('D:\\Leon-Coding\\Leon-PublicPlacesManage-System\\passer-attribute_get\\test_sample\\test_market.jpg')
out = model.forward(src)
pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
Dec = predict_decoder(args.dataset)
Dec.decode(pred)

