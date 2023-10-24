#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:38:56 2023

@author: talha
"""

from torchvision import models
import torch.nn as nn
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import SegformerFeatureExtractor, SegformerForImageClassification
from transformers import SegformerForImageClassification, SegformerConfig

num_clss = 10 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_trained_model(model_name):
    
    model = None
    
    if model_name == "alexnet":
        model = models.alexnet(pretrained=False)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        model.classifier[6] = nn.Linear(in_features=4096, out_features= num_clss, bias=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc =nn.Linear(in_features=2048, out_features=num_clss, bias=True)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False)
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc =nn.Linear(in_features=512, out_features=num_clss, bias=True)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_clss, bias=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_clss, bias=True)
    elif model_name == "googleNet":
        model = models.googlenet(pretrained=False,transform_input=False)
        model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=1024, out_features=num_clss, bias=True)
    elif model_name == "mobilnet":
        model = models.mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_clss, bias=True)
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=False)
        model.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
        model.classifier[1] = nn.Conv2d(512, num_clss, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == "inception":
        model = models.inception_v3(pretrained=False)
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=num_clss, bias=True)   
    elif model_name == "nvidia":
        model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        model.classifier = nn.Linear(in_features=256, out_features=num_clss, bias=True)
    return model