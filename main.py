#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:41:57 2023

@author: talha
"""

batch_size = 32
epoch_number = 50
learning_rate = 3e-6
weight_decay = 3e-6
model_name = None
path_results = "./outputs"
device_ = "cpu" # "cuda:0" "cuda:1" "cpu"


#Libraries 
import torch.optim as optim
import torch
import random
import numpy as np
import os
import torch.nn as nn
import matplotlib.pyplot as plt


# Dataloader
from Dataset import call_dataloader


# Train , test and validation
from model_train import ModelTrain
from model_test import ModelTest

def set_seed(seed = 42):
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def visualize(train_loss, validation_loss, title, img_name, epoch_number, status= "Loss"):

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)

    plt.title(title)

    color = 'tab:purple'
    plt.plot(train_loss, color=color)

    color = 'tab:blue'
    x_axis = list(range(0, epoch_number, 3))
    plt.plot(x_axis, validation_loss, color=color)

    class_names = ["Train", "Validation"]

    plt.xlabel("Epoch")
    plt.ylabel(status + " Value")

    plt.legend(class_names, loc=1)
    plt.show()

    fig.savefig(img_name, dpi=500)
    

from models import get_trained_model, count_parameters


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Pytorch Pretrained Model Comparison')
    parser.add_argument('--model', type=str, default='alexnet',
                        help='resnet50, resnet34, vgg19, vgg16, googleNet, mobilnet, squeezenet, inception (default: alexnet)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='32, 64, 128 (default: 64)')   
    parser.add_argument('--save_models', type=str, default = path_results, 
                        help='directory to save ai models')
    parser.add_argument('--dataset', type=str, default = "mnist", 
                        help='mnist, cifar10, fashionmnist, country211')
    args = parser.parse_args()
    #
    set_seed()
    
    #
    if device_ != None:
        device = torch.device(device_)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #Logging 
    import logging
    log_file_name = args.model + str(args.batch_size) + "_training_result"
    log_path = path_results
    log_file_name = os.path.join(log_path, log_file_name)
    logging.basicConfig(
                filename= log_file_name,
                level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
        )
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.debug(f"Device: {device}")
    logging.debug(f"Batch Size: {args.batch_size}")
    
    #Loader Operation
    train_loader, test_loader = call_dataloader(path = "./data", batch_size = args.batch_size, name = args.dataset)
    
    
    #Model
    model = get_trained_model(args.model)
    logging.debug(f" Model : {args.model} , Parameters : {count_parameters(model)}")
    logging.debug(f" Training Dataset: {len(train_loader)* args.batch_size}")
    logging.debug(f" Test Dataset: {len(test_loader)* args.batch_size}")
    logging.debug("###################################################################################")
    
    #Optimizer
    optimizer = optim.SGD(
                       params = model.parameters(),
                       lr = learning_rate,
                       weight_decay = weight_decay
                       )
    
    # Loss fucntion
    #criteria = nn.MSELoss()
    criteria = nn.CrossEntropyLoss()
    
    models_dict =  {
        'name' : args.model,
        'model' : model ,
        'criteria' :  criteria,
        'optimizer' : optimizer ,
        'n_epoch' : epoch_number,
        'train_loader' : train_loader,
        'validation_loader': None,
        'test_loader' : test_loader,
        'classes' : None
        }
    ############################ TRAIN #######################################
    # Train object for Model Train
    train_obj = ModelTrain(models_dict, device, debug=True, islogged = True,logger = logging)
    # if exist, used GPU_ object
    train_model_prf_rslt , trained_model = train_obj.run_()
    
    ############################ TEST ##################################
    # test object for Model Train
    models_dict["model"] = trained_model
    test_obj = ModelTest(models_dict, device, debug=True, islogged = True,logger = logging)
    # if exist, used GPU_ object
    test_obj.run_()
    
    #Loss graph
    # title = args.model + " Loss_Function"
    # img_name = path_results + title + ".png"
    # visualize(train_model_prf_rslt["loss_values"]["train_every_epoch"], train_model_prf_rslt["loss_values"]["validation_every_epoch"],title, img_name, epoch_number, status = "Loss")
    # title = args.model + " Accuracy Function"
    # img_name = path_results + title + ".png"
    # visualize(train_model_prf_rslt["accuracy_values"]["train_every_epoch"], train_model_prf_rslt["accuracy_values"]["validation_every_epoch"], title, img_name, epoch_number, status = "Accuracy")
