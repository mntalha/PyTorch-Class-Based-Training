#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:30:30 2023

@author: talha
"""

#libraries
import torch
import time

class ModelTrain:

    # values 
    loss_values = {
        'train_every_iteration' : [] ,
        'train_every_epoch' : [] ,
        'validation_every_iteration' : [] ,
        'validation_every_epoch' : []
        }
    
    accuracy_values = {
       'train_every_iteration' : [] ,
       'train_every_epoch' : [] ,
       'validation_every_iteration' : [] ,
       'validation_every_epoch' : []
       }
    
    model_prf_rslt = {
        'loss_values' : loss_values,
        'accuracy_values' : accuracy_values 
        }

    device = None
    use_gpu = None
    model = None
    criteria = None
    optimizer = None
    epoch = None
    debug = False
    loss_previous = 999999
    latent_sample = None
    bestmodel = None

    def __init__(self, models_dict, device, debug, islogged=True, logger = None):
        
        # Initial values
        self.model_name = models_dict["name"]
        self.model = models_dict["model"]
        self.criteria = models_dict["criteria"]
        self.optimizer = models_dict["optimizer"]
        self.epoch = models_dict["n_epoch"]
        self.train_loader =  models_dict["train_loader"]
        self.validation_loader =  models_dict["validation_loader"]
        self.classes = models_dict["classes"]
        self.device = device
        
        self.debug = debug
        self.islogsaved = islogged
        self.logger = logger

    def run_(self):
        
        if self.debug:
            print("---------------------TRAIN PROCESS------------------------------")

        if self.islogsaved:
            self.logger.debug("---------------------TRAIN PROCESS------------------------------")
        #Model to cpu or cuda(GPU)
        self.model.to(self.device)
        
        start = time.time()

        for epoch in range(self.epoch):
            
            if self.debug:
                print(f" ********** { {epoch} } EPOCH")
            
            if self.islogsaved:
                self.logger.debug (f" ********** { {epoch} } EPOCH")

            # make it 0 in each epoch
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0
            
             #start train gradient calculation active
            self.model.train()

            for idx, (imgs,clss) in enumerate(self.train_loader):
                
                
                imgs = imgs.to(self.device)
                #imgs = imgs[:, None, :, :]
                imgs =  imgs.to(torch.float32)

 
                #gradient refresh, makes the general faster
                for param in self.model.parameters():
                    param.grad = None
                     
               
                if self.model_name == "nvidia":
                    y_pred = self.model(imgs)[0]
                else:
                    y_pred = self.model(imgs)
                    #torch.tensor([self.classes.index(i) for i in clss]).to(self.device)
                target = clss.to(self.device)     
                loss = self.criteria(y_pred, target)                 
                loss.backward()
                    
                self.optimizer.step()

                #On each batch it sum up.
                train_loss += loss.item()* imgs.size(0)
                
                _, prediction = torch.max(y_pred, dim=1)
                correct_tensor = prediction.eq(target.data.view_as(prediction))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                
                #On each batch it sum up.
                train_acc += accuracy.item() * imgs.size(0)

                self.model_prf_rslt["loss_values"]["train_every_iteration"].append(loss.item()* imgs.size(0))
                self.model_prf_rslt["accuracy_values"]["train_every_iteration"].append(accuracy.item() * imgs.size(0))
            
            #Epoch losses and accuracy
            train_loss = train_loss / len(self.train_loader.sampler)
            self.model_prf_rslt["loss_values"]['train_every_epoch'].append(train_loss)
            
            #Epoch losses and accuracy
            train_acc = train_acc / len(self.train_loader.sampler)
            self.model_prf_rslt["accuracy_values"]['train_every_epoch'].append(train_acc)
            
            if self.debug:
                print(f' {epoch} --> Train Loss:  {train_loss}')
                print(f' {epoch} --> Train Acc:  {train_acc}')

            if self.islogsaved:
                self.logger.debug(f' {epoch} --> Train Loss:  {train_loss}')
                self.logger.debug(f' {epoch} --> Train Acc:  {train_acc}')


            if epoch % 3 == 0:
                
                #start evaluation gradient calculation passive
                self.model.eval()

                # doesn't #turn off Dropout and BatchNorm.                 
                with torch.no_grad():
                    
                    # Measure the performance in validation set.
                    for idx2, (imgs2,clss2) in enumerate(self.validation_loader):
                        
                        imgs2 = imgs2.to(self.device)
                        #imgs2 = imgs2[:, None, :, :]
                        imgs2 = imgs2.to(torch.float32)

                        if self.model_name == "nvidia":
                            y_pred2 = self.model(imgs2)[0]
                            target2 = clss2.to(self.device)
                        else:
                            y_pred2 = self.model(imgs2)
                            target2 = torch.tensor([self.classes.index(i) for i in clss2]).to(self.device)

                        loss2 = self.criteria(y_pred2, target2)
                        
                        #On each batch it sum up.
                        valid_loss += loss2.item()* imgs2.size(0)
                        
                        #Accuracy calculation part
                        _, prediction2 = torch.max(y_pred2, dim=1)
                        correct_tensor2 = prediction2.eq(target2.data.view_as(prediction2))
                        accuracy2 = torch.mean(correct_tensor2.type(torch.FloatTensor))
    
                        #On each batch it sum up.
                        valid_acc+= accuracy2.item() * imgs2.size(0)
                        
                        
                        self.model_prf_rslt["loss_values"]['validation_every_iteration'].append(loss2.item()* imgs2.size(0))
                        self.model_prf_rslt["accuracy_values"]['validation_every_iteration'].append(accuracy2.item() * imgs2.size(0))

                        # if loss2 < self.loss_previous:
                        #     self.bestmodel = self.model
                        #     self.loss_previous = loss2
                        # else:
                        #     pass


                #Epoch losses and accuracy
                valid_loss = valid_loss / (len(self.validation_loader.sampler))
                self.model_prf_rslt["loss_values"]['validation_every_epoch'].append(valid_loss)
                
                valid_acc = valid_acc / (len(self.validation_loader.sampler))
                self.model_prf_rslt["accuracy_values"]['validation_every_epoch'].append(valid_acc)
                
                if self.debug:
                    print(f' {epoch} --> Validation Loss:  {valid_loss}')
                    print(f' {epoch} --> Validation Acc:  {valid_acc}')
                
                if self.islogsaved:
                    self.logger.debug(f' {epoch} --> Validation Loss:  {valid_loss}')
                    self.logger.debug(f' {epoch} --> Validation Acc:  {valid_acc}')

        end = time.time()
        if self.debug:
            print('Total Elapsed time is %f seconds.' % (end - start))
        if self.islogsaved:
            self.logger.debug(f' Total Elapsed time is %f seconds {(end - start)}')
        
        return self.model_prf_rslt,self.model
