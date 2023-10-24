#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:37:13 2023

@author: talha
"""

#libraries
import torch

class ModelTest:

    # values 
    device = None
    use_gpu = None
    model = None
    criteria = None
    optimizer = None
    epoch = None
    debug = False
    latent_sample = None
 
    def __init__(self, models_dict, device, debug, islogged= True, logger = None):
     
        # Initial values
        self.model_name = models_dict["name"]
        self.model = models_dict["model"]
        self.criteria = models_dict["criteria"]
        self.optimizer = models_dict["optimizer"]
        self.test_loader =  models_dict["test_loader"]
        self.classes = models_dict["classes"]
        self.debug = debug
        self.device = device

        self.islogsaved = islogged
        self.logger = logger

    def run_(self):
        
        #TEST
        if self.debug:
            print("---------------------TEST PROCESS ------------------------------")
        if self.islogsaved:
            self.logger.debug("---------------------TEST PROCESS------------------------------")
        #Test Values
        test_loss = 0.0
        test_acc = 0.0

        #Model in Evaluatıon mode, no changes in models parameters        
        self.model.eval()
        
        with torch.no_grad():
            for idx2, (imgs2,clss2) in enumerate(self.test_loader):
                
                imgs2 = imgs2.to(self.device)
                #imgs2 = imgs2[:, None, :, :]
                imgs2 = imgs2.to(torch.float32)

                if self.model_name == "nvidia":
                    y_pred2 = self.model(imgs2)[0]
                else:
                    y_pred2 = self.model(imgs2)
                    
                target2 = clss2.to(self.device)     
                loss2 = self.criteria(y_pred2, target2)

                #On each batch it sum up.
                test_loss += loss2.item()* imgs2.size(0)
                
                _, prediction2 = torch.max(y_pred2, dim=1)
                correct_tensor2 = prediction2.eq(target2.data.view_as(prediction2))
                accuracy2 = torch.mean(correct_tensor2.type(torch.FloatTensor)) 
                
                test_acc+= accuracy2.item() * imgs2.size(0)

                
        #Epoch losses and accuracy
        test_loss = test_loss / (len(self.test_loader.sampler))
        test_acc = test_acc / (len(self.test_loader.sampler))

        
        if self.debug:
            print('Test Loss: %f ' % (test_loss))
            print('Test Acc: %f ' % (test_acc))
        if self.islogsaved:
            self.logger.debug(f"Test Loss: {test_loss}")
            self.logger.debug(f"Test Acc: {test_acc}")
        
        

def load_pytorch_model(path, raw_model):   
    """
    take the transform .cuda() and .cpu() into consideration.
    """
    import os 
    # Check the file whether it is exist
    isExist = os.path.isfile(path)
    if not isExist:
        print("model couldnt found...")
        return False
    
    # create raw model
    # raw_model = Model_Class(input_channel,
    #                   num_hiddens,
    #                   num_residual_layers, 
    #                   num_residual_hiddens,
    #                   num_embeddings, 
    #                   embedding_dim, 
    #                   commitment_cost,
    #                   decay)
    
    # load it
    if torch.cuda.is_available():
        raw_model.load_state_dict(torch.load(path))
    else:
        raw_model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

    return raw_model


if __name__ == "__main__":
    batch_size = 32
    epoch_number = 50
    learning_rate = 3e-4
    weight_decay = 3e-6
    model_name = None
    device_ = "cuda:0" # "cuda:0" "cuda:1" "cpu"

    #paths on server
    path = "/data/mnk7465/ebsd/"
    path_dataset = path + "dataset/original_data_0/" #concat
    path_outputs = path + "outputs/"
    path_graphics = path_outputs + "plots/"
    path_training_results = path_outputs + "trainining_results"
    path_trained_models =  path_outputs + "models"

    from EBSDDataset import call_dataloader, call_whole_dataloader
    train_loader, test_loader,  validation_loader, classes = call_dataloader(path = path_dataset, batch_size = batch_size)
    
    from models import get_trained_model, count_parameters
    model = load_pytorch_model(path_trained_models+"/nvidia",get_trained_model("nvidia"))
    print(f" Parameters : {count_parameters(model)}")
    print(f" Test Dataset: {len(test_loader)* batch_size}")

    #
    device_ = "cpu" # "cuda:0" "cuda:1" "cpu"
    if device_ != None:
        device = torch.device(device_)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    import torch.nn as nn
    criteria = nn.CrossEntropyLoss()

    models_dict =  {
        'name' : 'nvidia',
        'model' : model ,
        'criteria' :  criteria,
        'optimizer' : None ,
        'n_epoch' : None,
        'train_loader' : None,
        'validation_loader': None,
        'test_loader' : test_loader,
        'classes' : classes
        }
    test_obj = ModelTest(models_dict, device, debug=True, islogged = False)
    test_obj.run_()


                        

                    
                    
                    



                        
