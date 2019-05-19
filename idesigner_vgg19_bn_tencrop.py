# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:24:31 2019

@author: Harshit
"""

import os
import numpy as np
import torch.nn as nn
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import OrderedDict


class Data(ImageFolder):
    def __getitem__(self,index):
        path,target = self.samples[index]
        img = self.loader(path)
        img = self.transform(img)
        return (img,target)
    
class Data_test(ImageFolder):
    def __getitem__(self,index):
        path,target = self.samples[index]
        img = self.loader(path)
        img = self.transform(img)
        return img,path



class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = models.vgg19_bn(pretrained=True)
        #set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        self.model.classifier[6] = nn.Linear(4096,50)
        
    def forward(self,X):
        #X = X.view(-1,224*224*3)
        return self.model(X)


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def train(data_dir,val_dir):
    d = 0
    
    t = [transforms.Resize(299),transforms.TenCrop(224),transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.ToTensor()(crop)) for crop in crops]))]
    transform = transforms.Compose(t)
    data = Data(data_dir,transform)
    dataloader_train = torch.utils.data.DataLoader(data, shuffle = True, batch_size=2, num_workers=4)
    
    
    
    epochs = 80
    model = Model().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001,weight_decay = 0.0001)
    
    indexes = range(0,50)
    classes = np.sort(os.listdir(data_dir))
    d = dict(zip(indexes,classes))
    
    max_acc = 0.
    for epoch in range(epochs):
        total_error = 0.
        total_samples = 1
        for i,(imgs,target) in enumerate(dataloader_train):
            bs, ncrops, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w)
            imgs = imgs.cuda()
            
            target = tile(target,0,10)
            #import pdb;pdb.set_trace()
            target = target.cuda()
            optimizer.zero_grad()
            forward = model(imgs)
            
            loss = criterion(forward,target)
            
            total_error+=loss.item()
            
            total_samples+=1
            
            loss.backward()
            optimizer.step()
            
        print('Epoch : %s loss : %s' % (epoch,total_error/total_samples))
        
        
        accuracy = val(val_dir,model)
        
        print('Validation accuracy : %f\n' % accuracy)
        
        if accuracy > max_acc:
            max_acc = accuracy
            torch.save(model.state_dict(),'./vgg19_bn_ten_crop.pth')
            
        import time
        time.sleep(120)
    return model,d

def val(data_dir,model):
    t = [transforms.Resize(299),transforms.TenCrop(224),transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.ToTensor()(crop)) for crop in crops]))]

    transform = transforms.Compose(t)
    data = Data(data_dir,transform)   
    dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size=2, num_workers=4)
    
    correct = 0
    total = 0
    for i,(img,target) in enumerate(dataloader):
        img = img.cuda()
        bs, ncrops, c, h, w = img.size()
        img = img.view(-1, c, h, w)
        fwd = model(img)
        
        fwd = fwd.view(bs,ncrops,-1)
        
        fwd = fwd.mean(1)
        
        fwd = torch.softmax(fwd.detach().cpu(),dim=1)
        
        max_index = (torch.argmax(fwd,dim=1))
        correct+= (max_index==target).sum()
        total+=img.shape[0]
        
        
    return (int(correct)/total)*100
    
    
if __name__ == '__main__':
    data_dir_train = '/media/idesigner/train'
    data_dir_val = '/media/idesigner/validation'
    data_dir_test = '/media/idesigner/test'
    
    model,d = train(data_dir_train,data_dir_val)
    
        