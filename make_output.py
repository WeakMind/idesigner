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
        return self.model(X)

def train(data_dir,test_dir):
    model = Model().cuda()
    
    
    indexes = range(0,50)
    classes = np.sort(os.listdir(data_dir))
    d = dict(zip(indexes,classes))
    
    model.load_state_dict(torch.load('./vgg19_bn.pth'))
    test(test_dir,model,d,36)
    

def test(data_dir,model,d,num):
    '''t = [transforms.Resize((256,256)),transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]'''
    t = [transforms.Resize((299,299)),transforms.TenCrop(224),transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.ToTensor()(crop)) for crop in crops]))]

    transform = transforms.Compose(t)
    data = Data_test(data_dir,transform)   
    dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size=1, num_workers=4)
    names=[]
    results = []
    for i,(img,path) in enumerate(dataloader):
        img = img.cuda()
        bs, ncrops, c, h, w = img.size()
        img = img.view(-1, c, h, w)
        fwd = model(img)
        
        fwd = fwd.view(bs,ncrops,-1)
        
        fwd = fwd.mean(1)
        
        fwd = torch.softmax(fwd.detach().cpu(),dim=1)
        
        max_index = (torch.argmax(fwd,dim=1))
        
        result = d[max_index]
        
        path = path[0]
        path = path.split('/')[-1]
        names.append(path)
        results.append(result)
        
    
    import pandas as pd
    result = pd.DataFrame(OrderedDict({'Category': results,'Id': names}))
    result.to_csv('./output_{}.csv'.format(num),index=False)
    
    
if __name__ == '__main__':
    data_dir_train = '/media/idesigner/train'
    data_dir_test = '/media/idesigner/test'
    
    
    train(data_dir_train,data_dir_test)
    
    
        