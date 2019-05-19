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
        self.model = models.vgg16_bn(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096,50)
        
    def forward(self,X):
        return self.model(X)

def train(data_dir,model_dir,test_dir):
    d = 0
    t = [transforms.Resize((224,224)),transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    transform = transforms.Compose(t)
    data = Data(data_dir,transform)
    dataloader = torch.utils.data.DataLoader(data, shuffle = True, batch_size=32, num_workers=4)
    
    epochs = 80
    model = Model().cuda()
    model.load_state_dict(torch.load('./saved_models/epoch_28.pth'))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001,weight_decay = 0.00005)
    
    indexes = range(0,50)
    classes = np.sort(os.listdir(data_dir))
    d = dict(zip(indexes,classes))
    
    for epoch in range(epochs):
        total_error = 0.
        total_samples = 0
        for i,(imgs,target) in enumerate(dataloader):
            
            imgs = imgs.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            forward = model(imgs)
            
            loss = criterion(forward,target)
            
            total_error+=loss.item()
            
            total_samples+=1
            
            loss.backward()
            optimizer.step()
            
        print('Epoch : %s loss : %s' % (epoch+19,total_error/total_samples))
        if epoch % 3 == 0:
            torch.save(model.state_dict(),'./saved_models/epoch_{}.pth'.format(epoch+28))
            test(test_dir,model,d,epoch+28)
        import time
        time.sleep(120)
    return model,d

def test(data_dir,model,d,num):
    t = [transforms.Resize((224,224)),transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    transform = transforms.Compose(t)
    data = Data_test(data_dir,transform)   
    dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size=1, num_workers=4)
    names=[]
    results = []
    for i,(img,path) in enumerate(dataloader):
        img = img.cuda()
        
        fwd = model(img)
        fwd = torch.softmax(fwd.detach().cpu(),dim=1)
        max_index = int(torch.argmax(fwd))
        result = d[max_index]
        
        path = path[0]
        path = path.split('/')[-1]
        names.append(path)
        results.append(result)
        
    
    import pandas as pd
    result = pd.DataFrame(OrderedDict({'Category': results,'Id': names}))
    result.to_csv('./output_{}.csv'.format(num),index=False)
    
    
if __name__ == '__main__':
    data_dir_train = '/media/designer_image_train_v2_cropped'
    data_dir_test = '/media/test'
    
    model_dir = './saved_models'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model,d = train(data_dir_train,model_dir,data_dir_test)
    test(data_dir_test,model,d)
    
        