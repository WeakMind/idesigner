# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:03:33 2019

@author: Harshit
"""

import shutil,os
from skimage import io
val_dir = './validation'
if not os.path.exists(val_dir):
    os.mkdir(val_dir)
    
train_dir = '/media/idesigner/designer_image_train_v2_cropped/'

train = './train'
if not os.path.exists(train):
    os.mkdir(train)
    
dirs = os.listdir(train_dir)
for d in dirs:
    if not os.path.exists(os.path.join(val_dir,d)):
        os.mkdir(os.path.join(val_dir,d))
    if not os.path.exists(os.path.join(train,d)):
        os.mkdir(os.path.join(train,d))
    imgs = os.listdir(os.path.join(train_dir,d))
    print(len(imgs))
    val_imgs = imgs[:100]
    train_imgs = imgs[100:]
    for img in val_imgs:
        image = io.imread(os.path.join(os.path.join(train_dir,d),img))
        io.imsave(os.path.join(os.path.join(val_dir,d),img),image)
        #shutil.move(img,os.path.join(val_dir,d))
    
    for img in train_imgs:
        image = io.imread(os.path.join(os.path.join(train_dir,d),img))
        io.imsave(os.path.join(os.path.join(train,d),img),image)