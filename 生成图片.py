'''
Description描述: 
Autor作者: lhy
Date日期: 2023-11-12 22:06:12
LastEditTime: 2023-11-13 18:24:05
'''
import torch.nn as nn
import torch
import os
import time


#2导入所需的包
import random
import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

netG=torch.load('./netg79.pt')
print("----开始生成图片-----")
nz=100
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建一批噪声数据用来生成
img_list=[]
for a in range(100):#修改这里的100来选择生成多少图片
    fixed_noise = torch.randn(size=(1, nz, 1, 1), device=device)  # (1,100,1,1) 用于每次图像生成的
    fake=netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake,padding=2,normalize=True))
    i=vutils.make_grid(fake,padding=2,normalize=True)
    fig=plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(i,(1,2,0)))
    plt.axis("off")
    root="D:\\third_year\pose\project1\imgout\\"
    plt.savefig(root+str(a)+"_"+".png")
    plt.close(fig)