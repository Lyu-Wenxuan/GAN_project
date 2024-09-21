import  torch
import torch.nn as nn
#图片的通道数
nc=3
#一张图片的随机噪声
nz=100
#生成器generator的特征大小
ngf=64
#判别器discrimination的特征大小
ndf=64
#定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main=nn.Sequential(
            #input=nc(3)*64*64  （64，3,64,64）
            #1 nc=3，ndf=64
            nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1,bias=False),#64*32*32
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            #2
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),#128*16*16
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #3
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),#256*8*8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),#512*4*4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 添加
            # 4
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),  # 512*4*4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),


            #5
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),#1*1*1
            nn.Sigmoid()
        )
    def forward(self,input):
        return self.main(input)
