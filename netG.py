#图片的通道数
nc=3
#一张图片的随机噪声
nz=100
#生成器generator的特征大小
ngf=64
#判别器discrimination的特征大小
ndf=64
#定义生成器
import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main=nn.Sequential(
            #1 (64,100,1,1) nz=100,ngf=64
            nn.ConvTranspose2d(nz,ngf*8,kernel_size=4,stride=1,padding=0,bias=False),
            #反卷积要反着推导  比如你想从100*1*1--512*4*4，那么你要从结果也就是想4*4，怎么变成1*1，
            # 卷积核是不变的kernel4*4，那么从4*4-1*1我们只需要padding=0，stride=1即可
            nn.BatchNorm2d(ngf*8),#批量标准化
            nn.ReLU(True),#inplace会用执行随机失活后的结果覆盖原来的输入，
            # 改变了存储值，但随机失活并不会影响梯度计算和反向传播。
            # 即对原值进行操作，然后将得到的值又直接复制到该值中。类似于x = x +1
            #2
            nn.ConvTranspose2d(ngf*8,ngf*4,kernel_size=4,stride=2,padding=1,bias=False),
            #这里我们想要的结果是从4*4--8*8，那么从结果反推  8*8-4*4，卷积核是kernel=4*4不变，
            #那么我们思考padding=1，strade=2即可
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            #3
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #4
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 添加
            # 4
            nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),


            #5
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()#为什么使用tanh这是因为在生成图像时，它们通常被标准化为 [0,1] 或 [-1,1] 范围内。
            # 因此，如果你希望输出图像在 [0,1] 中，
            # 你可以使用 sigmoid，如果你希望它们在 [-1,1] 中，你可以使用 tanh。你总是可以使用 ReLU，
            # 但你只能保证它是非负的并且不在给定的范围内。
        )
    def forward(self,input):
        return self.main(input)
