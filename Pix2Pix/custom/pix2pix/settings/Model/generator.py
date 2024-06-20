import torch.nn as nn
import torch.nn.functional as F
import torch

#block을 먼저 쌓음
class UNetDown(nn.Module):
    def __init__(self,in_dim,out_dim,normalize=True,dropout=0.0):#모델을 만들고
        super(UNetDown,self).__init__()
        temp = []
        temp = [nn.Conv2d(in_dim,out_dim,kernel_size=4,stride=2, padding=1)]

        if normalize:
            temp.append(nn.BatchNorm2d(out_dim)),
        temp.append(nn.LeakyReLU(0.2))

        self.u_down = nn.Sequential(*temp)

    def forward(self,x):#위에서 만든 모델의 동작 정의
        return self.u_down(x)


class UNetUp(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.0):
        super(UNetUp,self).__init__()
        temp = []
        temp = [nn.ConvTranspose2d(in_dim,out_dim,kernel_size=4,stride=2, padding=1),nn.BatchNorm2d(out_dim),nn.ReLU()]

        if dropout:
            temp.append(nn.Dropout2d(dropout))
    

        self.u_up = nn.Sequential(*temp)
    
    def forward(self,x,skip):
        x = self.u_up(x)
        x = torch.cat((x,skip),1)
        return x
    
    
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet,self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128,256)
        self.down4 = UNetDown(256,512)
        self.down5 = UNetDown(512,512)
        self.down6 = UNetDown(512,512)
        self.down7 = UNetDown(512,512)
        self.down8 = UNetDown(512,512)

        self.up1 = UNetUp(512,512,dropout=0.5)
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,stride=2,padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,skip = d7)#x,skip
        u2 = self.up2(u1,skip = d6)
        u3 = self.up3(u2,skip = d5)#[16,1024,8,8]
        u4 = self.up4(u3, skip = d4)
        u5 = self.up5(u4, skip = d3)
        u6 = self.up6(u5, skip = d2)
        u7 = self.up7(u6, skip = d1)
        u8 = self.up8(u7)

        return u8

# check encoder
x = torch.randn(16,3,256,256)
model = UNetDown(3,64)
out = model(x)
print(out.shape)

# check decoder
x = torch.randn(16, 128, 64, 64)
model = UNetUp(128,64)
out = model(x,out)
print(out.shape)

# check Generator
x = torch.randn(16,3,256,256)
model = GeneratorUNet()
out = model(x)
print(out.shape)
