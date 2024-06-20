import torch
import torch.nn as nn

class Patch(nn.Module):
    def __init__(self,in_dim,out_dim,normalize=True):
        super(Patch,self).__init__()
        temp = []
        temp = [nn.Conv2d(in_dim,out_dim,kernel_size=4,stride=2, padding=1)]

        if normalize:
             temp.append(nn.BatchNorm2d(out_dim))
        temp.append(nn.LeakyReLU(0.2))

        self.patch = nn.Sequential(*temp)
   
    def forward(self, x):
         return self.patch(x)


class PatchDiscriminator(nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(PatchDiscriminator,self).__init__()
            self.dis1 = Patch(in_channels*2, 64, normalize=False)
            self.dis2 = Patch(64, 128)
            self.dis3 = Patch(128,256)
            self.dis4 = Patch(256,512)
            self.patch = nn.Conv2d(512, 1, 3, padding=1)

        def forward(self, x, photo):
             x = torch.cat((x,photo),1)
             dis1 = self.dis1(x)
             dis2 = self.dis2(dis1)
             dis3 = self.dis3(dis2)
             dis4 = self.dis4(dis3)
             dis5 = self.patch(dis4)
             x = torch.sigmoid(dis5)
             return x

#check Generator
x = torch.randn(16,3,256,256)
photo = torch.randn(16,3,256,256)
model = PatchDiscriminator()
out = model(x,photo)
print(out.shape)            
        