from idlelib import history
from os import listdir
from os.path import join
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Custom_dataset----------------------------------------------------------------------------
import glob
class Facade(Dataset):
  def __init__(self, path, transform = None):
    path = "C:/pix2pix_d/datasets/train/"
    self.filenames = glob.glob(path + '/*.jpg')  #데이터의 경로를 불러옴
    self.transform = transform

  def __getitem__(self, idx):
    photoname = self.filenames[idx]  #하나씩 읽어 들려옴
    sketchname = self.filenames[idx][:-3] + 'png'
    photo = Image.open(photoname).convert('RGB')

    width, height = photo.size # input과 label을 나누기 위하 width을 반으로 나눔
    # print(width)
    # print(height)

    input_img = photo.crop((width // 2, 0, width, height))

    label_img = photo.crop((0, 0, width // 2, height))
    # plt.imshow(label_img)
    # plt.show()
    sketch = input_img
    photo = label_img
    # plt.imshow(photo)
    # plt.show()



    if self.transform:
      photo = self.transform(photo)
      sketch = self.transform(sketch)

    return photo, sketch, (photoname, sketchname)

  def __len__(self):
    return len(self.filenames)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

data_path = "C:/pix2pix_d/datasets/train"
batch_size = 16
dataset_Facade = Facade(path = data_path,
                        transform=transform)

class UNetUp(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.0):
        super().__init__()
        temp = []
        temp = [nn.ConvTranspose2d(in_dim,out_dim,kernel_size=4,stride=2,padding=1,bias=False)
            ,nn.BatchNorm2d(out_dim),nn.ReLU()]

        if dropout:
            temp.append(nn.Dropout2d(dropout))

        self.u_up = nn.Sequential(*temp)


    def forward(self, x, skip):
        x = self.u_up(x)
        x = torch.cat((x,skip),1)
        return x

class UNetDown(nn.Module):
    def __init__(self,in_dim,out_dim,normalize=True,dropout=0.0):
        super().__init__()

        temp = [nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False)] #모든 conv 4x4 filter, stride 2

        if normalize:
            temp.append(nn.BatchNorm2d(out_dim)),
        temp.append(nn.LeakyReLU(0.2))

        self.u_down = nn.Sequential(*temp)


    def forward(self, x):
        return self.u_down(x)


# generator: 가짜 이미지를 생성합니다.
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,128)
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

#check encoder
# x = torch.randn(16,3,256,256,device=device)
# model = UNetDown(3,64).to(device)
# out = model(x)
# print(out.shape)

#check decoder
# x = torch.randn(16, 128, 64, 64, device=device)
# model = UNetUp(128,64).to(device)
# out = model(x,out)
#print(out.shape)

#check Generator
# x = torch.randn(16,3,256,256,device=device)
# model = GeneratorUNet().to(device)
# out = model(x)
#print(out.shape)

class Patch(nn.Module):
    def __init__(self,in_dim,out_dim,normalize=True):
        super().__init__()
        temp = []
        temp = [nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False)] #모든 conv 4x4 filter, stride 2

        if normalize:
            temp.append(nn.BatchNorm2d(out_dim)),
        temp.append(nn.LeakyReLU(0.2))

        self.patch = nn.Sequential(*temp)


    def forward(self, x):
        return self.patch(x)

class PatchDiscriminator(nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super().__init__()

            self.dis1 = Patch(in_channels*2,64,normalize=False)
            self.dis2 = Patch(64,128)
            self.dis3 = Patch(128,256)
            self.dis4 = Patch(256,512)
            self.patch = nn.Conv2d(512, 1, 3, padding=1)

        def forward(self, x, photo):
            x = torch.cat((x,photo), 1)
            dis1 = self.dis1(x)
            dis2 = self.dis2(dis1)
            dis3 = self.dis3(dis2)
            dis4 = self.dis4(dis3)
            patch = self.patch(dis4)
            x = torch.sigmoid(patch)
            return x

dataloader = DataLoader(dataset=dataset_Facade,batch_size=batch_size,shuffle=True)
#Model----------------------------------------------------------------------------------------



# x = torch.randn(16,64,128,128,device=device)
# model = Patch(64,128).to(device)
# out = model(x)
#print(out.shape)

# x = torch.randn(16,3,256,256,device=device)
# model = PatchDiscriminator().to(device)
# out = model(x,x)
# #print(out.shape)


Generator = GeneratorUNet().to(device)
Discriminator = PatchDiscriminator().to(device)

##가중치 초기화
def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

Generator.apply(initialize_weights)
Discriminator.apply(initialize_weights)

## Optimizer 설정하기
lr = 0.0002
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D= torch.optim.Adam(Discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

## 손실함수 정의하기

loss_gan = nn.BCELoss().to(device)
loss_l1 = nn.L1Loss().to(device)


Generator.train()
Discriminator.train()

batch_count = 0
num_epochs = 500
start_time = time.time()

loss_hist = {'gen': [],
             'dis': []}

for epoch in range(num_epochs):
    for photos, sketches, _ in dataloader:
        ba_si = photos.size(0)

        # real image
        real_a = photos.to(device)#input _image sketch

        real_b = sketches.to(device)#조건 image photos

        # patch label
        # real_label = torch.ones(ba_si,16, requires_grad=False).to(device)
        # fake_label = torch.zeros(ba_si,16, requires_grad=False).to(device)

        real_label = torch.ones(batch_size, 1, 16, 16,requires_grad = False).cuda()
        fake_label = torch.zeros(batch_size, 1, 16, 16,requires_grad = False).cuda()

        # generator
        Generator.zero_grad() #Gen optimizer 초기화
        fake_photo = Generator(real_a)
        out_dis = Discriminator(fake_photo,real_b)
        gen_loss = loss_gan(out_dis, real_label)
        pixel_loss = loss_l1(fake_photo, real_b)

        lambda_pixel = 100
        g_loss = gen_loss + lambda_pixel * pixel_loss

        g_loss.backward()#각 wight에 대한 모든 gradient 값 계산 저장

        optimizer_G.step()#weight들 update theta = theta - learning_rate * theta

        # discriminator
        Discriminator.zero_grad() #Dis optimizer 초기화

        out_dis = Discriminator(real_b, real_a)  # 진짜 이미지 식별
        real_loss = loss_gan(out_dis, real_label)

        out_dis = Discriminator(fake_photo.detach(), real_a)  # 가짜 이미지 식별
        fake_loss = loss_gan(out_dis, fake_label)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        loss_hist['gen'].append(g_loss.item())
        loss_hist['dis'].append(d_loss.item())

        batch_count += 1
        if batch_count % 100 == 0:
            print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' % (
            epoch, g_loss.item(), d_loss.item(), (time.time() - start_time) / 60))

    #Inference every step
    Generator.eval()

    # 가짜 이미지 생성
    with torch.no_grad():
        for a in dataloader:
            fake_imgs = Generator(a[1].to(device)).detach().cpu()
            real_imgs = a[0]

            break

    plt.figure(figsize=(10, 10))

    for ii in range(0, 16, 2):
        print(ii)
        plt.subplot(4, 4, ii + 1)
        plt.imshow(to_pil_image(0.5 * fake_imgs[ii] + 0.5), cmap='gray')
        plt.axis('off')
        plt.subplot(4, 4, ii + 2)
        plt.imshow(to_pil_image(0.5 * real_imgs[ii] + 0.5), cmap='gray')
        plt.axis('off')
    plt.savefig(f'./image/epoch_{epoch}.png')

# loss history
plt.figure(figsize=(10,5))
plt.title('Loss Progress')
plt.plot(loss_hist['gen'], label='Gen. Loss')
plt.plot(loss_hist['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()

path2models = './models/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

torch.save(Generator.state_dict(), path2weights_gen)
torch.save(Discriminator.state_dict(), path2weights_dis)

weights = torch.load(path2weights_gen)
Generator.load_state_dict(weights)

# evaluation model
Generator.eval()

# 가짜 이미지 생성
with torch.no_grad():
    for a,b in dataloader:
        fake_imgs = Generator(a.to(device)).detach().cpu()
        real_imgs = b
        break

plt.figure(figsize=(10,10))

for ii in range(0,16,2):
    plt.subplot(4,4,ii+1)
    plt.imshow(to_pil_image(0.5*real_imgs[ii]+0.5))
    plt.axis('off')
    plt.subplot(4,4,ii+2)
    plt.imshow(to_pil_image(0.5*fake_imgs[ii]+0.5))
    plt.axis('off')
