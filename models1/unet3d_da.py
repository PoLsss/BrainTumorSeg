import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn import init

class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True, res=True):
        super().__init__()
        self.res = res
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv3d(in_channels,c_m,kernel_size = 1)
        self.convB=nn.Conv3d(in_channels,c_n,kernel_size = 1)
        self.convV=nn.Conv3d(in_channels,c_n,kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv3d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w, d=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w,d
        B=self.convB(x) #b,c_n,h,w,d
        V=self.convV(x) #b,c_n,h,w,d
        tmpA=A.view(b,self.c_m,-1)

        attention_maps=F.softmax(B.view(b,self.c_n,-1),dim=-1)
        attention_vectors=F.softmax(V.view(b,self.c_n,-1),dim=-1)
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w,d) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)
        if self.res:
            tmpZ = tmpZ+x
        return tmpZ


class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UNet3d_da(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.doubleAttention_skip1 = DoubleAttention(n_channels, n_channels , n_channels, True)
        self.doubleAttention_skip2 = DoubleAttention(n_channels*2, n_channels*2, n_channels*2, True)
        self.doubleAttention_skip3 = DoubleAttention(n_channels*4, n_channels*4, n_channels*4, True)
        self.doubleAttention_skip4 = DoubleAttention(n_channels*8, n_channels*8, n_channels*8, True)

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        skip_conv1 = self.doubleAttention_skip1(x1)
        x2 = self.enc1(x1)
        skip_conv2 = self.doubleAttention_skip2(x2)
        x3 = self.enc2(x2)
        skip_conv3 = self.doubleAttention_skip3(x3)
        x4 = self.enc3(x3)
        skip_conv4 = self.doubleAttention_skip4(x4)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, skip_conv4)
        mask = self.dec2(mask, skip_conv3)
        mask = self.dec3(mask, skip_conv2)
        mask = self.dec4(mask, skip_conv1)
        mask = self.out(mask)
        return mask
