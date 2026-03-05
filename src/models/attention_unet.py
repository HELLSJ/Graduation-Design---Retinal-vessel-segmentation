import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.attention4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        
        self.attention3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        
        self.attention2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        
        self.attention1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        
        bottleneck = self.bottleneck(p4)
        
        d4 = self.upconv4(bottleneck)
        e4_att = self.attention4(d4, e4)
        d4 = torch.cat([e4_att, d4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        e3_att = self.attention3(d3, e3)
        d3 = torch.cat([e3_att, d3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        e2_att = self.attention2(d2, e2)
        d2 = torch.cat([e2_att, d2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        e1_att = self.attention1(d1, e1)
        d1 = torch.cat([e1_att, d1], dim=1)
        d1 = self.decoder1(d1)
        
        output = self.final_conv(d1)
        return output


class ImprovedAttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.attention4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        
        self.attention3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        
        self.attention2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        
        self.attention1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        
        bottleneck = self.bottleneck(p4)
        
        d4 = self.upconv4(bottleneck)
        e4_att = self.attention4(d4, e4)
        d4 = torch.cat([e4_att, d4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        e3_att = self.attention3(d3, e3)
        d3 = torch.cat([e3_att, d3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        e2_att = self.attention2(d2, e2)
        d2 = torch.cat([e2_att, d2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        e1_att = self.attention1(d1, e1)
        d1 = torch.cat([e1_att, d1], dim=1)
        d1 = self.decoder1(d1)
        
        output = self.final_conv(d1)
        return output


def get_model(model_type='attention_unet', in_channels=3, out_channels=1):
    if model_type == 'attention_unet':
        return AttentionUNet(in_channels, out_channels)
    elif model_type == 'improved_attention_unet':
        return ImprovedAttentionUNet(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")