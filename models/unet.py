import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=20):
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)

        # Decoder (Expansive Path)
        self.upconv3 = self.conv_block(256, 128)
        self.upconv2 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(F.max_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.max_pool2d(conv2, kernel_size=2, stride=2))

        # Decoder
        upconv3 = F.interpolate(conv3, scale_factor=2, mode='bilinear', align_corners=True)
        upconv3 = self.upconv3(upconv3)
        upconv2 = F.interpolate(upconv3 + conv2, scale_factor=2, mode='bilinear', align_corners=True)
        upconv2 = self.upconv2(upconv2)
        upconv1 = F.interpolate(upconv2 + conv1, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final convolution for segmentation map
        segmentation_map = self.final_conv(upconv1)
        return segmentation_map