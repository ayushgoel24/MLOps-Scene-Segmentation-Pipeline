import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50

class FCN_resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # feature extractor
        self.backbone = resnet50(pretrained=True)

        # layer4 of resnet50 has 2048 layers output
        backbone_output_channels = 2048
        # classifier will have 512 layers
        classifier_channels = 512

        # Classifier head
        self.classifier = nn.Sequential(
                nn.Conv2d(backbone_output_channels, classifier_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(classifier_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(classifier_channels, num_classes, 1)
                )
    
    def forward(self, x):
        # get input shape for upsampling
        input_shape = x.shape[-2:]

        # pass input through the backbone layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # classifier output shape = (batch_size, num_classes, feat_ext_height, feat_ext_width)
        x = self.classifier(x)

        # upsample the classfier output to input shape
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
    