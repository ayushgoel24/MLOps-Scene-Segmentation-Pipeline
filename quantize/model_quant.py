from copy import deepcopy
import torch
import torch.nn as nn

modules_to_fuse = [
        ['backbone.conv1', 'backbone.bn1', 'backbone.relu'],

        ['backbone.layer1.0.conv1', 'backbone.layer1.0.bn1', 'backbone.layer1.0.relu1'],
        ['backbone.layer1.0.conv2', 'backbone.layer1.0.bn2', 'backbone.layer1.0.relu2'],
        ['backbone.layer1.0.conv3', 'backbone.layer1.0.bn3'],
        ['backbone.layer1.0.downsample.0', 'backbone.layer1.0.downsample.1'],

        ['backbone.layer1.1.conv1', 'backbone.layer1.1.bn1', 'backbone.layer1.1.relu1'],
        ['backbone.layer1.1.conv2', 'backbone.layer1.1.bn2', 'backbone.layer1.1.relu2'],
        ['backbone.layer1.1.conv3', 'backbone.layer1.1.bn3'],

        ['backbone.layer1.2.conv1', 'backbone.layer1.2.bn1', 'backbone.layer1.2.relu1'],
        ['backbone.layer1.2.conv2', 'backbone.layer1.2.bn2', 'backbone.layer1.2.relu2'],
        ['backbone.layer1.2.conv3', 'backbone.layer1.2.bn3'],

        ['backbone.layer2.0.conv1', 'backbone.layer2.0.bn1', 'backbone.layer2.0.relu1'],
        ['backbone.layer2.0.conv2', 'backbone.layer2.0.bn2', 'backbone.layer2.0.relu2'],
        ['backbone.layer2.0.conv3', 'backbone.layer2.0.bn3'],
        ['backbone.layer2.0.downsample.0', 'backbone.layer2.0.downsample.1'],

        ['backbone.layer2.1.conv1', 'backbone.layer2.1.bn1', 'backbone.layer2.1.relu1'],
        ['backbone.layer2.1.conv2', 'backbone.layer2.1.bn2', 'backbone.layer2.1.relu2'],
        ['backbone.layer2.1.conv3', 'backbone.layer2.1.bn3'],

        ['backbone.layer2.2.conv1', 'backbone.layer2.2.bn1', 'backbone.layer2.2.relu1'],
        ['backbone.layer2.2.conv2', 'backbone.layer2.2.bn2', 'backbone.layer2.2.relu2'],
        ['backbone.layer2.2.conv3', 'backbone.layer2.2.bn3'],

        ['backbone.layer2.3.conv1', 'backbone.layer2.3.bn1', 'backbone.layer2.3.relu1'],
        ['backbone.layer2.3.conv2', 'backbone.layer2.3.bn2', 'backbone.layer2.3.relu2'],
        ['backbone.layer2.3.conv3', 'backbone.layer2.3.bn3'],

        ['backbone.layer3.0.conv1', 'backbone.layer3.0.bn1', 'backbone.layer3.0.relu1'],
        ['backbone.layer3.0.conv2', 'backbone.layer3.0.bn2', 'backbone.layer3.0.relu2'],
        ['backbone.layer3.0.conv3', 'backbone.layer3.0.bn3'],
        ['backbone.layer3.0.downsample.0', 'backbone.layer3.0.downsample.1'],

        ['backbone.layer3.1.conv1', 'backbone.layer3.1.bn1', 'backbone.layer3.1.relu1'],
        ['backbone.layer3.1.conv2', 'backbone.layer3.1.bn2', 'backbone.layer3.1.relu2'],
        ['backbone.layer3.1.conv3', 'backbone.layer3.1.bn3'],

        ['backbone.layer3.2.conv1', 'backbone.layer3.2.bn1', 'backbone.layer3.2.relu1'],
        ['backbone.layer3.2.conv2', 'backbone.layer3.2.bn2', 'backbone.layer3.2.relu2'],
        ['backbone.layer3.2.conv3', 'backbone.layer3.2.bn3'],

        ['backbone.layer3.3.conv1', 'backbone.layer3.3.bn1', 'backbone.layer3.3.relu1'],
        ['backbone.layer3.3.conv2', 'backbone.layer3.3.bn2', 'backbone.layer3.3.relu2'],
        ['backbone.layer3.3.conv3', 'backbone.layer3.3.bn3'],

        ['backbone.layer3.4.conv1', 'backbone.layer3.4.bn1', 'backbone.layer3.4.relu1'],
        ['backbone.layer3.4.conv2', 'backbone.layer3.4.bn2', 'backbone.layer3.4.relu2'],
        ['backbone.layer3.4.conv3', 'backbone.layer3.4.bn3'],

        ['backbone.layer3.5.conv1', 'backbone.layer3.5.bn1', 'backbone.layer3.5.relu1'],
        ['backbone.layer3.5.conv2', 'backbone.layer3.5.bn2', 'backbone.layer3.5.relu2'],
        ['backbone.layer3.5.conv3', 'backbone.layer3.5.bn3'],

        ['backbone.layer4.0.conv1', 'backbone.layer4.0.bn1', 'backbone.layer4.0.relu1'],
        ['backbone.layer4.0.conv2', 'backbone.layer4.0.bn2', 'backbone.layer4.0.relu2'],
        ['backbone.layer4.0.conv3', 'backbone.layer4.0.bn3'],
        ['backbone.layer4.0.downsample.0', 'backbone.layer4.0.downsample.1'],

        ['backbone.layer4.1.conv1', 'backbone.layer4.1.bn1', 'backbone.layer4.1.relu1'],
        ['backbone.layer4.1.conv2', 'backbone.layer4.1.bn2', 'backbone.layer4.1.relu2'],
        ['backbone.layer4.1.conv3', 'backbone.layer4.1.bn3'],

        ['backbone.layer4.2.conv1', 'backbone.layer4.2.bn1', 'backbone.layer4.2.relu1'],
        ['backbone.layer4.2.conv2', 'backbone.layer4.2.bn2', 'backbone.layer4.2.relu2'],
        ['backbone.layer4.2.conv3', 'backbone.layer4.2.bn3'],

        ['classifier.0.convs.0.0', 'classifier.0.convs.0.1', 'classifier.0.convs.0.2'],
        ['classifier.0.convs.1.0', 'classifier.0.convs.1.1', 'classifier.0.convs.1.2'],
        ['classifier.0.convs.2.0', 'classifier.0.convs.2.1', 'classifier.0.convs.2.2'],
        ['classifier.0.convs.3.0', 'classifier.0.convs.3.1', 'classifier.0.convs.3.2'],
        ['classifier.0.convs.4.1', 'classifier.0.convs.4.2', 'classifier.0.convs.4.3'],

        ['classifier.0.project.0', 'classifier.0.project.1', 'classifier.0.project.2'],
        ['classifier.1', 'classifier.2', 'classifier.3']
    ]

class quantizeModel(nn.Module):

    def __init__(self, model):
        super(quantizeModel, self).__init__()
        # converts tensors from floating point to quantized.
        self.quant = torch.quantization.QuantStub()
        
        # converts tensors from quantized to floating point.
        self.dequant = torch.quantization.DeQuantStub()
        
        self.model = model

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
    

class ModelQuantization:

    @staticmethod
    def run(model):
        model_to_quantize = deepcopy(model)
        model_to_quantize.eval()
        model_to_quantize.to( torch.device('cpu') )

        fused_model = torch.quantization.fuse_modules(model_to_quantize, modules_to_fuse, inplace=True)

        quantized_model = quantizeModel(model=fused_model)

        quantization_config = torch.quantization.default_qconfig
        torch.backends.quantized.engine = 'qnnpack'
        quantized_model.qconfig = quantization_config

        torch.quantization.prepare( quantized_model, inplace=True )
        quantized_model = torch.quantization.convert( quantized_model, inplace=True )

        return quantized_model