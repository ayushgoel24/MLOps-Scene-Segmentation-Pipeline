import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from torch import nn
from typing import Any, Optional
from _utils import IntermediateLayerGetter
from quantized_resnet import load_state_dict_from_url
from deeplabv3 import DeepLabHead, DeepLabV3
import quantized_resnet

__all__ = ['DeepLabV3ResNet50']

model_urls = {
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth'}


class DeepLabV3ResNet50:
    def __init__(
        self,
        pretrained: bool = False,
        progress: bool = True,
        num_classes: int = 21,
        aux_loss: Optional[bool] = None,
        **kwargs: Any
    ):
        """Constructs a DeepLabV3 model with a ResNet-50 backbone.

        Args:
            pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
                contains the same classes as Pascal VOC
            progress (bool): If True, displays a progress bar of the download to stderr
            num_classes (int): number of output classes of the model (including the background)
            aux_loss (bool): If True, it uses an auxiliary loss
        """
        self.pretrained = pretrained
        self.progress = progress
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.kwargs = kwargs
        self.model = self._load_model()

    def _segm_model(
        self,
        name: str,
        backbone_name: str,
        num_classes: int,
        aux: Optional[bool],
        pretrained_backbone: bool = True
    ) -> nn.Module:
        if 'resnet' in backbone_name:
            backbone = quantized_resnet.__dict__[backbone_name](
                pretrained=pretrained_backbone,
                replace_stride_with_dilation=[False, True, True])
            out_layer = 'layer4'
            out_inplanes = 2048
            aux_layer = 'layer3'
            aux_inplanes = 1024
        else:
            raise NotImplementedError(f'backbone {backbone_name} is not supported as of now')

        return_layers = {out_layer: 'out'}
        if aux:
            return_layers[aux_layer] = 'aux'
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        aux_classifier = None
        if aux:
            aux_classifier = FCNHead(aux_inplanes, num_classes)

        model_map = {
            'deeplabv3': (DeepLabHead, DeepLabV3)
        }
        classifier = model_map[name][0](out_inplanes, num_classes)
        base_model = model_map[name][1]

        model = base_model(backbone, classifier, aux_classifier)
        return model

    def _load_model(
        self
    ) -> nn.Module:
        if self.pretrained:
            self.aux_loss = True
            self.kwargs["pretrained_backbone"] = False
        model = self._segm_model('deeplabv3', 'resnet50', self.num_classes, self.aux_loss, **self.kwargs)
        if self.pretrained:
            self._load_weights(model)
        return model

    def _load_weights(self, model: nn.Module) -> None:
        arch = 'deeplabv3_resnet50_coco'
        model_url = model_urls.get(arch, None)
        if model_url is None:
            raise NotImplementedError(f'pretrained {arch} is not supported as of now')
        else:
            state_dict = load_state_dict_from_url(model_url, progress=self.progress)
            model.load_state_dict(state_dict)


if __name__ == '__main__':
    model = DeepLabV3ResNet50(pretrained=True)
