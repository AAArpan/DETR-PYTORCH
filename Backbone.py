import torch
import torchvision.models as models
import torch.nn as nn

class BackboneNetwork(nn.Module):
    def __init__(self, pretrained=True, freeze_bn=True, finetune=False):
        super(BackboneNetwork, self).__init__()

        self.resnet = models.resnet50(pretrained=pretrained)
        self.freeze_batchnorm(freeze_bn)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) # Remove the last two layers (avgpool and fc)
        self.set_requires_grad(finetune)

    def freeze_batchnorm(self, freeze_bn):
        """Freeze BatchNorm layers by setting them to eval mode."""
        if freeze_bn:
            for m in self.resnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()  

    def set_requires_grad(self, finetune):
        """Freeze or unfreeze layers for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = finetune 

    def forward(self, x):
        # Get the feature maps from ResNet50
        x = self.resnet(x)
        return x

