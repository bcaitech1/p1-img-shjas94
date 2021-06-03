import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from efficientnet_pytorch import EfficientNet

__all__ = ['EffNet']


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class EffNet(nn.Module):
    def __init__(self, t=1.5):
        super(EffNet, self).__init__()
        self.t = t
        self.effnet = EfficientNet.from_pretrained(
            'efficientnet-b3', num_classes=18)
        # activation

    def forward(self, x):
        h = self.effnet(x)
        return h
