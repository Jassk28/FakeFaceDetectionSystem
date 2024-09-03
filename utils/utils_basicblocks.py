import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)

def conv1x1(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride,
                     padding = 0, bias = False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out