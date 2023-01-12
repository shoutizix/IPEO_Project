import torch

from torch.nn import Sequential
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torchvision.transforms import CenterCrop
from torch.nn import functional as F

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.double_conv = Sequential(
            Conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(outChannels),
            ReLU(inplace=True),
            Conv2d(outChannels, outChannels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(outChannels),
            ReLU(inplace=True)
        )
    def forward(self, x):
        # apply CONV => BN => ReLU x2 blocks to the inputs and return it
        return self.double_conv(x)
    
class Encoder(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.maxpool_conv = Sequential(
            MaxPool2d(2),
            Block(inChannels, outChannels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Decoder(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.up = ConvTranspose2d(inChannels, outChannels, kernel_size=2, stride=2)
        self.conv = Block(inChannels, outChannels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)