from torch.nn import functional as F
from torch.nn import Conv2d
from torch.nn import Module
import torch

from src.model_blocks import Block, Encoder, Decoder

class UNet(Module):
    def __init__(self, nbClasses=1,
                       nChannels=3,
                       flag_scale=True):
        super().__init__()
        self.nChannels = nChannels
        self.nbClasses = nbClasses
        self.flag_scale = flag_scale
        
        if self.flag_scale:
            self.inc = (Block(nChannels, 64))
            self.encoder1 = (Encoder(64, 128))
            self.encoder2 = (Encoder(128, 256))
            self.encoder3 = (Encoder(256, 512))
            self.encoder4 = (Encoder(512, 1024))
            self.decoder1 = (Decoder(1024, 512))
            self.decoder2 = (Decoder(512, 256))
            self.decoder3 = (Decoder(256, 128))
            self.decoder4 = (Decoder(128, 64))
            self.last = Conv2d(64, nbClasses, kernel_size=1)
        else:
            self.inc = (Block(nChannels, 32))
            self.encoder1 = (Encoder(32, 64))
            self.encoder2 = (Encoder(64, 128))
            self.encoder3 = (Encoder(128, 256))
            self.decoder1 = (Decoder(256, 128))
            self.decoder2 = (Decoder(128, 64))
            self.decoder3 = (Decoder(64, 32))
            self.last = Conv2d(32, nbClasses, kernel_size=1)

    def forward(self, x):
        
        if self.flag_scale:
            x1 = self.inc(x)
            x2 = self.encoder1(x1)
            x3 = self.encoder2(x2)
            x4 = self.encoder3(x3)
            x5 = self.encoder4(x4)
            x = self.decoder1(x5, x4)
            x = self.decoder2(x, x3)
            x = self.decoder3(x, x2)
            x = self.decoder4(x, x1)
        else:
            x1 = self.inc(x)
            x2 = self.encoder1(x1)
            x3 = self.encoder2(x2)
            x4 = self.encoder3(x3)
            x = self.decoder1(x4, x3)
            x = self.decoder2(x, x2)
            x = self.decoder3(x, x1)

        logits = self.last(x)
        return logits