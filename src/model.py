from torch.nn import functional as F
from torch.nn import Conv2d
from torch.nn import Module
import torch

from src.model_blocks import Block, Encoder, Decoder

class UNet(Module):
    def __init__(self, encChannels=(3,64,128,256,512,1024),
                       decChannels=(1024, 512, 256, 128, 64),
                       nbClasses=1, 
                       retainDim=True,
                       outSize=(200,  200)):
        
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
          encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        mask = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            mask = F.interpolate(mask, self.outSize)
        # return the segmentation map
        return mask