import torch

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
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.batchNorm = BatchNorm2d(outChannels)
		self.relu = ReLU()
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.relu(self.batchNorm(self.conv1(x)))
    
class Encoder(Module):
	def __init__(self, channels=(3,64,128,256,512,1024)):
		super().__init__()
		# store the encoder blocks and maxpooling layer
		self.encoderStep = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(kernel_size=3, stride=2)
	
	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		# loop through the encoder blocks
		for step in self.encoderStep:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = step(x)
			blockOutputs.append(x)
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs
    
class Decoder(Module):
	def __init__(self, channels=(1024, 512, 256, 128, 64)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.fusingConvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.decoderStep = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.fusingConvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block

			# STACKING
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)

			x = self.decoderStep[i](x)
		# return the final decoder output
		return x
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures