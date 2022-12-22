from torch.utils.data import Dataset

import csv
from PIL import Image
from matplotlib.colors import ListedColormap
import torchvision.transforms as T
import torch

import os
import glob

class LandCoverData(Dataset):

    # mapping between label class names and indices
    LABEL_CLASSES = {
      'Grass and other': 0,
      'Wald': 1,
      'Bushes and sparse forest': 2,
      'Water and wetlands': 3,
      'Glaciers and permanent snow': 4,
      'Sparse rocks (rocks mixed with grass)': 5,
      'Loose rocks, scree': 6,
      'Bed rocks': 7
    }
    
    colormap_names = ["green", "saddlebrown", "darkseagreen" ,"steelblue",
                      "snow", "tan", "lightgrey" ,"dimgrey",]
    
    colormap = ListedColormap(colormap_names)
    
    input_transform =  T.Compose([
        T.Resize((200, 200)),
        T.ToTensor()])

    label_transform = T.ToTensor()

    def __init__(self, path, transforms=None, split='train'):
        if transforms is not None:
            self.input_transform=transforms


        self.data = []     

        with open(f'{path}ipeo_data/splits/{split}.csv', newline='') as csvfile:
            csvReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvReader:
                rowStr=row[0]
                if (rowStr=="25595_11025"):
                    continue
                imgNameRGB = os.path.join(f'{path}ipeo_data/rgb/', f'{rowStr}_rgb.tif')
                imgNameLabel = os.path.join(f'{path}ipeo_data/alpine_label/', f'{rowStr}_label.tif')
                self.data.append((
                    imgNameRGB,
                    imgNameLabel
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        imgName, imgLabel = self.data[x]

        img = Image.open(imgName)
        label = Image.open(imgLabel)
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, self.label_transform(label)