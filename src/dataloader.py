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
    
    mean=torch.tensor([0.43315597, 0.48058005, 0.42299773])
    std=torch.tensor([0.23198403, 0.22022461, 0.19084829])
    normalize = T.Normalize(mean, std)
    # we invert normalization for plotting later
    std_inv = 1 / (std + 1e-7)
    unnormalize = T.Normalize(-mean * std_inv, std_inv)
    
    meanAugmented=torch.tensor([0.44071006, 0.49269303, 0.420523])
    stdAugmented=torch.tensor([0.21948738, 0.20629372, 0.18073172])
    normalizeAugmented = T.Normalize(meanAugmented, stdAugmented)
    std_invAug = 1 / (stdAugmented + 1e-7)
    unnormalizeAugmented = T.Normalize(-meanAugmented * std_invAug, std_invAug)
    
    input_transform = T.Compose([
        T.Resize((200, 200)),
        T.ToTensor()])
    
    input_transform_norm = T.Compose([
        T.Resize((200, 200)),
        T.ToTensor(),
        normalize])
    
    input_transform_norm_augmented = T.Compose([
        T.Resize((200, 200)),
        T.ToTensor(),
        normalizeAugmented])

    label_transform = T.ToTensor()

    def __init__(self, path, transforms=input_transform_norm, split='train', ignore_last_number=0, use_augmented=False):
        if transforms is not None:
            if use_augmented:
                self.input_transform=self.input_transform_norm_augmented
            else:
                self.input_transform=self.input_transform_norm

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
            if ignore_last_number != 0:
                self.data = self.data[:-ignore_last_number]
        if use_augmented:
            path_augmented_rgb = f"{path}ipeo_data/augmented_data_rgb/"
            path_augmented_label = f"{path}ipeo_data/augmented_data_label/"
            
            name_list = os.listdir(path_augmented_rgb)
            for name in name_list:
                name_without_extension = name[:-4]
                imgNameRGB = path_augmented_rgb+name_without_extension+".tif"
                imgNameLabel = path_augmented_label+name_without_extension[:-3]+"label.tif"
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