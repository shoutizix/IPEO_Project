# IPEO-Project: Semantic segmentation of alpine land cover.

Repository link: https://github.com/shoutizix/IPEO_Project/edit/main/README.md

# Setup

```
# create a local virtual environment in the venv folder
python -m venv venv
# activate this environment
source venv/bin/activate
# install requirements
pip install pandas seaborn torch torchmetrics torchvision torchsummary torchtext pytorch_lightning torchmetrics tensorboard matplotlib tqdm datetime time
```

# File Structure:

```
checkpoints/
data/
src/
evaluation.ipynb
report.pdf
```

# Data
Datasets available here :  https://enacshare.epfl.ch/drCz5HgLJyFPXifNBWad7

The structure of the files should be as follows:
```
data/
     ipeo_data/
               augmented_data_label/
               augmented_data_rgb/
               alpine_label/
               rgb/
               splits/
```

# Model Weights 

Link for the model weights trained with cross entropy loss : https://drive.google.com/file/d/1zaTizD0_rBXPleV6hwcX8W7usJ_lgcvG/view?usp=share_link

The structure of the files should be as follows
```
checkpoints/
            model_weights.pth
```
