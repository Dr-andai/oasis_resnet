import sys
sys.path.append("/app")

import os
import torch
import numpy as np
from PIL import Image

from model.resnet import MRI_AgeModel, resnet18_3d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_volume(folder_path):
    slice_files = sorted(os.listdir(folder_path))
    slices = []

    for f in slice_files:
        img = Image.open(os.path.join(folder_path, f)).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        slices.append(img)
    
    volume = np.stack(slices, axis=0)
    volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return volume

def predict_age(volume_folder, model_ckpt="/data/checkpoints/best.pt", diagnosis=0):
    model = MRI_AgeModel(resnet18_3d()).to(DEVICE)
    model.load_state_dict(torch.load(model_ckpt, map_location=DEVICE))
    model.eval()

    volume = load_volume(volume_folder).to(DEVICE)
    diag = torch.tensor([diagnosis], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred = model(volume, diag).item()

    return pred
