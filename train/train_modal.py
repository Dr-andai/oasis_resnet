import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import modal
from modal import Image, Volume
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

# Import model + dataset
from model.resnet import MRI_AgeModel, resnet18_3d
from datasets.mri_datasets import MRISessionDataset

# Modal setup
app = modal.App("andaidavid8")
vol = Volume.from_name("oasis-storage")

image = (
    Image.debian_slim()
    .pip_install("torch", "torchvision", "pandas", "tqdm", "scikit-learn")
    .add_local_dir(str(ROOT), remote_path="/app")
)

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_loss = 0

    for volume, age, diagnosis in tqdm(loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False):
        volume = volume.to(device)
        age = age.to(device)
        diagnosis = diagnosis.to(device)

        optimizer.zero_grad()
        pred = model(volume, diagnosis).squeeze(1)

        loss = criterion(pred, age)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * volume.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, label="Val"):
    model.eval()
    total_loss = 0
    preds, gts = [], []

    with torch.no_grad():
        for volume, age, diagnosis in tqdm(loader, desc=label, leave=False):
            volume = volume.to(device)
            age = age.to(device)
            diagnosis = diagnosis.to(device)

            pred = model(volume, diagnosis).squeeze(1)
            loss = criterion(pred, age)

            total_loss += loss.item() * volume.size(0)
            preds.extend(pred.cpu().numpy())
            gts.extend(age.cpu().numpy())

    return total_loss / len(loader.dataset), mean_absolute_error(gts, preds)


@app.function(image=image, gpu="A10G", volumes={"/data": vol})
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load CSVs from volume
    train_df = pd.read_csv("/data/splits/train.csv")
    val_df   = pd.read_csv("/data/splits/val.csv")
    test_df  = pd.read_csv("/data/splits/test.csv")

    # Dataset paths in volume
    root = "/data/oasis_png_per_volume"

    train_ds = MRISessionDataset(train_df, root_dir=root)
    val_ds   = MRISessionDataset(val_df, root_dir=root)
    test_ds  = MRISessionDataset(test_df, root_dir=root)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

    model = MRI_AgeModel(resnet18_3d()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_val = float("inf")
    epochs = 10

    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)
        vl, vmae = evaluate(model, val_loader, criterion, device, "Val")

        print(f"Epoch {epoch+1}/{epochs} | Train {tr:.4f} | Val {vl:.4f} | MAE={vmae:.2f}")

        if vmae < best_val:
            best_val = vmae
            torch.save(model.state_dict(), "/data/checkpoints/best.pt")

    print("Testing best model…")
    model.load_state_dict(torch.load("/data/checkpoints/best.pt"))
    tl, tmae = evaluate(model, test_loader, criterion, device, "Test")
    print(f"Test Loss {tl:.4f} | Test MAE {tmae:.2f}")


@app.local_entrypoint()
def main():
    train.remote()
