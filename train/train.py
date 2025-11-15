import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import modal
from modal import App, Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from datasets.mri_datasets import MRISessionDataset
from model.resnet import resnet18_3d

# --- Modal setup ---
app = modal.App("andaidavid8")
vol = modal.Volume.from_name("oasis-storage")
image = (
    Image.debian_slim()
    .pip_install("torch", "torchvision", "tqdm", "pandas", "scikit-learn")
    .add_local_dir(str(ROOT), remote_path="/root/OASIS_RESNET")
)

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0

    for volumes, ages in tqdm(loader, desc=f"Training [Epoch {epoch+1}/{num_epochs}]", leave=False):
        volumes, ages = volumes.to(device), ages.to(device)

        optimizer.zero_grad()
        outputs = model(volumes).squeeze(1)
        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * volumes.size(0)
    
    return running_loss / len(loader.dataset)

### Evaluation

def evaluate(model, loader, criterion, device, phase="Evaluating", epoch=None, num_epochs=None):
    model.eval()
    running_loss = 0.0
    preds, gts = [], []

    desc = phase
    if epoch is not None and num_epochs is not None:
        desc = f"{phase} [Epoch {epoch+1}/{num_epochs}]"

    with torch.no_grad():
        for volumes, ages in tqdm(loader, desc="Evaluating", leave=False):
            volumes, ages = volumes.to(device), ages.to(device)
            outputs = model(volumes).squeeze(1)

            loss = criterion(outputs, ages)
            running_loss += loss.item() * volumes.size(0)

            preds.extend(outputs.cpu().numpy())
            gts.extend(ages.cpu().numpy())
    
    avg_loss = running_loss / len(loader.dataset)
    mae = mean_absolute_error(gts, preds)
    return avg_loss, mae

## Train

@app.function(image=image, gpu="A10G") 
def train():
    import sys
    sys.path.append("/root/OASIS_RESNET")

    from datasets.mri_datasets import MRISessionDataset
    from model.resnet import resnet18_3d
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load csvs
    train_df = pd.read_csv("../data/splits/train.csv")
    val_df = pd.read_csv("../data/splits/val.csv")
    test_df = pd.read_csv("../data/splits/test.csv")

    # Build datasets and loaders
    train_dataset = MRISessionDataset(train_df, root_dir="../data/oasis_png_per_volume")
    val_dataset = MRISessionDataset(val_df, root_dir="../data/oasis_png_per_volume")
    test_dataset = MRISessionDataset(test_df, root_dir="../data/oasis_png_per_volume")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # Model, optimizer, loss
    model = resnet18_3d(num_classes=1, in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    criterion = nn.MSELoss()

    # Training Loop
    num_epochs = 10
    best_val_mae = float("inf")
    train_losses, val_losses, val_maes = [], [], []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, phase="Validating", epoch=epoch, num_epochs=num_epochs)

        print(f"Epoch {epoch+1}/{num_epochs}"
              f"| Train Loss: {train_loss: .4f}"
              f"| Val Loss: {val_loss: .4f}"
              f"| Val MAE: {val_mae: .2f}"
              )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "../checkpoints/best_model.pt")
    
    # ---- Save metrics to CSV ----
    df = pd.DataFrame({
        "Epoch": range(1, num_epochs + 1),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
        "Val MAE": val_maes
    })
    df.to_csv("metrics.csv", index=False)
    print("Metrics saved to metrics.csv")

    # save final model
    torch.save(model.state_dict(), "../checkpoints/final_model.pt")

    model.load_state_dict(torch.load("../checkpoints/best_model.pt"))
    test_loss, test_mae = evaluate(model, test_loader, criterion, device, phase="Testing")
    print(f"Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.2f}")

@app.local_entrypoint()
def main():
    """Runs locally by default."""
    train.remote()





