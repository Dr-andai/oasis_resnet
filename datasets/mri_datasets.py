from pathlib import Path
import os, torch, numpy as np
from torch.utils.data import Dataset
from PIL import Image

# dataset class that loads volume session wise
class MRISessionDataset(Dataset):
    def __init__(self, dataframe, root_dir="/data/oasis_png_per_volume", transform=None):
        self.df = dataframe
        self.root_dir = Path(root_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        scan_path = self.root_dir / row["path"]

        age = torch.tensor(row["age"], dtype=torch.float32)
        diagnosis = torch.tensor(row["diagnosis"], dtype=torch.float32)

        # load slices
        slice_files = sorted(os.listdir(scan_path))
        slices = []
        for f in slice_files:
            img = Image.open(os.path.join(scan_path, f)).convert("L")
            img = np.array(img, dtype=np.float32) / 255.0
            slices.append(img)

        # Stack into a 3D volume
        volume = np.stack(slices, axis=0)  # (D, H, W)

        # Convert to torch and add channel dimension
        volume = torch.tensor(volume, dtype=torch.float32)    # (D, H, W)
        volume = volume.unsqueeze(0)                         # (1, D, H, W)

        if self.transform:
            volume = self.transform(volume)

        return {
            "volume":volume, 
            "age" : age,
            "diagnosis" : diagnosis
            }
