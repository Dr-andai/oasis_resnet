# local_infer.py
import sys
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT.parent / "checkpoints" / "checkpoints" / "best_model.pt" 
TEST_CSV = ROOT.parent / "data" / "splits" / "test.csv"
OUT_CSV = ROOT.parent / "data" / "inference_results_2.csv"
ROOT_DATA_DIR = "../data/oasis_png_per_volume" 

sys.path.append(str(ROOT.parent))
from model.resnet import MRI_AgeModel, resnet18_3d
from datasets.mri_datasets import MRISessionDataset

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load model
model = MRI_AgeModel(resnet18_3d()).to(device)
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT}")
state = torch.load(CHECKPOINT, map_location=device)

# Support both state_dict and full-model saves
if isinstance(state, dict) and not any(k.startswith("_") for k in state.keys()):
    # probably a state_dict
    model.load_state_dict(state)
else:
    model = state.to(device)
model.eval()

# Ttest CSV and dataset ----
test_df = pd.read_csv(TEST_CSV)

test_dataset = MRISessionDataset(test_df, root_dir=ROOT_DATA_DIR)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


# Inference ----
session_ids = []
participant_ids = []
true_ages = []
predictions = []

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader, desc="Infer sessions")):
        # dataset returns tuple (volume, age, diagnosis)
        volume, age, diagnosis = batch

        # move to device
        volume = volume.to(device)            # shape: [B, 1, D, H, W]
        age = age.to(device)
        diagnosis = diagnosis.to(device)

        # forward
        out = model(volume, diagnosis).squeeze(1)  # -> [B]
        out_cpu = out.cpu().numpy()

        # map back to CSV row (assuming same order)
        # if test_df has multiple rows per participant, index i maps to test_df.iloc[i]
        row = test_df.iloc[i]
        participant_ids.append(row.get("participant_id", row.get("participant", f"pid_{i}")))
        session_ids.append(row.get("session_id", row.get("path", f"sid_{i}")))
        true_ages.append(float(age.cpu().numpy().ravel()[0]))
        predictions.append(float(out_cpu.ravel()[0]))

# ---- Build results dataframe (session-level) ----
session_df = pd.DataFrame({
    "participant_id": participant_ids,
    "session_id": session_ids,
    "age": true_ages,
    "predicted_age": predictions
})

# ---- Aggregate per participant (mean prediction) ----
results_grouped = (
    session_df.groupby("participant_id", as_index=False)
    .agg({
        "age": "first",               # assume true age same across sessions
        "predicted_age": "mean"       # average across sessions
    })
)

# ---- Save outputs ----
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results_grouped.to_csv(OUT_CSV, index=False)
print(f"Saved averaged inference results to {OUT_CSV}")
print(results_grouped.head())
