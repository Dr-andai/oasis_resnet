import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import modal
from modal import Image, Volume
import pandas as pd

app = modal.App("andaidavid8")
vol = Volume.from_name("oasis-storage")

image = (
    Image.debian_slim()
    .pip_install("torch", "torchvision", "pandas", "numpy", "Pillow")
    .workdir("/app")
    .add_local_dir(str(ROOT / "model"), "/app/model")
    .add_local_file(str(ROOT / "model/infer.py"), "/app/model/infer.py")
)

@app.function(gpu="H100", image=image, volumes={"/data": vol}, timeout=60 * 60 * 12)
def batch_infer(csv_path="/data/data/splits/test.csv"):
    import sys
    sys.path.append("/app")
    from model.infer import predict_age
    import pandas as pd
    import shutil

    local_dir = "/tmp/volume"
    remote_dir = "/data/data/oasis_png_per_volume"

    if not os.path.exists(local_dir):
        shutil.copytree(remote_dir, local_dir)


    df = pd.read_csv(csv_path)
    preds = []

    for _, row in df.iterrows():
        vol_path = f"/tmp/volume/{row['path'].replace('\\','/')}"
        pred = predict_age(vol_path, diagnosis=row["diagnosis"])
        preds.append(pred)

    os.makedirs("/data/results", exist_ok=True)

    df["pred_age"] = preds
    out_path = "/data/results/test_predictions.csv"
    df.to_csv(out_path, index=False)
    
    return f"Saved results to {out_path}"

@app.local_entrypoint()
def main():
    print(batch_infer.remote())
