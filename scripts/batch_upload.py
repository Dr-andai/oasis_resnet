import modal
from pathlib import Path

app = modal.App("andaidavid8")
vol = modal.Volume.from_name("oasis-storage")

oasis_png = Path("../data/oasis_png_per_volume/")
splits = Path("../data/splits/")

@app.local_entrypoint()
def main():  
    remote_dir = "/data/oasis_png_per_volume"
    remote_dir_splits = "/data/splits"  
    
    with vol.batch_upload() as batch:
        batch.put_directory(str(oasis_png), remote_dir)
        batch.put_directory(str(splits), remote_dir_splits)
    print("✅ Upload completed!")