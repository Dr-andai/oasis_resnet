import sys
from pathlib import Path

# ROOT = Path(__file__).resolve().parent.parent
# sys.path.append(str(ROOT))

import os, re, csv, pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT = Path("../data/oasis_png_per_volume")
OUT = Path("../data/splits/sessions.csv")
DEMOGRAPHICS = Path("../data/oasis_label.csv")

# Loading demographics into memory once
demo_df = pd.read_csv(DEMOGRAPHICS)
demo_df.columns = demo_df.columns.str.lower()
demo_df["subject id"] = demo_df["subject id"].str.strip()

def lookup_age(participant_id: str):
    base_id = participant_id.split("_MR*")[0]  # from OAS2_0001_MR1 → OAS2_0001 
    row = demo_df.loc[demo_df["subject id"] == base_id]
    if not row.empty:
        return float(row["age"].values[0])
    else:
        return None

def lookup_diagnosis(participant_id: str):
    base_id = participant_id.split("_MR*")[0]
    row = demo_df.loc[demo_df["subject id"] == base_id]
    if not row.empty:
        return float(row["diagnosis"].values[0])
    else:
        return None

def build_session_csv(data_dir: Path, out_csv: Path):
    rows = []

    for visit_dir in tqdm(sorted(data_dir.glob("OAS2_*"))):
        if not visit_dir.is_dir():
            continue

        match = re.match(r"(OAS2_\d+_MR\d+)", visit_dir.name)
        if not match:
            continue

        participant_id = match.group(1)

        for t1_dir in sorted(visit_dir.glob("t1_*")):
            if not t1_dir.is_dir():
                continue

            session_id = f"{participant_id}_{t1_dir.name}"
            age = lookup_age(participant_id)
            diag = lookup_diagnosis(participant_id)
            
            rows.append({
                "participant_id": participant_id,
                "session_id": session_id,
                "age": age,
                "diagnosis": diag,
                "path": str(t1_dir.relative_to(ROOT))
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "session_id", "age", "diagnosis", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    build_session_csv(ROOT, OUT)
