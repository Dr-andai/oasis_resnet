import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(session_csv, out_dir):
    df = pd.read_csv("../data/splits/sessions.csv")

    # subjects in a unique list
    subjects = df["participant_id"].unique()

    # subject level split
    train_subject, test_subject = train_test_split(subjects, test_size = 0.15, random_state= 42)
    train_subject, val_subject = train_test_split(train_subject, test_size = 0.176, random_state= 42)

    def subset_df(subjects):
        return df[df["participant_id"].isin(subjects)].reset_index(drop=True)
    
    train_df = subset_df(train_subject)
    val_df = subset_df(val_subject)
    test_df = subset_df(test_subject)

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"[INFO] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    IN = "../data/splits/sessions.csv"
    OUT = "../data/splits"
    split_dataset(IN, OUT)



