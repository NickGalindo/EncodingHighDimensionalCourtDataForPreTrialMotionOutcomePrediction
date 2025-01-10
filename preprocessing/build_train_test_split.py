import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

#full_data_path: str = "/mnt/research/aguiarlab/proj/law/data/PaperData/motionStrike_TVcodes_data.tsv"
full_data_path: str = "/mnt/research/aguiarlab/proj/law/data/PaperData/motionStrike_ALLcodes_data.tsv"

df: pd.DataFrame = pd.read_csv(full_data_path, sep="\t")

print(f"Original dataset length: {len(df)}")

df = df[df["MotionResultCode"].isin(["GR", "DN"])] # type: ignore

print(f"Lenght of dataset after subseting on GR and DN: {len(df)}")

train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["MotionResultCode"])

test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42, stratify=test_val_df["MotionResultCode"]) # type: ignore

print(f"Length of train dataset: {len(train_df)}")
print(f"Length of test dataset: {len(test_df)}")
print(f"Length of val dataset: {len(val_df)}")

train_df.to_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/full_train.csv") # type: ignore
test_df.to_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/full_test.csv") # type: ignore
val_df.to_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/full_val.csv") #type: ignore
