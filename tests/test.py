from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import os

import pandas as pd
import numpy as np
import pickle

#base_dir = "/home/coder/base/PaperData"
base_dir = "/mnt/research/aguiarlab/proj/law/data/PaperData/"

train_data = pd.read_csv(os.path.join(base_dir, "mapped_full_train.csv"))
train_data = train_data[["filepath", "document_no", "MotionResultCode"]]
train_data["label"] = train_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)

val_data = pd.read_csv(os.path.join(base_dir, "mapped_full_val.csv"))
val_data = val_data[["filepath", "document_no", "MotionResultCode"]]
val_data["label"] = val_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)

test_data = pd.read_csv(os.path.join(base_dir, "mapped_full_test.csv"))
test_data = test_data[["filepath", "document_no", "MotionResultCode"]]
test_data["label"] = test_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)

full_corpus = pickle.load(open(os.path.join(os.path.join(base_dir, "textData/alltext"), "indexed_text.pkl"), "rb"))

print(train_data)

train_data["text"] = train_data["document_no"].map(full_corpus)
val_data["text"] = val_data["document_no"].map(full_corpus)
test_data["text"] = test_data["document_no"].map(full_corpus)

print(train_data)

train_data = train_data[["text", "label"]].dropna()
val_data = val_data[["text", "label"]].dropna()
test_data = test_data[["text", "label"]].dropna()

print(train_data)

print(f"TRAIN DATASET SIZE: {len(train_data)}")
print(f"VAL DATASET SIZE: {len(val_data)}")
print(f"TEST DATASET SIZE: {len(test_data)}")
