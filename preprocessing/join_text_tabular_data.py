import pandas as pd
import numpy as np


train_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/full_train.csv")
val_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/full_val.csv")
test_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/full_test.csv")


text_data_paths = pd.read_csv("/mnt/research/aguiarlab/proj/law/pdfs/all_txt_files_abs_path.txt", names=["filepath"], header=None)
text_data_paths["document_no"] = text_data_paths["filepath"].str.extract(r'(\d+)(?=\D*$)')
text_data_paths = text_data_paths.dropna()
text_data_paths["document_no"] = text_data_paths["document_no"].astype(int)
mapping_DocumentNo_CaseRefNum = pd.read_csv("/mnt/research/aguiarlab/proj/law/jz_script/judcaseid_docid_translationtable.tsv", sep="\t")


full_mapper = pd.merge(mapping_DocumentNo_CaseRefNum, text_data_paths, left_on="DocumentNo", right_on="document_no", how="left")


print(f"shape train_data:\t\t{train_data.shape}")
print(f"shape val_data:\t\t\t{val_data.shape}")
print(f"shape test_data:\t\t{test_data.shape}")

print()

print(f"Number of CaseReferenceNumber without file:\t{full_mapper['filepath'].isna().sum()}")

print()

train_data_full = pd.merge(train_data, full_mapper, left_on="CaseReferenceNumber", right_on="CaseRefNum ", how="left")
val_data_full = pd.merge(val_data, full_mapper, left_on="CaseReferenceNumber", right_on="CaseRefNum ", how="left")
test_data_full = pd.merge(test_data, full_mapper, left_on="CaseReferenceNumber", right_on="CaseRefNum ", how="left")

print(f"shape mapped train data:\t{train_data_full.shape}")
print(f"shape mapped val data:\t\t{val_data_full.shape}")
print(f"shape mapped test data:\t\t{test_data_full.shape}")

print()

print(f"mapped train data without file:\t{train_data_full['filepath'].isna().sum()}")
print(f"mapped val data without file:\t{val_data_full['filepath'].isna().sum()}")
print(f"mapped test data without file:\t{test_data_full['filepath'].isna().sum()}")
