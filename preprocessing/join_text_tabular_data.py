import pandas as pd
import numpy as np

tabular_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/motionStrike_ALLcodes_data.tsv", sep="\t")
text_data_paths = pd.read_csv("/mnt/research/aguiarlab/proj/law/pdfs/all_txt_files_abs_path.txt", names=["filepath"], header=None)
mapping_DocumentNo_CaseRefNum = pd.read_csv("/mnt/research/aguiarlab/proj/law/jz_script/judcaseid_docid_translationtable.tsv", sep="\t")


print(f"tabular_data shape: {tabular_data.shape}")
print(f"text_data_paths shape: {text_data_paths.shape}")
print(f"mapping_DocumentNo_CaseRefNum shape: {mapping_DocumentNo_CaseRefNum.shape}")


text_data_paths["document_no"] = text_data_paths["filepath"].str.extract(r"(\d+)(?!.*\d)").astype(int)
text_data_paths = text_data_paths.dropna()

print(f"text_data_paths shape after building DocumentNo: {text_data_paths.shape}")


full_data_with_documentNo = pd.merge(tabular_data, mapping_DocumentNo_CaseRefNum, left_on="CaseReferenceNumber", right_on="CaseRefNum ", how="inner")

print(f"left join on tabular data and the document no to case reference number mapping shape: {full_data_with_documentNo.shape}")
