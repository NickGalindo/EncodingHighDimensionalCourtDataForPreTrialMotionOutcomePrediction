import pandas as pd
import numpy as np

tabular_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/motionStrike_ALLcodes_data.tsv", sep="\t")
text_data_paths = pd.read_csv("/mnt/research/aguiarlab/proj/law/pdfs/all_txt_files_abs_path.txt", names=["filepath"], header=None)
mapping_DocumentNo_CaseRefNum = pd.read_csv("/mnt/research/aguiarlab/proj/law/jz_script/judcaseid_docid_translationtable.tsv", sep="\t")


print(f"tabular_data shape: {tabular_data.shape}")
print(f"text_data_paths shape: {text_data_paths.shape}")
print(f"mapping_DocumentNo_CaseRefNum shape: {mapping_DocumentNo_CaseRefNum.shape}")


text_data_paths["document_no"] = text_data_paths["filepath"].str.extract(r"(\d+)(?!.*\d)")
text_data_paths = text_data_paths.dropna()
text_data_paths["document_no"] = text_data_paths["document_no"].astype(int)

print(f"text_data_paths shape after building DocumentNo: {text_data_paths.shape}")

case_ref_num_not_in_mapping = set(tabular_data["CaseReferenceNumber"].tolist()) - set(mapping_DocumentNo_CaseRefNum["CaseRefNum "].tolist())
document_no_not_in_mapping = set(text_data_paths["document_no"].tolist()) - set(mapping_DocumentNo_CaseRefNum["DocumentNo"].tolist())

print(f"Amount of case reference numbers in tabular data not in the judcaseid_docid_translationtable: {len(case_ref_num_not_in_mapping)}")
print(f"Amount of document no in text data not in the judcaseid_docid_translationtable: {len(document_no_not_in_mapping)}")


full_data_with_documentNo = pd.merge(tabular_data, mapping_DocumentNo_CaseRefNum, left_on="CaseReferenceNumber", right_on="CaseRefNum ", how="inner")
a = full_data_with_documentNo.copy()

print(f"left join on tabular data and the document no to case reference number mapping shape: {full_data_with_documentNo.shape}")

full_data_with_documentNo = pd.merge(full_data_with_documentNo, text_data_paths, left_on="DocumentNo", right_on="document_no", how="inner")
b = full_data_with_documentNo.copy()


print(f"left join on previous calculated tabular data with document no on the text data and their numbers to correlate filepaths with tabular data shape: {full_data_with_documentNo.shape}")


print(full_data_with_documentNo["CaseReferenceNumber"][full_data_with_documentNo["CaseReferenceNumber"].duplicated()].unique())
