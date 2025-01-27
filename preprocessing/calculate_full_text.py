from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
import os
import pickle

train_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_train.csv")
train_corpus = train_data[["filepath", "document_no"]]

val_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_val.csv")
val_corpus = val_data[["filepath", "document_no"]]

test_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_test.csv")
test_corpus = test_data[["filepath", "document_no"]]


corpus = pd.concat([train_corpus, val_corpus, test_corpus], ignore_index=True)
corpus["document_no"] = corpus["filepath"].str.extract(r'(\d+)(?=\D*$)')
corpus = corpus.dropna()
corpus["document_no"] = corpus["document_no"].astype(int)

full_corpus = []
full_corpus_document_no = []

some_index = 0
for _, row in corpus.iterrows():
    print(f"DOCUMENT EXTRACTION PROGRESS {some_index}/{len(corpus)}")
    some_index += 1
    with open(row["filepath"], "r") as file: #type: ignore
        full_corpus.append(file.read())
        full_corpus_document_no.append(row["document_no"])

save_path = "/mnt/research/aguiarlab/proj/law/data/PaperData/textData/alltext"
os.makedirs(save_path, exist_ok=True)
indexed_corpus = {}
for doc, document_id in zip(full_corpus, full_corpus_document_no):
    indexed_corpus[document_id] = doc

with open(os.path.join(save_path, "indexed_text.pkl"), "wb") as file:
    pickle.dump(indexed_corpus, file)
