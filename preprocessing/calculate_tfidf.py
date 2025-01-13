# Calculate the 400 word with highest tfidf in the documents. This because the BERT context windows is 512 tokens which is roughly 400 words

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

train_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_train.csv")
val_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_val.csv")
test_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_test.csv")

corpus = set(train_data["filepath"].dropna().unique().tolist() + val_data["filepath"].dropna().unique().tolist() + test_data["filepath"].dropna().unique().tolist()) # type: ignore

full_corpus = []

for doc_path in corpus:
    with open(doc_file, "r") as file: #type:ignore
        full_corpus.append(file.read())


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(full_corpus)


