# Calculate the 400 word with highest tfidf in the documents. This because the BERT context windows is 512 tokens which is roughly 400 words

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

train_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_train.csv")
val_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_val.csv")
test_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_test.csv")

corpus = set(train_data["filepath"].unique().tolist() + val_data["filepath"].unique().tolist() + test_data["filepath"].unique().tolist())
