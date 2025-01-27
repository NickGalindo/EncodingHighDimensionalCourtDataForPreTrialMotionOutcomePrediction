from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re
import os
import pickle
from scipy import sparse

class TFIGMVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1):
        self.min_df = min_df

    def fit(self, documents, y=None):
        # Fit the count vectorizer to calculate term frequencies
        self.vectorizer = CountVectorizer(min_df=self.min_df)
        self.X_counts = self.vectorizer.fit_transform(documents)
        return self

    def transform(self, documents):
        # Compute the term frequency (TF) matrix
        X_counts = self.vectorizer.transform(documents)
        term_frequencies = X_counts.toarray()

        # Calculate the maximum frequency for each term across all documents
        max_frequencies = np.max(term_frequencies, axis=0)

        # Calculate the Inverse Gravity Moment (IGM) for each term
        inverse_gravity_moments = []
        
        for t_idx in range(term_frequencies.shape[1]):
            # For each term, calculate the sum of 1/dist(t,d)
            dist = term_frequencies[:, t_idx] / max_frequencies[t_idx]
            igm = np.sum(np.where(dist != 0, 1 / dist, 0))  # Avoid division by zero
            inverse_gravity_moments.append(igm)

        inverse_gravity_moments = np.array(inverse_gravity_moments)

        # Compute TF-IGM: multiply TF by IGM
        tf_igm_matrix = term_frequencies * inverse_gravity_moments
        return tf_igm_matrix

# Load and prepare data
train_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_train.csv")
train_corpus = train_data[["filepath", "document_no"]]
val_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_val.csv")
val_corpus = val_data[["filepath", "document_no"]]
test_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_test.csv")
test_corpus = test_data[["filepath", "document_no"]]

corpus = pd.concat([train_corpus, val_corpus, test_corpus], ignore_index=True)
corpus = corpus.dropna()

full_corpus = []
full_corpus_document_no = []
some_index = 0
for _, row in corpus.iterrows():
    print(f"DOCUMENT EXTRACTION PROGRESS {some_index}/{len(corpus)}")
    some_index += 1
    with open(row["filepath"], "r") as file:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", file.read())
        cleaned = re.sub(r"\b[a-zA-Z]\b", "", cleaned)
        full_corpus.append(cleaned)
        full_corpus_document_no.append(row["document_no"])

# Use TF-IGM instead of TF-IDF
tfigm_vectorizer = TFIGMVectorizer()
tfigm_matrix = tfigm_vectorizer.fit_transform(full_corpus)
feature_names = np.array(tfigm_vectorizer.vectorizer.get_feature_names_out())

corpus_tfigm = []
for i, doc in enumerate(full_corpus):
    print(f"SUBSETTING PROGRESS {i}/{len(full_corpus)}")
    doc_tfigm_scores = tfigm_matrix[i].toarray().flatten()
    word_scores = list(zip(feature_names, doc_tfigm_scores))
    sorted_words = [word for word, score in sorted(word_scores, key=lambda x: x[1], reverse=True)]
    sorted_words = sorted_words[:512]
    sorted_words_set = set(sorted_words)
    
    doc_tfigm = ""
    for word in doc.split():
        if word in sorted_words_set:
            doc_tfigm += word + " "
            
    doc_tfigm_aux = doc_tfigm.split()
    sorted_words_pos = len(sorted_words)-1
    while len(doc_tfigm_aux) > 512:
        doc_tfigm = ""
        sorted_words_set.remove(sorted_words[sorted_words_pos])
        sorted_words_pos -= 1
        for word in doc_tfigm_aux:
            if word in sorted_words_set:
                doc_tfigm += word + " "
        doc_tfigm_aux = doc_tfigm.split()
    
    corpus_tfigm.append(doc_tfigm)

# Save results
save_path = "/mnt/research/aguiarlab/proj/law/data/PaperData/textData/tfigm"
os.makedirs(save_path, exist_ok=True)

indexed_corpus_tfigm = {}
for doc, document_id in zip(corpus_tfigm, full_corpus_document_no):
    indexed_corpus_tfigm[document_id] = doc

with open(os.path.join(save_path, "indexed_text.pkl"), "wb") as file:
    pickle.dump(indexed_corpus_tfigm, file)
