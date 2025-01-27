from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re
import os
import pickle
from scipy import sparse

class TFIGMVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df
        self.count_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        
    def fit(self, raw_documents):
        # Get term frequencies
        self.tf_matrix = self.count_vectorizer.fit_transform(raw_documents)
        self.feature_names = self.count_vectorizer.get_feature_names_out()
        
        # Calculate document lengths (total terms in each document)
        self.doc_lengths = np.array(self.tf_matrix.sum(axis=1)).flatten()
        
        # Calculate max frequency for each term
        self.max_term_freqs = np.array(self.tf_matrix.max(axis=0)).flatten()
        
        # Calculate IGM values
        self.igm_values = self._calculate_igm()
        
        return self
    
    def _calculate_igm(self):
        n_terms = len(self.feature_names)
        igm_values = np.zeros((self.tf_matrix.shape[0], n_terms))
        
        # Convert sparse matrix to dense for easier calculations
        tf_array = self.tf_matrix.toarray()
        
        for term_idx in range(n_terms):
            # Get max frequency for this term
            max_freq = self.max_term_freqs[term_idx]
            
            if max_freq > 0:
                # Calculate distance function for each document
                distances = tf_array[:, term_idx] / max_freq
                
                # Calculate sum of 1/distance for documents where term appears
                mask = distances > 0
                if np.any(mask):
                    igm_sum = np.sum(1 / distances[mask])
                    
                    # Calculate normalized term frequency
                    norm_tf = tf_array[:, term_idx] / self.doc_lengths
                    
                    # Final TF-IGM calculation
                    igm_values[:, term_idx] = norm_tf * igm_sum
        
        return igm_values
    
    def transform(self, raw_documents):
        # Get term frequencies for new documents
        tf_matrix = self.count_vectorizer.transform(raw_documents)
        doc_lengths = np.array(tf_matrix.sum(axis=1)).flatten()
        
        n_docs, n_terms = tf_matrix.shape
        tfigm_matrix = np.zeros((n_docs, n_terms))
        tf_array = tf_matrix.toarray()
        
        for term_idx in range(n_terms):
            max_freq = self.max_term_freqs[term_idx]
            
            if max_freq > 0:
                distances = tf_array[:, term_idx] / max_freq
                mask = distances > 0
                if np.any(mask):
                    igm_sum = np.sum(1 / distances[mask])
                    norm_tf = tf_array[:, term_idx] / doc_lengths
                    tfigm_matrix[:, term_idx] = norm_tf * igm_sum
        
        return sparse.csr_matrix(tfigm_matrix)
    
    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return sparse.csr_matrix(self.igm_values)
    
    def get_feature_names_out(self):
        return self.feature_names

# Load and prepare data
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
    with open(row["filepath"], "r") as file:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", file.read())
        cleaned = re.sub(r"\b[a-zA-Z]\b", "", cleaned)
        full_corpus.append(cleaned)
        full_corpus_document_no.append(row["document_no"])

# Use TF-IGM instead of TF-IDF
tfigm_vectorizer = TFIGMVectorizer()
tfigm_matrix = tfigm_vectorizer.fit_transform(full_corpus)
feature_names = np.array(tfigm_vectorizer.get_feature_names_out())

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
