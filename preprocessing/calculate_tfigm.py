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
        
        # Calculate IGM for each term
        self.igm_values = self._calculate_igm()
        
        return self
    
    def _calculate_igm(self):
        n_docs = self.tf_matrix.shape[0]
        igm_values = np.zeros(len(self.feature_names))
        
        for term_idx in range(len(self.feature_names)):
            # Get term frequencies for each document
            term_freqs = self.tf_matrix.getcol(term_idx).toarray().flatten()
            
            # Find documents where term appears
            doc_positions = np.where(term_freqs > 0)[0]
            
            if len(doc_positions) > 0:
                # Get max frequency for this term across all documents
                max_freq = np.max(term_freqs)
                
                if max_freq > 0:
                    # Calculate relative frequencies (f(t,d)/max_f(t,d))
                    relative_freqs = term_freqs[doc_positions] / max_freq
                    
                    # Calculate distances between consecutive appearances using relative frequencies
                    distances = np.diff(relative_freqs)
                    
                    if len(distances) > 0:
                        # Calculate gravity moment using the relative frequency differences
                        gravity_moment = np.sum(1 / (distances ** 2))
                        igm_values[term_idx] = np.log(1 + (n_docs / gravity_moment))
                    else:
                        # If term appears in only one document
                        igm_values[term_idx] = np.log(1 + n_docs)
            
        return igm_values
    
    def transform(self, raw_documents):
        # Get term frequencies for new documents
        tf_matrix = self.count_vectorizer.transform(raw_documents)
        
        # Multiply TF by IGM values
        tf_igm_matrix = tf_matrix.multiply(sparse.csr_matrix(self.igm_values))
        
        return tf_igm_matrix
    
    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)
    
    def get_feature_names_out(self):
        return self.feature_names

# Modified main code
corpus = pd.read_csv("/mnt/research/aguiarlab/proj/law/pdfs/all_txt_files_abs_path.txt", names=["filepath"], header=None)
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
