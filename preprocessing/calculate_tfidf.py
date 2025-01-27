from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re


corpus = pd.read_csv("/mnt/research/aguiarlab/proj/law/pdfs/all_txt_files_abs_path.txt", names=["filepath"], header=None)
corpus["document_no"] = corpus["filepath"].str.extract(r'(\d+)(?=\D*$)')
corpus = corpus.dropna()
corpus["document_no"] = corpus["document_no"].astype(int)

full_corpus = []
full_corpus_document_no = []

for _, row in corpus.iterrows():
    with open(row["filepath"], "r") as file: #type: ignore
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", file.read())
        cleaned = re.sub(r"\b[a-zA-Z]\b", "", cleaned)
        full_corpus.append(cleaned)
        full_corpus_document_no.append(row["document_no"])

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(full_corpus)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

corpus_tfidf = []
for i, doc in enumerate(full_corpus):
    doc_tfidf_scores = tfidf_matrix[i].toarray().flatten()
    word_scores = list(zip(feature_names, doc_tfidf_scores))

    sorted_words = [word for word, score in sorted(word_scores, key=lambda x: x[1], reverse=True)]

    sorted_words = sorted_words[:512]
    sorted_words_set = set(sorted_words)

    doc_tfidf = ""

    for word in doc.split():
        if word in sorted_words_set:
            doc_tfidf += word + " "

    doc_tfidf_aux = doc_tfidf.split()
    sorted_words_pos = len(sorted_words)-1
    while len(doc_tfidf_aux) > 512:
        doc_tfidf = ""
        sorted_words_set.remove(sorted_words[sorted_words_pos])

        for word in doc_tfidf_aux:
            if word in sorted_words_set:
                doc_tfidf += word + " "

        doc_tfidf_aux = doc_tfidf.split()
    
    corpus_tfidf.append(doc_tfidf)
