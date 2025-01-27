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
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", file.read())
        cleaned = re.sub(r"\b[a-zA-Z]\b", "", cleaned)
        full_corpus.append(cleaned)
        full_corpus_document_no.append(row["document_no"])

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(full_corpus)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

corpus_tfidf = []
for i, doc in enumerate(full_corpus):
    print(f"SUBSETTING PROGRESS {i}/{len(full_corpus)}")
    doc_tfidf_scores = tfidf_matrix[i].toarray().flatten() # type: ignore
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
        sorted_words_pos -= 1

        for word in doc_tfidf_aux:
            if word in sorted_words_set:
                doc_tfidf += word + " "

        doc_tfidf_aux = doc_tfidf.split()
    
    corpus_tfidf.append(doc_tfidf)

save_path = "/mnt/research/aguiarlab/proj/law/data/PaperData/textData/tfidf"
os.makedirs(save_path, exist_ok=True)
indexed_corpus_tfidf = {}
for doc, document_id in zip(corpus_tfidf, full_corpus_document_no):
    indexed_corpus_tfidf[document_id] = doc

with open(os.path.join(save_path, "indexed_text.pkl"), "wb") as file:
    pickle.dump(indexed_corpus_tfidf, file)
