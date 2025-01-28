import torch
#import intel_extension_for_pytorch as ipex
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import torch
import os
from datasets import Dataset
import evaluate
import pandas as pd
import numpy as np



os.environ["WANDB_PROJECT"] = 'LegalTextEncoding'

base_dir = "/home/coder/second/PaperData"
#base_dir = "/home/vyral/Documents/UCONN/fullData/PaperData3/PaperData"
#base_dir = "/mnt/research/aguiarlab/proj/law/data/PaperData/"

train_data = pd.read_csv(os.path.join(base_dir, "mapped_full_train.csv"))
train_data = train_data[["MotionID", "document_no", "MotionResultCode"]]
train_data["label"] = train_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)

val_data = pd.read_csv(os.path.join(base_dir, "mapped_full_val.csv"))
val_data = val_data[["MotionID", "document_no", "MotionResultCode"]]
val_data["label"] = val_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)

test_data = pd.read_csv(os.path.join(base_dir, "mapped_full_test.csv"))
test_data = test_data[["MotionID", "document_no", "MotionResultCode"]]
test_data["label"] = test_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)



full_corpus = pickle.load(open(os.path.join(os.path.join(base_dir, "textData/tfidf"), "indexed_text.pkl"), "rb"))



train_data["text"] = train_data["document_no"].map(full_corpus)
val_data["text"] = val_data["document_no"].map(full_corpus)
test_data["text"] = test_data["document_no"].map(full_corpus)

train_data = train_data[["text", "label", "MotionID"]].dropna().reset_index()
val_data = val_data[["text", "label", "MotionID"]].dropna().reset_index()
test_data = test_data[["text", "label", "MotionID"]].dropna().reset_index()


model_path = "/home/coder/second/PaperData/bertTraining/models/tfidf/extra"
#model_path = "/home/vyral/Documents/UCONN/fullData/PaperData3/PaperData/bertTraining/models/alltext/extra"
tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, "model")).bert

model.cuda()
torch.cuda.empty_cache()
print("INITIATED CUDA")

print(train_data)

train_tokenized = tokenizer(train_data["text"].tolist(), padding="max_length", truncation=True, max_length=512, return_tensors="pt")
val_tokenized = tokenizer(val_data["text"].tolist(), padding="max_length", truncation=True, max_length=512, return_tensors="pt")
test_tokenized = tokenizer(test_data["text"].tolist(), padding="max_length", truncation=True, max_length=512, return_tensors="pt")

print(f"FILTERED TRAIN DATASET SIZE: {len(train_data)}")
print(f"FILTERED VAL DATASET SIZE: {len(val_data)}")
print(f"FILTERED TEST DATASET SIZE: {len(test_data)}")

def extractCLSEmbeddings(input):
    with torch.no_grad():
        embedding = model(**input).last_hidden_state[:,0,:]
    return embedding

def batchEmbeddingExtraction(input, batch_size):
    all_embedding = []
    num_sequences = input["input_ids"].size(0)
    for i in range(0, num_sequences, batch_size):
        batch_input = {key: value[i:i + batch_size].cuda() for key, value in input.items()}
        embedding = extractCLSEmbeddings(batch_input)
        embedding = embedding.cpu()
        all_embedding.append(embedding)

    all_embedding = torch.cat(all_embedding, dim=0)

    return all_embedding

train_embedding = batchEmbeddingExtraction(train_tokenized, 16)
val_embedding = batchEmbeddingExtraction(val_tokenized, 16)
test_embedding = batchEmbeddingExtraction(test_tokenized, 16)

def relateEmbeddingToMotionID(data_embedding, data):
    related_dict = {}
    for idx, row in data.iterrows():
        print(idx)
        related_dict[row["MotionID"]] = data_embedding[idx]
    return related_dict

train_embedding_dict = relateEmbeddingToMotionID(train_embedding, train_data)
val_embedding_dict = relateEmbeddingToMotionID(val_embedding, val_data)
test_embedding_dict = relateEmbeddingToMotionID(test_embedding, test_data)

save_path = "/home/coder/second/PaperData/nlp_embeddings/bert_tfidf"
#save_path = "/home/vyral/Documents/UCONN/fullData/PaperData3/PaperData/nlp_embeddings/bert_truncation"
os.makedirs(save_path, exist_ok=True)
torch.save(train_embedding_dict, os.path.join(save_path, "train_embedding.pth"))
torch.save(val_embedding_dict, os.path.join(save_path, "val_embedding.pth"))
torch.save(test_embedding_dict, os.path.join(save_path, "test_embedding.pth"))
