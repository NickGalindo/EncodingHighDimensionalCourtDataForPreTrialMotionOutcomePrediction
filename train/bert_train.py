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


class LegalDocDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_train.csv")
train_data = train_data[["filepath", "document_no", "MotionResultCode"]]
train_data["label"] = train_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)

val_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_val.csv")
val_data = val_data[["filepath", "document_no", "MotionResultCode"]]
val_data["label"] = val_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)

test_data = pd.read_csv("/mnt/research/aguiarlab/proj/law/data/PaperData/mapped_full_test.csv")
test_data = test_data[["filepath", "document_no", "MotionResultCode"]]
test_data["label"] = test_data["MotionResultCode"].apply(lambda x: 1 if x == "GR" else 0)

full_corpus = pickle.load(open(os.path.join("/mnt/research/aguiarlab/proj/law/data/PaperData/textData/tfidf", "indexed_text.pkl"), "rb"))

train_data["text"] = train_data["document_no"].map(full_corpus)
val_data["text"] = val_data["document_no"].map(full_corpus)
test_data["text"] = test_data["document_no"].map(full_corpus)

train_data = train_data[["text", "label"]].dropna()
val_data = val_data[["text", "label"]].dropna()
test_data = test_data[["text", "label"]].dropna()

print(f"TRAIN DATASET SIZE: {len(train_data)}")
print(f"VAL DATASET SIZE: {len(val_data)}")
print(f"TEST DATASET SIZE: {len(test_data)}")

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", num_labels=2)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

def tokenizeFunction(some_dataset):
    return tokenizer(some_dataset["text"], padding="max_length", truncation=True, max_length=512)

train_tokenized = train_dataset.map(tokenizeFunction, batched=True)
val_tokenized = val_dataset.map(tokenizeFunction, batched=True)
test_tokenized = test_dataset.map(tokenizeFunction, batched=True)

accuracy = evaluate.load("accuracy")
def computeMetrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predicitions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='/mnt/research/aguiarlab/proj/law/data/PaperData/bertTraining/results',          # output directory for the model and logs
    eval_strategy="epoch",     # evaluate after each epoch
    save_strategy="epoch",
    learning_rate=5e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/mnt/research/aguiarlab/proj/law/data/PaperData/bertTraining/logs',            # directory for storing logs
    logging_steps=10,                # log every 10 steps
    save_steps=10,                   # save checkpoint every 10 steps
    load_best_model_at_end=True,     # load the best model when finished training (based on validation)
    fp16=True
)

trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments, defined above
    train_dataset=train_tokenized,    # training dataset
    eval_dataset=val_tokenized,     # evaluation dataset
    processing_class=tokenizer,                 # tokenizer used for data preprocessing
    compute_metrics=computeMetrics
)

trainer.train()

results = trainer.evaluate()
print(f"EVALUATIONRESULTS: {results}")

pred, labels = trainer.predict(test_tokenized)

pred_classes = np.argmax(pred, axis=1)

accuracy = accuracy_score(labels, predicted_classes)

print(f"ACCURACY ON TEST: {accuracy}")

model.save_pretrained("/mnt/research/aguiarlab/proj/law/data/PaperData/bertTraining/models/tfigm")
tokenizer.save_pretrained("/mnt/research/aguiarlab/proj/law/data/PaperData/bertTraining/models/tfidf")
