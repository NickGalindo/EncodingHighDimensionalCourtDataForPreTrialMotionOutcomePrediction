from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import numpy as np


oc: OrdinalEncoder = OrdinalEncoder(encoded_missing_value=0)

train_df: pd.DataFrame = pd.read_csv("/mnt/research/aguiarlab/proj/law/code/law/tabtransformers-lightning/tmp_data/train.csv")

train_df[train_df.columns] = oc.fit_transform(train_df)

X = train_df[list(set(train_df.columns)-{"MotionResultCode"})]
y = train_df["MotionResultCode"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbdt = GradientBoostingClassifier(
    n_estimators=100,   # Number of boosting stages
    learning_rate=0.1,  # Step size shrinkage
    max_depth=3,        # Max depth of individual trees
    random_state=42     # Ensures reproducibility
)

gbdt.fit(X_train, y_train)

y_pred = gbdt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
