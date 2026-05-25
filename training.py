import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# LOAD CLEANED DATASET

df = pd.read_csv("cleaned_online_shoppers.csv")

# SELECT FEATURES

X = df[
    [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay"
    ]
]

# TARGET

y = df["Revenue"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# FEATURE SCALING

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)

x_test = scaler.transform(X_test)

# MODEL TRAINING

model = LogisticRegression()

model.fit(x_train, y_train)

# PREDICTION

y_pred = model.predict(x_test)

# EVALUATION

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred)

print("Model Evaluation:\n")

print(f"Accuracy Score: {accuracy}")

print(f"Precision Score: {precision}")

print(f"Recall Score: {recall}")

print(f"F1 Score: {f1}")

print("\nConfusion Matrix:\n")

print(cm)

print("\nClassification Report:\n")

print(report)

# SAVE MODEL

pickle.dump(model, open("model.pkl", "wb"))

# SAVE SCALER

pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel and Scaler Saved Successfully")