print("RUNNING TRAIN MODEL FILE...")
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


print("✅ Training started...")

# Load dataset
df = pd.read_csv("PS_20174392719_1491204439457_log.csv",nrows=100000)
print("✅ Dataset loaded")

# Drop unnecessary columns
df.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

# Encode categorical column
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

# Features and target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained successfully")

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)

# Save model in main folder
model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("🔥 model.pkl created successfully at:", model_path)
