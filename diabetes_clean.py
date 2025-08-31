import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import kagglehub
import joblib

# Load dataset
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
df = pd.read_csv(path + "/diabetes.csv")
print("Dataset loaded:", df.shape)

# Preprocess
X = df.drop(columns="Outcome", axis=1)
Y = df["Outcome"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Evaluate
print("Train Accuracy:", accuracy_score(classifier.predict(X_train), Y_train))
print("Test Accuracy:", accuracy_score(classifier.predict(X_test), Y_test))

# Save model + scaler
joblib.dump(classifier, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Example prediction
sample = np.array((4,110,92,0,0,37.6,0.191,30)).reshape(1,-1)
sample_scaled = scaler.transform(sample)
print("Prediction:", classifier.predict(sample_scaled))
