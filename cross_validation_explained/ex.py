import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df  = pd.read_csv(url)

# Display dataset info
print("Dataset Info:\n")
print(df.info())
print("\n Class Distribution:\n")
print(df['Class'].value_counts())

# Define Features and target
X =  df.drop(columns=['Class'])
y = df['Class']

#Split dataset
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.2, random_state=42)

# Intialize k-Fold
kf =  KFold(n_splits=5,shuffle=True, random_state=42)

# TRain and evaluate model
rf_model = RandomForestClassifier(random_state=42)
scores_kf = cross_val_score(rf_model,X_train, y_train, cv=kf, scoring='accuracy')

print(f"K-Fold cross validation scores: {scores_kf}")
print(f"Mean Accuracy (K-Fold): {scores_kf.mean():.2f}")

#Initialized Stratified K-Fold
skf =  StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate modek
scores_stratified = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')

print(f"Stratified K-fold cross validation scores: {scores_stratified}")
print(f"Stratified Mean Accuracy (K-Fold): {scores_stratified.mean():.2f}")