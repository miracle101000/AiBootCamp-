import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score,accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df  = pd.read_csv(url)

# Explore dataset
print("Dataset Info:\n")
print(df.info())
print("\n Class Distiribution:\n")
print(df['Class'].value_counts())

# Split dataset
X = df.drop(columns=['Class'])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)


# Predict and evaluate
y_pred =  rf_model.predict(X_test)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc_auc: .2f}")

# Apply SMOTE
smote =  SMOTE(random_state=42)
X_resampled, y_resampled =  smote.fit_resample(X_train, y_train)

#Display new class distribution
print("\n Class Distribution After SMOTE: \n")
print(pd.Series(y_resampled).value_counts())

# Train Random Forest on resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred_smote = rf_model_smote.predict(X_test)
print("\n Classification Report (SMOTE):\n")
print(classification_report(y_test, y_pred))

# Train XGBoost model
xgb_model =  XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict and evaluate
xgb_pred =  xgb_model.predict(X_test)
print(f"XGB Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")