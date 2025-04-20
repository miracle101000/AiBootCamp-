# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'reviews.csv with the actual dataset path)
df = pd.read_csv('reviews.csv')

# Display the first few rows of the dataset
print("Dataset Sample:\n", df.head())

# Define features (X) and target (y)
X = df['review_text'] # Assuming the dataset has a 'review_text' column
y = df['label'] # Assuming the dataset has 'label' column with values 'fake' or 'genuine'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

# Initialize the TfidfVectorize to convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_df=.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Logistic Regression classifier
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", report)