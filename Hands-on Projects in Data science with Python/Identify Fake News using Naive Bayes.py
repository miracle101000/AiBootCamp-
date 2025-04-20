import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'fake_news.csv' with actual dataset path)
df = pd.read_csv('fake_news.csv')

# Display the first few rows of the dataset
print("Dataset Sample:\n", df.head())

# Define features (X) and target (y)
X = df['title'] # Assuming the dataset has a 'text' column for the news content
y = df['real']  # Assuming the dataset has a 'label' column with values 'fake' or 'real'

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

# Initialize the TfidfVectorizer to convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_df=.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Intialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = model.predict(X_test_tfidf)

print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", report)