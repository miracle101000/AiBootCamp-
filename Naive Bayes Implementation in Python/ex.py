from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Sample data (e.g, hours and grades vs. pass/fail)
X = np.array()
y = np.array()

#Split the data into training and testing data
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.2,random_state=42)

# Initilaize and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make Predicitons
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy", accuracy)
print("Confusion Matrix:\n", conf_matrix)

