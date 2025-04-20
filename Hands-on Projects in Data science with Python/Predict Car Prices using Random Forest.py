import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (replace with a real dataset if available)
data = {
    'make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Toyota', 'Honda', 'Ford', 'BMW', 'Audi'],
    'model': ['Corolla', 'Civic', 'F-150', '3-Series', 'A4', 'Camry', 'Accord', 'Mustang', '5 Series', 'A6'],
    'year': [2015, 2017, 2018, 2016, 2015, 2018, 2016, 2015, 2017, 2018],
    'price': [12000, 15000, 25000, 22000, 24000, 20000, 16000, 26000, 27000, 30000]  # example values
}

# Convert to DataFrame
df = pd.DataFrame(data) 
print("Dataset:\n", df)

# Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['make', 'model'])
print("\nEncoded Dataset:\n", df_encoded)

# Define features (X) and target (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

