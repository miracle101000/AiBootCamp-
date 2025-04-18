import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {}

df = pd.DataFrame(data)

X =  df[["Temperature", "Humidity", "Wind Speed", "Precipitation"]]
y = df["Next Day Temperature"]

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse =  mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Temperature", marker="o")
plt.plot(y_pred, label="Predicted Temperature", marker="x")
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Test Sample Index")
plt.ylabel("Temperature")
plt.legend()
plt.show()

new_data = pd.DataFrame({
    
})

predicted_temperature = model.predict(new_data)
print(f"\n\nPredicted temperature: {predicted_temperature[0]:2f}C")