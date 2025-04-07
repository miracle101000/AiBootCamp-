import ssl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


#Load the Carlifonia Housing dataset
data =  fetch_california_housing(as_frame=True)
df = data.frame


# Select feature (Median Income) and target (Median House Value)
X = df[['MedInc']]
y = df[['MedHouseVal']]


# Transform features to polynomial
poly  =  PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)


#Split data into training  and testing
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

#Ridge Regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
lasso_predictions = lasso_model.predict(X_test)

#Evaluate Ridge Regression
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge Regression MSE: ", ridge_mse)

#Evaluate Lasso Regression
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("Lasso Regression MSE: ", lasso_mse)

#Visualize Ridge vs Lasso Predictions
plt.figure(figsize=(10,6))
plt.scatter(X_test[:, 0], y_test, color="blue", label="Actual Data", alpha=0.5)
plt.scatter(X_test[:, 0], ridge_predictions, color="red", label="Ridge Prediction", alpha=0.5)
plt.scatter(X_test[:, 0], lasso_predictions, color="orange", label="Lasso Predicitions", alpha=0.5)
plt.title("Ridge vs Lasso Regression")
plt.xlabel("Median Income")
plt.xlabel("Median House Value in California")
plt.legend()
plt.show()

# Fit polynomial regression model
# model =  LinearRegression()
# model.fit(X_train, y_train)

# #Make Predictions
# y_pred =  model.predict(X_test)

# linear_regression_mse = mean_squared_error(y_test,y_pred)
# print("Linear Regression MSE: ", linear_regression_mse)



# #Plot  actual vs predicted values
# plt.figure(figsize=(10,6))
# plt.scatter(X, y, color="blue", label="Actual Data", alpha=0.5)
# plt.scatter(X, y_pred, color="red", label="Predicted Curve", alpha=0.5)
# plt.title("Polynomial Regression")
# plt.xlabel("Median Income in California")
# plt.ylabel("Median House Value in Carlifonia")
# plt.legend()
# plt.show()


# #Evaluate model performance
# mse = mean_squared_error(y, y_pred=y_pred)
# print("Mean Squared Error (MSE): ", mse)