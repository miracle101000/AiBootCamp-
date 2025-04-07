from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data (e.g.house size vs. house price)
X = np.array()
y = np.array()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Ridge Regression
ridge_model =  Ridge(alpha=1.0) # alpha controls the rgularization strength
ridge_model.fit(X_train, y_train)
ridge_pred =  ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
print("Ridge Mean Square Error:", ridge_mse)

# Lasso Regression
lasso_model =  Lasso(alpha=.1) # alpha controls the reqgulariztion strength
lasso_model.fit(X_train, y_train)
lasso_pred =  lasso_model.predict(X_test)
lasso_mse =  mean_squared_error(y_test, lasso_pred)
print("Lasso Mean Squared Error", lasso_mse)


