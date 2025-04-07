from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.svm import SVC


# Load dataset
data = load_iris()
X, y = data.data, data.target


#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

print("Dataset Loaded and Split Successfully")

# DEfine parmater grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Intialize GridSearchCV
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Get best parameters and score
best_params_grid = grid_search.best_params_
best_score_grid =  grid_search.best_score_

print(f"Best Paramters (GridSearchCV): {best_params_grid}")
print(f"Best Cross-Validation Accuracy (GridSearchCV): {best_score_grid}")

# Get best model
best_grid_model = grid_search.best_estimator_


# Get best model
y_pred_grid =  best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)

print(f"Test Accuracy (GridSearchCV): {accuracy_grid:.4f}")
print(f"\n Classification Report:\n {classification_report(y_test,y_pred_grid)}")


# Define parameter distribution
param_dist = {
    'C': np.logspace(-3, 3, 10),
    "kernel": ['linear','rbf','poly','sigmoid'],
    'gamma': ['scale','auto']
}

#INITIALIZE THE RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42
)

# Perform Randomized Search
random_search.fit(X_train, y_train)

# Get best paraameters and score
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print(f"Best Parameters (RandomizedSearchCV): {best_params_random}")
print(f"Best Cross-Validation (RandomizedSearchCV): {best_score_random:.4f}")

# Get best model
best_random_model =  random_search.best_estimator_

# Predict and evaluate
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Test Accuracy (RandomizedSearchCV): {accuracy_random:.4f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred_random))
