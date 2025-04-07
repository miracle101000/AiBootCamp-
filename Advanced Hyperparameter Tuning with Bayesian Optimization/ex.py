
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna

# Load the dataset
data =  load_breast_cancer()
X, y = data.data, data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Train a baseline XGBoost model
baseline_model = XGBClassifier(eval_metric='logloss',random_state=42)
baseline_model.fit(X_train, y_train)

# Evaluate the model
baseline_preds =  baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_preds)
print(f"Baseline XGBoost Accuracy: {baseline_accuracy}")

# Define the objective function for Optuna
def objective(trial):
    params = {
        'n_estimators':trial.suggest_int('n_estimators',50, 500),
        'max_depth': trial.suggest_int('max_depth',3, 100),
        'learning_rate':trial.suggest_float('learning_rate',0.01, 0.3),
        'subsample': trial.suggest_float('subsample',0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    
#     # Train XGBoost model with suggested params
    model  =  XGBClassifier(eval_metrics ='logloss', random_state=42, **params)
    model.fit(X_train, y_train)
    
#     # Evaluate model on validate set
    preds =  model.predict(X_test)
    accuracy =  accuracy_score(y_test, preds)
    return accuracy

# Create an Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best hyperprameters
print("Best Hyperparameters:", study.best_params)
print("Best Accuracy:", study.best_value)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}

# Train XGBoost with Grid Search
grid_search =  GridSearchCV(
           estimator=XGBClassifier(eval_metric='logloss', random_state=42),
           param_grid=param_grid,
           scoring='accuracy',
           cv=3,
           verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("\n\nGrid Search Best Parameters:\n\n", grid_search.best_params_)
print("\n\nGrid Search Best Accuracy:\n\n", grid_search.best_score_)

param_dist= {
    'n_estimators': [50,100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6,0.7, 0.8, 0.9, 1.0]
}

# Train XGBoost with Grid Search
random_search =  RandomizedSearchCV(
           estimator=XGBClassifier(eval_metric='logloss', random_state=42),
           param_grid=param_dist,
           scoring='accuracy',
           n_iter=50,
           cv=3,
           verbose=1,
           random_state=42,
           n_jobs=-1
)

random_search.fit(X_train, y_train)

# Best parameters and accuracy
print("\n\nRandom Search Best Parameters:\n\n", random_search.best_params_)
print("\n\nRandom Search Best Accuracy:\n\n", random_search.best_score_)




