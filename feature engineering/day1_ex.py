# This is to prepare data for machine learning

import pandas as pd
import ssl

ssl._create_default_https_context =  ssl._create_unverified_context

# Load Titanic
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv") 


# Display dataset Information
print("Dataset Info: \n")
print(df.info())

# Preview the first few rows
print("\n Dataset Preview:\n")
print(df.head())

# Separate features
categorical_features =  df.select_dtypes(include=['object']).columns
numberical_features =  df.select_dtypes(include=['int64','float64']).columns

print("\nCategorical Features:\n", categorical_features.tolist())
print("\nNumerical Features:\n", numberical_features.tolist())

# Display summary of categorical features
print("\n Categorical Feature Summary:\n")
for col in categorical_features:
    print(f"{col}:\n", df[col].value_counts(), "\n")
    
print("\n Numerical Feature Summary:\n")    
print(df[numberical_features].describe())

