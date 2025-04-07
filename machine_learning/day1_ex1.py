import pandas as pd
import ssl
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Perforiming Exploratory Data Analysis

ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv") 


#Define features and target
features =df[['total_bill','size']] #x
target = df['tip'] #y

print("Features: \n", features.head())
print("Target:  \n", target.tail())

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("Training Data Set: \n", X_train.shape)
print("Test Data Set: \n", X_test.shape)

#Visualize relations
sns.pairplot(df, x_vars=["total_bill", "size"],y_vars="tip",height=5,aspect=0.8,kind="scatter")
plt.title("Feature vs Target Relationship")
plt.show()

