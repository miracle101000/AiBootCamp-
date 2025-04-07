import ssl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ssl._create_default_https_context = ssl._create_unverified_context

#Load Ttitanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv") 

#Inspect Data
# print(df.info())
# print(df.describe())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

#Remove duplicates
df = df.drop_duplicates()

#Filter data: Passengers in first class
first_class_passengers = df[df["Pclass"]==1]
# print(first_class_passengers)

#Bar Chart: Survival rate by class
# survival_rate_by_class = df.groupby("Pclass")["Survived"].mean()
# survival_rate_by_class.plot(kind="bar", color="skyblue")
# plt.title("Survival Rate by Class")
# plt.ylabel("Survival Rate")
# plt.show()

# Histogram: Age distribution
# sns.histplot(df["Age"],kde=True,bins=20, color="purple")
# plt.title("Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.show()

# Scatter Plot: Age vs Fare
plt.scatter(df["Age"], df["Fare"], color="red",alpha=0.5)
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

