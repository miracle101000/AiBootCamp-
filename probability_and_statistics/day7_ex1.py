import pandas as pd
import seaborn as sns
import ssl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#Perforiming Exploratory Data Analysis

ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv") 

# 1. Inspect Data
# print(df.info())
# print(df.describe())

#2. Visualize Distributions
# sns.histplot(df["total_bill"], kde=True)
# plt.title("Distirbution of Total Bill")
# plt.show()



#3. Correlation heatmap
# del df['sex']
# del df['smoker']
# del df['day']
# del df['time']
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()


#4. Conducting Hypothesis Testing (checking if something is by coincidence)
# from scipy.stats import ttest_ind, ttest_1samp

# #Seperate data by gender
# male_tips = df[df['sex'] == 'Male']['tip']
# female_tips =  df[df['sex'] == 'Female']['tip']


# #Perform t-test
# t_stat, p_value =  ttest_ind(male_tips,female_tips)
# print("T-Statistics: ", t_stat)
# print("P-Value: ",p_value)

# #Interpret results
# alpha = 0.05
# if p_value <= alpha:
#     print("Reject all null hypothesis: Significant diff")
# else:
#     print("Fail to Reject all null hypothesis: NO Significant diff")    


# Define variables
X = df['total_bill'].values.reshape(-1, 1)
y= df['tip'].values

#Fit linear regression
model =  LinearRegression()
model.fit(X, y)

# Output coefficients
print("Slope: ",model.coef_[0])
print("Intercept: ", model.intercept_)
print("R-Squared:", model.score(X, y))

# Plot regression
sns.scatterplot(x = df['total_bill'], y = df['tip'], color="blue")
plt.plot(df["total_bill"], model.predict(X), color = "red", label = "Regression Line")
plt.title("Total Bill vs Tip")
plt.show()


