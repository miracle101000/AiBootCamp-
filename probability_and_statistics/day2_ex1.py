#url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

from scipy.stats import skew, kurtosis
import pandas as pd
import seaborn as sns
import ssl
import matplotlib.pyplot as plt


ssl._create_default_https_context = ssl._create_unverified_context
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df =  pd.read_csv(url)

#Analyse sepal_length
feature = df["sepal_length"]
print("Skewness: ", skew(feature))
print("Kurtosis: ",kurtosis(feature))


#Visualize distribution
sns.histplot(feature, kde=True)
plt.title("Distirbution od Sepal Length")
plt.title("Distirbution od Sepal Length")
plt.show()
