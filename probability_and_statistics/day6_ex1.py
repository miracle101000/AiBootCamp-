import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

#Load Iris dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv") 

del df['species']
#Compute correlation matrix
correlation_matrix =  df.corr()


# Plot Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()


