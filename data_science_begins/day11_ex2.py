import ssl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv") 

# df_numeric = df.select_dtypes(include=['float64', 'int64']) or
del df['species']

# Calculate correlation matrix
correlation_matrix = df.corr()

# plot heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()