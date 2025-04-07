import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data =np.random.rand(5,5)
sns.heatmap(data, annot=True,cmap='coolwarm')
# plt.title("Heatmap")
# plt.show()
# sns.pairplot(df)
# plt.show()



#Basic plot
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 35]
# plt.plot(x, y)
# plt.show()

#Line Plot
# plt.plot([1, 2, 3], [10, 20, 30], label='Trend')
# plt.title("Line Plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.legend()
# plt.show()

# Bar Chart
# categories = ['A', 'B', 'C']
# values = [11, 22, 33]   
# plt.bar(categories, values,color="blue")
# plt.title("Bar Chart")
# plt.show()

# Histogram
# data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# plt.hist(data, bins=4, color="green", edgecolor="black")
# plt.title("Histogram")
# plt.show()

#Scatter Plot
x = [1, 2, 3, 4, 5]
y = [10, 12, 25, 35, 40]

plt.scatter(x, y, color="red")
# plt.title("Scatter Plot")
# plt.xlabel("X-axis Label")
# plt.ylabel("Y-axis Label")
# plt.legend(["Dataset 1"])
# plt.show()