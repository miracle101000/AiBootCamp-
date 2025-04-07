# import numpy as np
# from scipy.stats import pearsonr, spearmanr

# x =  np.array([1, 2, 3, 4, 5])
# y =  np.array([2, 4, 6, 8, 10])


# r, _ =  pearsonr(x, y)
# # Pearson Correlation
# print("Pearson Correlation Coefficient: ", r)

# #Spearman Correlation
# rho, _ =  spearmanr(x, y)
# print("Spearman Correlation Coefficient", rho)

from sklearn.linear_model import LinearRegression
import numpy as np

#Sample Data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) #making it a 2-dimensional array
y = np.array([2, 4, 5, 8, 10])

model =  LinearRegression()
model.fit(x, y)

print("Slope: ", model.coef_[0])
print("Intercept: ", model.intercept_)
print("R-Squared: ", model.score(x,y))


