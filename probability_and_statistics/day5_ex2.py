#Performm a Chi-Square Test
from scipy.stats import chi2_contingency

#Contingency Table
data = [[50, 30, 20], [30, 40, 30]]


#Perform Chi_Square Test
chi2, p, dof, excepted = chi2_contingency(data)
print("Chi-Square Statistics:", chi2)
print("P-Value", p)
print("Expected Frequency", excepted)