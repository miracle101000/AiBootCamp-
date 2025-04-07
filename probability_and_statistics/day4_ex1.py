import numpy as np
from scipy.stats import ttest_1samp

#Sample data
data = [12, 14, 15, 16, 17, 18, 19]

#Null Hyposthesis: mean = 15
population_mean = 15

# Perform t-test
t_stat, p_value =  ttest_1samp(data, population_mean)
print("T-statistics: ",t_stat)
print("P-value: ",p_value)

#Interpret Results
alpha = 0.05
if p_value <= alpha:
   print("Reject the null hypothesis: significant difference")
else:          
   print("Fail to Reject the null hyposthese: no significnt difference")