import pandas as pd
from scipy.stats import norm
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#Load Iris dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv") 

#Sampling
sample = df["sepal_length"].sample(30, random_state=42)

# Sample statistics
mean =  sample.mean()
std = sample.std()
n = len(sample)


# Confidence Interval
z_value =  norm.pdf(0.975)
margin_of_error = z_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print("Sample Mean ", mean)
print("95% Confidence Interval ", ci)