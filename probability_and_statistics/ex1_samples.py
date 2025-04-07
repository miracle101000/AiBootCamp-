import numpy as np

# Random variable dice roll
outcomes = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6]*6)

#Expectations
expectations = np.sum(outcomes * probabilities)
print("Expectaation (Mean): ",expectations)

# Variance and Standard Deviation
variance = np.sum((outcomes - expectations)**2 * probabilities)
std_dev = np.sqrt(variance)
print("Variaance", variance)
print("Standard deviation",std_dev)

# from itertools  import product

# # Sample space of a dice roll
# sample_space = list(range(1,7))

# #Probability of rolling an even number
# even_numbers = [2, 4, 6]
# P_even = len(even_numbers) / len(sample_space)
# print("P{Even}:", P_even)