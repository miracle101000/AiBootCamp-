import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson


#Guassian Distribution
# x = np.linspace(-4, 4, 100)
# y = norm.pdf(x, loc=0, scale=1)
# plt.plot(x, y,label="Guassian")
# plt.title("Guassian Distribution")
# plt.show()

#Binomial distribution
# n, p = 10, 0.5
# x = np.arange(0, n+1)
# y = binom.pmf(x, n, p)
# plt.bar(x, y, color="green",label="Binomial")
# plt.title("Binomial Distribution")
# plt.show()

# Poission Distribution
# lam =3
# x = np.arange(0, 10)
# y = poisson.pmf(x, lam)
# plt.bar(x, y, color="orange",label="Poisson")
# plt.title("Poisson Distribution")
# plt.show()


# Parameters for the multinomial distribution
# n = 100  # Number of trials
# p = [0.2, 0.5, 0.3]  # Probabilities for each outcome

# # Generate random samples from the multinomial distribution
# samples = np.random.multinomial(n, p, size=1000)

# # Compute the average frequencies of outcomes from the samples
# average_frequencies = np.mean(samples, axis=0)

# # Visualize the distribution
# categories = [f"Category {i+1}" for i in range(len(p))]

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.bar(categories, average_frequencies, color=['blue', 'orange', 'green'], alpha=0.7)
# ax.set_title("Visualization of Multinomial Distribution")
# ax.set_ylabel("Average Frequency")
# ax.set_xlabel("Categories")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


# Generate continuous data for Gaussian and Uniform distributions
# np.random.seed(42)
# n_samples = 10000

# # Gaussian (Normal) distribution parameters
# mean = 0
# std_dev = 1
# gaussian_data = np.random.normal(mean, std_dev, n_samples)

# # Uniform distribution parameters
# low = -2
# high = 2
# uniform_data = np.random.uniform(low, high, n_samples)

# # Plotting the distributions
# fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# # Gaussian Distribution
# ax[0].hist(gaussian_data, bins=50, density=True, color='skyblue', alpha=0.7, edgecolor='black')
# ax[0].set_title("Gaussian Distribution")
# ax[0].set_xlabel("Value")
# ax[0].set_ylabel("Density")
# ax[0].grid(axis='y', linestyle='--', alpha=0.7)

# # Uniform Distribution
# ax[1].hist(uniform_data, bins=50, density=True, color='salmon', alpha=0.7, edgecolor='black')
# ax[1].set_title("Uniform Distribution")
# ax[1].set_xlabel("Value")
# ax[1].grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()

