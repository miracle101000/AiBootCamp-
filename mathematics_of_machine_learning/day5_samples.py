def bayes_theorem(prior, likelihood, evidence):
    return (prior * likelihood) / evidence

import numpy as np
import matplotlib.pyplot as plt

#Guassian Distribution
# mu,sigma = 0, 1
# x = np.linspace(-4, 4, 100)
# y = (1/(np.sqrt(2 * np.pi * sigma**2 ))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# plt.plot(x, y)
# plt.title("Guassian Distribution")
# plt.show()

#Bernoulli Distribution
# p = 0.5
# plt.bar([0, 1], [1-p, p],color="blue")
# plt.title("Bernoulli Distribution")
# plt.xticks([0, 1], labels=["0 (Failure)", "1 (Success)"])
# plt.show()


#Binomial distribution
# from scipy.stats import binom

# n, p = 10, 0.5
# x = np.arange(0, n+1)
# y= binom.pmf(x, n, p)
# plt.bar(x, y, color="green")
# plt.title("Binomial Distribution")
# plt.show()

#Poission Distribution
# from scipy.stats import poisson
# lam = 3
# x = np.arange(0, 10)
# y = poisson.pmf(x, lam)
# plt.bar(x, y, color="orange")
# plt.title("Poisson Distribution")
# plt.show()



