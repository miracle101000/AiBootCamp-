import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

#Discrete random variable: Dice roll
# outcomes = [1, 2, 3, 4, 5, 6]
# probailities = [1/6] * 6
# plt.bar(outcomes, probailities, color="blue", alpha=0.7)
# plt.title("PMF of a Dice Roll")
# plt.xlabel("Outcomes")
# plt.ylabel("probability")
# plt.show()

#Discrete random variable
outcomes = [1, 2]
probabilites = [1,2] * 2
expectaions =  np.sum(outcomes * probabilites)
variance = np.sum((outcomes - expectaions)**2 * probabilites)
std_dev =  np.sqrt(variance)

#Continuos random variable: Uniform distribution
# x = np.linspace(0,1, 100)
# pdf = uniform.pdf(x, loc=0, scale=1)
# plt.plot(x, pdf, color="red")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.show()


