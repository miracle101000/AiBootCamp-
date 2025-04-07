#Problem
# - A disease affects 1% of a population
# - A test is 95% accurate for the diseased individuals and 90% accurate for the non-diseased individuals
# - What is the probability that a person who tests positive has the disease?


def bayes_theorem(prior, sensivity, specificity):
    evidence = ((prior * sensivity) + ((1-prior) * (1-specificity)))
    posterior = (prior * sensivity) / evidence
    return posterior

prior = 0.01
sensitivity = 0.95 #True Positive Rate
specificity = 0.90 #True Negative Rate

posterior = bayes_theorem(prior, sensitivity, specificity)
print("Probability that a person who tests positive has the disease: ", posterior) #0.095

