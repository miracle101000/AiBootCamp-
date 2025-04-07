from sklearn.mixture import GaussianMixture
import numpy as np

# Sample data (e.g., points in 2D space)
X = np.array()

# Intialize and fit the model
gnm = GaussianMixture(n_components=2, random_state=42)
gnm.fit(X)

# Get the cluster labels and probabilities
labels = gnm.predict(X)
probs = gnm.predict_proba(X)

print("Cluster Labels:", labels)
print("Cluster Probabilities:\n", probs)