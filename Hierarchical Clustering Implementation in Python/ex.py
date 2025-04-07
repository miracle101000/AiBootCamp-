from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Sample data (e.g., points in 2D space)
X = np.array()

# Perform heirachical/agglomeratuve clustering
Z = linkage(X, method='ward') 

# Plot dendogram
plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.title("Dendogram for Heirachical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()