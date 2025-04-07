import numpy as np

A = np.array([[3,1,1],[-1,3,1],[1,1,3]])
U,S,Vt = np.linalg.svd(A)

print("U:\n", U)
print("Singular Values: \n", U)
print("V Transpose: \n", Vt)

#Reconstruct
Sigma = np.zeros((3,3))
np.fill_diagonal(Sigma, S)
reconstruct = U @ Sigma @ Vt
print("Reconstucted Matrix\n", reconstruct)