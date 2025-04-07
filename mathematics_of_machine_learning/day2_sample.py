import numpy as np

A = np.array([[2,3], [1,4]])

U, S, Vt = np.linalg.svd(A)
print("U: \n", U)
print("Singular Values: \n", S)
print("V Transpose: \n", Vt)


determinant = np.linalg.det(A)
# print("Determinant: ",determinant)

inverse = np.linalg.inv(A)
res= np.dot(A,inverse)
# print(res)
# print("Inverse of A\n",A*inverse)\

eigenValues, eigeneVectors = np.linalg.eig(A)
# print(eigeneVectors)
# print(eigenValues)

B = np.array([[4, 2], [1, 1]])
eigval, eigvec = np.linalg.eig(B)
# print("Eigval", eigval)
# print("Eigvect: ", eigvec)