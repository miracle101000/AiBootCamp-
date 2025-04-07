import numpy as np

#Identity Matrix
I = np.eye(3)
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("I * A: \n", np.dot(I,A))

#Diagonal and Zero Matrix
D = np.diag([1,2,3])
Z = np.zeros((3,3))

print("diag([1,2,3]) * zeros(3,3): \n", np.dot(D,Z))