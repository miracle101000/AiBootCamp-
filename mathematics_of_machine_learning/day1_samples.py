import numpy as np

#Create matrices
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

# #Additon
# print("Addition: \n", A+B)

# #Subtraction
# print("Subtraction: \n", A-B)

# #Scalar multiplication
# print("Scalar Multiplication: \n", 2*A)

# #Scalar Multiplication
# print("Scalar Division: \n", A/2)

# #Matrix Multiplication: Number of Columns A = Number of Rows B
result = np.dot(A,B)
# print("Matrix Multiplication: \n", result)

#Special Matrices
I = np.eye(5)
# print("Identity Matrix: \n", I)

Z = np.zeros((3,3))
# print("Zero Matrix: \n", Z)

D =  np.diag([1,2,3])
print("Diagonal Matrix: \n", D)