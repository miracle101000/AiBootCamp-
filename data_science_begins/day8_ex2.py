import numpy as np


matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print("Original Matrix: \n", matrix)

transpose = matrix.T
# print("Transpose:\n",transpose)

another_matrix = np.array([[9,8,7],[6,5,4],[3,2,1]])
# print("Addition: \n",matrix + another_matrix)
# print("Mutliplication: \n",matrix * another_matrix)


matrix_4by4 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
sum_4by4 = np.sum(matrix_4by4)

# print(f"Sum of:\n {matrix_4by4}\n is\n {sum_4by4}")


arr = np.array([5,10,15,20,25])

min_val = np.min(arr)
max_val = np.max(arr)
normalized_array = (arr - min_val) / (max_val - min_val)
# print("Normalized Array:", normalized_array)


random_arr = np.random.randint(10,size=20)
# print(np.max(random_arr))