import numpy as np

arr = np.array([1,2,3,4,5])
# print(arr)
# [1 2 3 4 5]

zeroes = np.zeros((3,3))
# print(zeroes)
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]

ones= np.ones((2,4))
# print(ones)
# [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]]

range_array= np.arange(1,10,2)
# print(range_array)
# [1 3 5 7 9]

linspace_array = np.linspace(0,1,5)
# print(linspace_array)
# [0.   0.25 0.5  0.75 1.  ]

arr = np.array([1,2,3,4,5,6,7,8,9])
reshaped = arr.reshape((3,3))
# print(reshaped)

arr = np.array([1,2,3])
expanded = arr[:,np.newaxis]
# print(expanded)

a = np.array([1,2,3])
b = np.array([4,5,6])

# print(a+b) #[5 7 9]
# print(a*b) #[ 4 10 18]
# print(a/b) #[0.25 0.4  0.5 

# arr = np.array([4,16,25])
# print(np.sqrt(arr))
# print(np.sum(arr))
# print(np.mean(arr))
# print(np.max(arr))


arr = np.array([10,20,30,40,50,60])
# print(arr[2]) #30
# print(arr[-1]) #50

# print(arr[1:4]) #[20 30 40]
# print(arr[:3])  #[10,20,30]
# print(arr[3:]) #[40,50]

# reshaped = arr.reshape((2,3))
# print(reshaped)