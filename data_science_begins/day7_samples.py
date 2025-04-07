import sys

print(sys.argv)
print(sys.version)



# import os

# print(os.getcwd())
# os.mkdir("test_dir")
# os.remove("file.txt")

# from functools import reduce
# numbers = [1, 2, 3, 4]
# product = reduce(lambda x,y: x * y, numbers)
# # print(product)


# numbers = [1, 2, 3, 4]
# evenList =  filter(lambda x: x % 2 == 0, numbers)
# # print(list(evenList))

# numbers = [1,2,3,4]
# squares = map(lambda x: x**2, numbers)
# # print(list(squares))

# # [expression for item in iterable if condition]

# #Create a list of squares
# squares = [x**2 for x in range(10)]
# # print(squares)

# #Filter Even Numbers
# evens = [x for x in range(10) if  x % 2 == 0]
# # print(evens)

# # lambda arguments: expression
# add = lambda x, y: x + y
# print(add(3,5))
