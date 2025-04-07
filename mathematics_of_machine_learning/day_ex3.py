import numpy as np
import sympy as sp



#Define the gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = np.dot(X,theta)
        errors = predictions - y
        gradients = (1/m) * np.dot(X.T, errors)
        theta -= learning_rate * gradients
    return theta    

#Sample Data
X = np.array([[1,1],[1,2],[1,3]])
y = np.array([2, 2.5, 3.5])
theta = np.array([0.1,0.1])
learning_rate = 0.1
iteration = 1000

#Perform gradient descent
optimized_theta =  gradient_descent(X=X,y=y,theta=theta,learning_rate=learning_rate,iterations=iteration)

# print("Optimised Theta\n", optimized_theta)



# # Define variables
# x, y = sp.symbols('x y')

# # Define the function
# f = x**2 + x*y + y**2

# # Compute the Hessian matrix
# hessian = sp.hessian(f, (x, y))

# # Display the Hessian
# print(hessian)


# # Define the function
# def f(x, y):
#     return x**2 + x*y + y**2

# # Gradient (first derivatives)
# def gradient(x, y):
#     df_dx = 2*x + y
#     df_dy = x + 2*y
#     return np.array([df_dx, df_dy])

# Hessian (second derivatives)
def hessian(x, y):
    d2f_dx2 = 2  # Second derivative w.r.t x
    d2f_dy2 = 2  # Second derivative w.r.t y
    d2f_dxdy = 1 # Mixed derivative
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])