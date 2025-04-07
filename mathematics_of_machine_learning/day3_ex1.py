import sympy as sp

#Define a function
x = sp.Symbol('x') #One symbol derivative
f =  x**3 - 5*x + 7

#Compute Derivative 
derivative = sp.diff(f,x)

print(f"Derivative\n:", derivative)