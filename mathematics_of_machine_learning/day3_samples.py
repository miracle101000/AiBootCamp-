import sympy as sp

x = sp.Symbol('x')
f = x**2
derivative = sp.diff(f,x)
# print("Derivative\n", derivative)

x, y = sp.symbols('x y')
f = x**2 + y**2
grad_x = sp.diff(f,x)
grad_y = sp.diff(f,y)
# print("Partial derivative\n",grad_x,grad_y)