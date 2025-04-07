import sympy as sp

#Define a multivariable 
x, y = sp.symbols('x y') #more than one symbol for partial derivative
f = x**2 + 3*y**2 -4*x*y

#Compute partial derivatives
grad_x=  sp.diff(f,x)
grad_y =  sp.diff(f,y)

print("Gradients: ")
print("Grad X:", grad_x)
print("GradY:",grad_y)




