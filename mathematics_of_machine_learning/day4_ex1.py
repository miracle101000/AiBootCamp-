import sympy as sp

x = sp.Symbol('x')
f = sp.exp(-x)

#Compute indefinite integral
indefinite_integral = sp.integrate(f, x)
print("Indefinte integral:\n",indefinite_integral)

#Compute definite integral
definite_integral = sp.integrate(f, (x, 0, sp.oo))
print('Definite Integral: ', definite_integral)

