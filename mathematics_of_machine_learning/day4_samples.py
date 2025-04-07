import sympy as sp

x = sp.Symbol('x')
f = x**2

definite_integral = sp.integrate(f, (x, 0, 2))
indefinite_integral = sp.integrate(f, x)

print(definite_integral)
print(indefinite_integral)

