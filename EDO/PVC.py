import numpy as np
from math import sin, pi, cos, exp, sqrt, log
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def func(r, t):
    return [t[1], -t[1]/r]
def bc(ya,yb):
    return [ya[0]-150,yb[0]-60]
r1 = 0.06
r2 = 0.08
t1 = 150
t2 = 60
L = 20
k = 20

x = np.linspace(r1, r2, 10)
y_a = np.zeros((2,x.size))
y_a[0] = 1000
sol = solve_bvp(func, bc, x, y_a)
fig = plt.figure(figsize=(10,7))
fig.add_subplot(311)
plt.plot(x, sol.y[0], 'ko-')
plt.title('Solução encontrada')
Tr = np.zeros(x.size)
for i in range(0,10):
    Tr[i] = (log(x[i]/r1)/log(r2/r1))*(t2-t1)+t1
fig.add_subplot(312)
plt.plot(x, Tr, 'ro-')
plt.title('Solução analítica')

a = 2*pi*r2*L - 2*pi*r1*L
q = -k*a*sol.y[0]
fig.add_subplot(313)
plt.plot(x, q, 'go-')
plt.title('Fluxo de Calor')
plt.show()







