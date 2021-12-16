import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from math import sin, pi, cos, sqrt, exp

def funcao(t, x):

    c = 5
    k = 20
    m = 20

    #xi = np.array([x[1], (-c*x[1] - k*x[0])/m])
    #xi = np.array([-2*x[0] + 4*exp(-t), -(x[0]*x[1])/3])
    #xi = np.array([-0.5*x[0], 4 - 0.3*x[1] - 0.1*x[0]])
    xi = (-2*x)/t

    return xi

def euler(f, x0, t0, tf, h):

    t = np.arange(t0, tf+h, h)
    nt = t.size

    x = np.zeros(nt)
    fx = np.zeros(nt)
    xp = np.zeros(nt)
    cdx = np.zeros(nt)


    x[0] = x0
    tol = 0.01
    
    for i in range(nt - 1):

        fx[i] = f(t[i], x[i])
        xp[i+1] = x[i] + h*fx[i]
        fx[i+1] = f(t[i+1], xp[i+1])
        cdx[i] = (fx[i]+fx[i+1])/2
        x[i+1] = x[i] + h*cdx[i]
        erro = abs((x[i+1] - xp[i+1]))/xp[i+1]
        while erro>tol:
            xp[i+1] = x[i+1]
            fx[i+1] = f(t[i+1], xp[i+1])
            cdx[i] = (fx[i]+fx[i+1])/2
            x[i+1] = x[i] + h*cdx[i]
            erro = abs((x[i+1] - xp[i+1]))/xp[i+1]

    return x, t


f = lambda t, x: funcao(t,x)

#x0 = np.array([1, 0])
#x0 = np.array([2, 4])
#x0 = np.array([4, 6])
x0 = 0.1875

t0 = 4
tf = 5
h = 0.5

x, t = euler(f, x0, t0, tf, h)
print(t)
print(x)
#print(x[1,:])
plt.plot(t, x)
#plt.grid()
#plt.show()
#plt.plot(t, x[1,:])
#plt.ylim([-2,2])
plt.grid()
plt.show()
