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


    x = x0
    
    for i in range(3, nt - 1):

        x[i+1] = x[i] + ((55*f(t[i], x[i])-59*f(t[i-1], x[i-1])+37*f(t[i-2], x[i-2])-9*f(t[i-3], x[i-3]))*h)/24

    return x, t


f = lambda t, x: funcao(t,x)

#x0 = np.array([1, 0])
#x0 = np.array([2, 4])
#x0 = np.array([4, 6])
x0 = [0.48, 0.33333333, 0.244898, 0.1875, 0, 0]

t0 = 2.5
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
