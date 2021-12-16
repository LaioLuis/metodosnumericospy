import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
np.set_printoptions(precision=2)

alfa = 0.0005952
dt = 32
dx = 0.25
l = 1.25
tf = 32
eps = 0.1989
p = 1900
C = 0.84
kmax = round(tf/dt) + 1
lambd1 = alfa*dt/(dx**2)
lambd2 = dt*eps/(p*C)

n = round(l/dx) + 1
print(n)
t = np.zeros((n,kmax))
x = np.linspace(0,l,n)
#vt = np.linspace(0,tf,kmax)
I = lambda vt: -2.4644e-6*(vt**2) + 0.060438*vt + 547.9
t[0][1] = 8.4597e-13*(tf**3) - 5.849e-8*(tf**2) + 1.1978e-3*tf + 301.51
t[n-1][1] = 3.2575e-12*(tf**3) - 2.187e-7*(tf**2) + 3.6667e-3*tf + 300.904
for i in range(0,n):
    t[i][0] = -25.825*(x[i]**2) + 35.847*x[i] + 302.59
for k in range(0, kmax-1):
    for i in range(1, n-1):
        t[i][k+1] = lambd1*(t[i+1][k]-2*t[i][k]+t[i-1][k]) + t[i][k] + lambd2*I(tf)
        #print(t)
print(x)
#print(vt)
print(t)
plt.plot(x, t[:,kmax-1], '.-k')
#plt.plot(x, t[:, round((3*kmax)/4)-1], '.-g')
#plt.plot(x, t[:, round((2*kmax)/4)-1], '.-m')
#plt.plot(x, t[:, round((1*kmax)/4)-1], '.-b')
plt.show()