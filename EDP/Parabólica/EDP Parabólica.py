import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

k = 0.835
dt = 0.1
dx = 1
l = 10
tf = 12
kmax = round(tf/dt)
lambd = k*dt/(dx**2)
Tx0 = 100
TxL = 50
n = round(l/dx)

t = np.empty((n,kmax))
t0 = 0
t.fill(t0)
t[0,:] = Tx0
t[n-1,:] = TxL

for k in range(0, kmax-1):
    for i in range(1, n-1):
        t[i][k+1] = lambd*(t[i+1][k]-2*t[i][k]+t[i-1][k]) + t[i][k]
        #print(t)

vl = np.linspace(0,l,n)
vt = np.linspace(0,kmax,kmax)
plt.plot(vl, t[:,kmax-1], '.-k')
plt.plot(vl, t[:, round((3*kmax)/4)+1], '.-g')
plt.plot(vl, t[:, round((2*kmax)/4)+1], '.-m')
plt.plot(vl, t[:, round((1*kmax)/4)+1], '.-b')
plt.show()
plt.plot(vt, t[n-2, :])
plt.show()
