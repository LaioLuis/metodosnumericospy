import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
#comprimento, tempo e passo para os dois
dt = 0.5
dr = 0.1
l = 1
tf = 10
uf = 1
#Parametros
alfa = dt/(2*(dr**2))
beta = dt/(4*dr)

kmax = round(tf/dt) #Máximo de iterações
n = round(l/dr) #Numero de intervalos

r = np.linspace(0,l,n+1)
vt = np.linspace(0,kmax,kmax)

u = np.empty((n+1,kmax))
u0 = 0
u.fill(u0)
u[n,:] = uf

a = np.zeros((n,n))
b = np.zeros((n))
for m in range(1,kmax-1):
    for i in range(n):
        if i==0:
            a[i,:] = [2*alfa+1 if j==0 else -2*alfa if j==1 else 0 for j in range(n)]
            b[i] = (1-2*alfa)*u[i,m] + 2*alfa*u[i+1,m] 
        elif i == n-1:
            a[i,:] = [beta/r[i]-alfa if j==n-2 else 2*alfa+1 if j==n-1 else 0 for j in range(n)]
            b[i] = (alfa - beta/r[i])*u[i-1,m] + (1-2*alfa)*u[i,m] + 2*(alfa + beta/r[i])
        else:
            a[i,:] = [beta/r[i]-alfa if j==i-1 else -alfa-(beta/r[i]) if j==i+1 else 2*alfa+1 if j==i else 0 for j in range(n)]
            b[i] = (alfa - beta/r[i])*u[i-1,m] + (1-2*alfa)*u[i,m] + (alfa + beta/r[i])*u[i+1,m]

    u[:-1,m+1] = np.linalg.solve(a, b)
plt.plot(r,u[:,kmax-1], '.-k')
plt.plot(r,u[:, round((3*kmax)/4)+1], '.-g')
plt.plot(r,u[:, round((2*kmax)/4)+1], '.-m')
plt.plot(r,u[:, round((1*kmax)/4)+1], '.-b')
plt.xlabel('Raio')
plt.ylabel('Fluxo de calor')
plt.show()