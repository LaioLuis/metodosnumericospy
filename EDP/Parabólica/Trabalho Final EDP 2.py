import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

dt = 0.1
dx = 0.1
L = 10
tf = 10

kmax = round(tf/dt) #Máximo de iterações
n = round(L/dx) #Numero de intervalos


x = np.linspace(0,L,n)
vt = np.linspace(0,tf,kmax)

Da = 0.1
Db = 10
k0 = 0.067
gamma = 1
delt = 1
K = 1
p=round(n/2)
alfa = (Da*dt)/(dx**2)
beta = (Db*dt)/(dx**2)

func = lambda a, b: b*(k0 + (a**2)/(1+(a**2))) - a

b0 = 2
a0 = 0.2683312
a = np.zeros((n,kmax))
#b = np.zeros((n,kmax))
a = [[a0 if i<=round(n/2) else 2*a0*b0 for i in range(n)], [0 for i in range(kmax)]]
b = np.empty((n,kmax))
#a0 = 2*0.2683*b0*random.random()
#a[:,0] = a0
#b[:,0] = b0
#a.fill(a0)
b.fill(b0)

for k in range(0, kmax-1):
  i=0
  a[i][k+1] = alfa*(2*a[i+1][k] - 2*a[i][k]) + a[i][k] + dt*func(a[i][k],b[i][k])
  b[i][k+1] = beta*(2*b[i+1][k] - 2*b[i][k]) + b[i][k] - dt*func(a[i][k],b[i][k])
  for i in range(1, n-1):
    a[i][k+1] = alfa*(a[i+1][k] - 2*a[i][k] + a[i-1][k]) + a[i][k] + dt*func(a[i][k],b[i][k])
    b[i][k+1] = beta*(b[i+1][k] - 2*b[i][k] + b[i-1][k]) + b[i][k] - dt*func(a[i][k],b[i][k])
  i=n-1
  a[i][k+1] = alfa*(2*a[i-1][k] - 2*a[i][k]) + a[i][k] + dt*func(a[i][k],b[i][k])
  b[i][k+1] = beta*(2*b[i-1][k] - 2*b[i][k]) + b[i][k] - dt*func(a[i][k],b[i][k])

#print(a)
a = np.transpose(a)
b = np.transpose(b)
print(b)
# Gráficos
vl = np.linspace(0,L,n)
vt = np.linspace(0,tf,kmax)
fig = plt.figure(figsize = (8,6))
plt.plot(vl, a[:,kmax-1], '-b', label = 'a')
plt.plot(vl, b[:,kmax-1], '-k', label = 'b')
#plt.plot(vt, b[-1,:], '-k', label = '20 min')
#plt.plot(vl, b[:, round((3*kmax)/4)-1], '-g', label = '15 min')
#plt.plot(vl, b[:, round((2*kmax)/4)-1], '-m', label = '10 min')
#plt.plot(vl, b[:, round((1*kmax)/4)-1], '-b', label = '5 min')
plt.title('Perfil de concentração ao longo do comprimento do reator')
plt.xlabel('Comprimento (m)')
plt.ylabel('Concentração (mg/m3)')
#plt.ylim([10,80])
plt.legend()
plt.show()