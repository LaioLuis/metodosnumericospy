import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

kl = 0.2
dt = 0.005 #Passo de tempo
dx = 1 #Passo de comprimento
l = 10 #Comprimento do reator
tf = 20 #Tempo final
D = 2
U = 1
Cin = 100
####
lambda1=D/dx**2
lambda2=U/(2*dx)
####

kmax = round(tf/dt) #Máximo de iterações
n = round(l/dx) #Número de pontos

C = np.empty((n,kmax))
C0 = 0
C.fill(C0)

for k in range(0, kmax-1):
  i=0
  C[i][k+1] = dt*(lambda1*(2*C[i+1][k]-2*C[i][k] - (2*dx/D)*(C[i][k]-Cin))-lambda2*((2*dx/D)*(C[i][k]-Cin))-kl*C[i][k])+C[i][k]
  for i in range(1, n-1):
    C[i][k+1] = dt*(lambda1*(C[i+1][k]-2*C[i][k]+C[i-1][k])-lambda2*(C[i+1][k]-C[i-1][k])-kl*C[i][k])+C[i][k]
  i=n-1
  C[i][k+1]=dt*(lambda1*(2*C[i-1][k]-2*C[i][k])-kl*C[i][k])+C[i][k]
    

# Gráficos
vl = np.linspace(0,l,n)
vt = np.linspace(0,kmax,kmax)
fig = plt.figure(figsize = (8,8))
plt.plot(vl, C[:,kmax-1], '.-k', label = '20 min')
plt.plot(vl, C[:, round((3*kmax)/4)-1], '.-g', label = '15 min')
plt.plot(vl, C[:, round((2*kmax)/4)-1], '.-m', label = '10 min')
plt.plot(vl, C[:, round((1*kmax)/4)-1], '.-b', label = '5 min')
plt.title('Perfil de concentração ao longo do comprimento do reator')
plt.xlabel('Comprimento (m)')
plt.ylabel('Concentração (mg/m3)')
plt.ylim([10,80])
plt.legend()
plt.show()