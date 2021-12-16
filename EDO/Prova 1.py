import numpy as np
from math import sin, pi, cos, exp, sqrt, log
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns

def sistedo(t, x): #Definição do sistema de EDO

    q = 1
    miMax = 2
    y = 0.8
    ks = 1.2
    b = 0.1
    fp = 0.1
    #X = x[0], Z = x[1], 
    mi = miMax*(x[2]/(ks+x[2]))
    xi = np.array([(mi-b)*x[0]+(q*(Xin - x[0]))/V, fp*b*x[0] + (q*(Zin - x[1]))/V, (mi/y - (1-fp)*b)*-x[0] + (q*(Sin - x[2]))/V])

    return xi

def RK4(f, x0, t0, tf, h): #Método de Runge-Kutta de 4ª ordem

    t = np.arange(t0, tf+h, h)
    nt = t.size

    nx = x0.size
    x = np.zeros((nx, nt))

    x[:,0] = x0
    
    for i in range(nt - 1):

        k1 = h*f(t[i], x[:,i])
        k2 = h*f(t[i] + h/2, x[:,i] + k1/2)
        k3 = h*f(t[i] + h/2, x[:,i] + k2/2)
        k4 = h*f(t[i] + h, x[:,i] + k3)

        dx = (k1 + 2*k2 + 2*k3 + k4)/6

        x[:,i+1] = x[:,i] + dx

    return x, t

f = lambda t, x: sistedo(t,x)

x0 = np.array([5, 0.035, 3.6]) #Condições iniciais
#Valores de X, Z e S iniciais
Xin=0
Zin=0
Sin=10
V=0.712 #Volume para questão 1

t0 = 0
tf = 2
h = 0.05

x, t = RK4(f, x0, t0, tf, h)
#print(t)
fig = plt.figure(figsize=(10,5))
print('Valor de X: %1.5f' %x[0,-1])
print('Valor de Z: %1.5f' %x[1,-1])
print('Valor de S: %1.5f' %x[2,-1])
plt.plot(t, x[0,:], 'y-', label = 'X')
plt.plot(t, x[1,:], 'g-', label = 'Z')
plt.plot(t, x[2,:], 'b-', label = 'S')
fig.legend()
plt.title('Gráfico para questão 1')
plt.show()
V1=0.712
Vt = 1.1
N = 3 #Quantidade de reatores
V23 = (Vt-V1)/(N-1) #Atualização dos Volumes para os reatores 2 e 3
V = V23
#Alteração das condições iniciais de X, Z e S para os reatores seguintes
#print(t.size)
for i in range(0,t.size):
  Xin=x[0,i]
  Zin=x[1,i]
  Sin=x[2,i]
  x2, t = RK4(f, x0, t0, tf, h)
  #print(x2[:,i])


print('Valor de X no reator 2: %1.5f' %x2[0,-1])
print('Valor de Z no reator 2: %1.5f' %x2[1,-1])
print('Valor de S no reator 2: %1.5f' %x2[2,-1])
fig = plt.figure(figsize=(10,5))
plt.plot(t, x2[0,:],'y-', label = 'X do reator 2')
plt.plot(t, x2[1,:],'g-', label = 'Z do reator 2')
plt.plot(t, x2[2,:],'b-', label = 'S do reator 2')


#Alteração das condições iniciais de X, Z e S para os reatores seguintes
for i in range(0,t.size):
  Xin=x2[0,i]
  Zin=x2[1,i]
  Sin=x2[2,i]
  x3, t = RK4(f, x0, t0, tf, h)
  #print(x3[:,i])

print('\nValor de X no reator 3: %1.5f' %x3[0,-1])
print('Valor de Z no reator 3: %1.5f' %x3[1,-1])
print('Valor de S no reator 3: %1.5f' %x3[2,-1])
plt.plot(t, x3[0,:],'y--', label = 'X do reator 3')
plt.plot(t, x3[1,:],'g--', label = 'Z do reator 3')
plt.plot(t, x3[2,:],'b--', label = 'S do reator 3')
fig.legend()
plt.title('Gráfico para os reatores 2 e 3')
plt.show()