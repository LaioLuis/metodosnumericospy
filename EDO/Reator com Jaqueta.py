import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from math import sin, pi, cos, sqrt, exp

def funcao(t, x): #Definição do sistema de equações

    q = 50
    cai = 1
    Tc = 300
    v = 100
    ko = 7.2e10
    e = 72751.63
    r = 8.314472
    Dh = -5e4
    p = 1000
    Cp = 0.239
    ua = 5e4

    xi = np.array([(q/v)*(cai - x[0]) - ko*exp(-e/(r*x[1]))*x[0],
                     (q/v)*(To - x[1]) - (Dh/(p*Cp))*ko*exp(-e/(r*x[1]))*x[0] + (ua/(v*p*Cp))*(Tc - x[1])])

    return xi

def RK4(f, x0, t0, tf, h): #Método de Runge-Kutta de quarta ordem

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

To = 350 #Temperatura de alimentação inicial

f = lambda t, x: funcao(t,x) #Comando equivalente ao feval do matlab

x0 = np.array([0.5, 350]) #Condições iniciais de Ca e T

t0 = 0 #Tempo inicial
tf = 20 #Tempo final
tf1 = 3*tf #Tempo final após alteração das condições de temperatura
h = 1 #Passo

x, t = RK4(f, x0, t0, tf, h)
print(x[0,:]) #Valores de Ca
print(x[1,:]) #Valores de T

x01 = np.array([x[0,-1], x[1,-1]]) #Nova condição inicial para os degraus

To = 350*0.95 #Degrau de -5%
x1, tx = RK4(f, x01, tf, tf1, h)

To = 350*1.05 #Degrau de +5%
x2, tx = RK4(f, x01, tf, tf1, h)

To = 350*0.88 #Degrau de -12%
x3, tx = RK4(f, x01, tf, tf1, h)

To = 350*1.12 #Degrau de +12%
x4, tx = RK4(f, x01, tf, tf1, h)

fig = plt.figure(figsize=(9,7))
fig.add_subplot(211)
plt.plot(t, x[0,:])
plt.plot(tx, x1[0,:], label = '-5%')
plt.plot(tx, x2[0,:], label = '+5%')
plt.plot(tx, x3[0,:], label = '-12%')
plt.plot(tx, x4[0,:], label = '+12%')
plt.ylim([0.4,1])
plt.ylabel('Ca(mol/L)')
plt.legend()

fig.add_subplot(212)
plt.plot(t, x[1,:])
plt.plot(tx, x1[1,:], label = '-5%')
plt.plot(tx, x2[1,:], label = '+5%')
plt.plot(tx, x3[1,:], label = '-12%')
plt.plot(tx, x4[1,:], label = '+12%')
plt.ylim([300,350])
plt.ylabel('T(K)')
plt.xlabel('tempo(min)')
plt.legend()
plt.show()


