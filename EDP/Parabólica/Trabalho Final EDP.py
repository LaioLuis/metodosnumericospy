import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.set_printoptions(precision=3)

dt = 0.05 #Passo de tempo
dx = 0.05 #Passo de espaço
L = 10 #Diametro da célula
tf = 60 #Tempo de simulação em segundos

kmax = round(tf/dt) #Máximo de iterações
n = round(L/dx) #Numero de intervalos


x = np.linspace(0,L,n) #Malha de pontos no espaço
vt = np.linspace(0,tf,kmax) #Malha de pontos no tempo

Da = 0.1 #Coeficiente de difusão de a
Db = 10 #Coeficiente de difusão de b
k0 = 0.067 #Taxa de conversão

alfa = (Da*dt)/(2*dx**2)
beta = (Db*dt)/(2*dx**2)

fa = lambda a, b: b*(k0 + (a**2)/(1+(a**2)))

a = np.empty((n))
b = np.empty((n))
a0 = 0.2683312
b0 = 2
a.fill(2*fa(a0,b0)*0.75)
b.fill(b0)

Aa =  np.diagflat([-alfa for i in range(n-1)], -1) +\
      np.diagflat([1+alfa]+[1+2*alfa for i in range(n-2)]+[1+alfa]) +\
      np.diagflat([-alfa for i in range(n-1)], 1)

Ba =  np.diagflat([alfa for i in range(n-1)], -1) +\
      np.diagflat([1-alfa]+[1-2*alfa for i in range(n-2)]+[1-alfa]) +\
      np.diagflat([alfa for i in range(n-1)], 1)
        
Ab =  np.diagflat([-beta for i in range(n-1)], -1) +\
      np.diagflat([1+beta]+[1+2*beta for i in range(n-2)]+[1+beta]) +\
      np.diagflat([-beta for i in range(n-1)], 1)
        
Bb =  np.diagflat([beta for i in range(n-1)], -1) +\
      np.diagflat([1-beta]+[1-2*beta for i in range(n-2)]+[1-beta]) +\
      np.diagflat([beta for i in range(n-1)], 1)

f = lambda A, B: np.multiply(dt, np.subtract(np.multiply(B, 
                     np.add(k0, np.divide(np.multiply(A,A), np.add(1, np.multiply(A,A))))), A))

Ar = []
Br = []

Ar.append(a)
Br.append(b)

for i in range(1,kmax):
    An = np.linalg.solve(Aa,Ba.dot(a) + f(a,b))
    Bn = np.linalg.solve(Ab,Bb.dot(b) - f(a,b))
    a = An
    b = Bn

    Ar.append(a)
    Br.append(b)

Ar = np.transpose(Ar)
Br = np.transpose(Br)

plt.plot(x, a, '-m', label = 'a')
plt.plot(x, b, '-y', label = 'b')
plt.xlabel('Posição na célula (μm)')
plt.ylabel('Concentração (μM)')
plt.legend()
plt.show()
plt.plot(vt, Ar[-1,:], '-m', label = 'a')
plt.plot(vt, Br[-1][:], '-y', label = 'b')
plt.xlabel('Tempo (s)')
plt.ylabel('Concentração (μM)')
plt.legend()
plt.show()
plt.pcolormesh(vt, x, Ar, cmap = 'plasma', vmin = 2*fa(a0,b0)*0.75, vmax = a[-1], shading='nearest')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição na célula (μm)')
plt.show()
plt.pcolormesh(vt, x, Br, cmap = 'plasma', vmin = b[-1], vmax = b0, shading='nearest')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição na célula (μm)')
plt.show()