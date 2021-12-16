import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML
#%matplotlib inline
#Tamanho da placa
l = 40
dy = 1
#Máximo de iterações
kmax = 100

#Constante do material
kl = 0.49
#Fluxo de calor
q = 1
# Condição inicial dentro da malha
u_initial = 0

# Condições de contorno
u_top = 120.0
u_left = 60.0
u_right = 50.0

u = np.empty((kmax, l, l))
def calculate(u, n):
    # Initialize solution: the grid of u(k, i, j)
    u = np.empty((kmax, n, n))
    # Set the initial condition
    u.fill(u_initial)
    # Set the boundary conditions
    u[:, (n-1):, :] = u_top
    u[:, :, :1] = u_left
    #u[:, :1, 1:] = u_bottom
    u[:, :, (n-1):] = u_right
    for k in range(0, kmax-1, 1):
        for i in range(1, n-1):
            for j in range(1, n-1):
                #Equação discretizada
                u[k+1, i, j] = (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1])/4
                u[k+1, 0, j] = (u[k][0][j+1] + u[k][0][j-1] + 2*u[k][1][j] + 2*dy*q/kl)/4


    return u

def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperatura em t = {k:.1f} unidade de tempo")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=120) #Lembrar de alterar o valor máximo e mínimo de acordo com a maior e menor temperatura de contorno
    plt.colorbar()

    return plt

# Calculo com o tamanho l da placa
u = calculate(u, l)
def animate(k):
    plotheatmap(u[k], k)

anim = animation.FuncAnimation(plt.figure(figsize = (10,7)), animate, interval=50, frames=kmax)
anim.save("prática 2 edp epliptica.gif")
#HTML(anim.to_html5_video())