import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

l = 40
kmax = 100
n=20
dx = int(l/n)
dy = int(l/n)
k = 0.49
# Initialize solution: the grid of u(k, i, j)
u = np.empty((kmax, n, n))

# Initial condition everywhere inside the grid
u_initial = 0

# Boundary conditions
u_top = 100.0
u_left = 75.0
u_bottom = 0.0
u_right = 50.0

# Set the initial condition
u.fill(u_initial)

# Set the boundary conditions
u[:, (n-1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (n-1):] = u_right

def calculate(u):
    for k in range(0, kmax-1, 1):
        for i in range(1, n-1):
            for j in range(1, n-1):
                #Equação discretizada
                u[k+1, i, j] = (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1])/4

    return u

def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*dx:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

# Do the calculation here
u = calculate(u)
usol = u[-1,:,:]
def animate(k):
    plotheatmap(u[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=kmax, repeat=False)
anim.save("edp eliptica.gif")
plt.show()
vx = np.linspace(0,l,n)
vy = np.linspace(0,l,n)
x, y = np.meshgrid(vx, vy)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, usol,cmap=plt.cm.jet, edgecolor='none')
ax.set_title('Surface plot')
plt.show()
px, py = np.gradient(usol, dx, dy)
qx = -k*px
qy = -k*py
print(qx.shape)
fig, ay = plt.subplots()
ay.quiver(qy, qx)

plt.show()
