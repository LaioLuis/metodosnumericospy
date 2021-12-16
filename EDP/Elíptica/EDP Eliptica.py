import numpy as np
from math import sin, pi, cos, exp, sqrt, log
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

kmax = 100
tol = 0.01
lambd = 1.5
n = 20
l = 40
dx = l/n
dy = l/n

To = np.zeros(n)
Tx0 = 75
TxL = 50
Ty0 = 0
TyL = 100

u = np.zeros((n, n, kmax))
erro = np.zeros((n,n))
usol = np.zeros((n,n))
u[(n-1), :, :] = TyL
u[:, 0, :] = Tx0
u[0, :, :] = Ty0
u[:, (n-1), :] = TxL
def liebman(u):
    for k in range(0,kmax-1):
        for i in range(1,n-1):
            for j in range(1,n-1):
                u[i, j, k+1] = (u[i+1][j][k] + u[i-1][j][k+1] + u[i][j+1][k] + u[i][j-1][k+1])/4
                u[i, j, k+1] = lambd*u[i][j][k+1] + (1-lambd)*u[i][j][k]
        cont = 0
        for i in range(1,n-1):
            for j in range(1,n-1):
                erro[i,j] = abs((u[i][j][k+1]-u[i][j][k])/u[i][j][k+1])
                if erro[i,j] < tol:
                    cont = cont+1
            if cont == (n-1)**2:
                usol[:, :] = u[:,:,k+1]



