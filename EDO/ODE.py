from scipy.integrate import solve_ivp
import numpy as np

def func(t,z):
    x, y = z
    dx = y
    dy = -1001*x -1000*y
    return [dx, dy]

sol = solve_ivp(func, [0,5], [1,0])
#t = np.arange(0,6,1)
#z = sol.sol(t)
print(sol.y)
