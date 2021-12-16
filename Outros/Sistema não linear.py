from scipy.optimize import fsolve
import numpy as np
from math import exp

def func(x):

    q = 50
    cai = 1
    To = 350
    Tc = 300
    v = 100
    ko = 7.2e10
    e = 72751.63
    r = 8.314472
    Dh = -5e4
    p = 1000
    Cp = 0.239
    ua = 5e4

    return [(q/v)*(cai - x[0]) - ko*exp(-e/(r*x[1]))*x[0], (q/v)*(To - x[1]) - (Dh/(p*Cp))*ko*exp(-e/(r*x[1]))*x[0] + (ua/(v*p*Cp))*(Tc - x[1])]

raiz = fsolve(func, [0.5,1000])
print(raiz)