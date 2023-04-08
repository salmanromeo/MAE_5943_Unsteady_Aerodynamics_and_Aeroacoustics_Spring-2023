# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 06:52:07 2023

@author: Shafi Romeo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def hiemenz(nmax=10, N=40, itermax=40, eProfile=1e-6, eBC=1e-6):
    
    dn = nmax / N
    dnn = dn ** 2
    
    iter = 0
    errorProfile = 1.0
    errorBC = 1.0

    # Initialization of the arrays
    A = np.zeros(N+1)
    B = np.zeros(N+1)
    C = np.zeros(N+1)
    D = np.zeros(N+1)

    G = np.zeros(N+1)
    F = np.zeros(N+1)
    
    p = np.zeros(N+1)
    f = np.zeros(N+1)
    n = np.zeros(N+1)

    n = np.array([(i-1)*dn for i in range(1, N+2)])

    # BCs
    f[0] = 0.0   # f(n=0) = 0
    f[N] = 0.0   # f(n=∞) = 0
    p[0] = 0.0   # f(n=0) = 0

    G[0] = 0.0
    F[0] = 0.0

    # Solution initialization
    f = np.array([(i-1)/N for i in range(1, N+2)])
    
    for i in range(1, N+1):
        p[i] = p[i-1] + (f[i] + f[i-1])*dn/2
      
    print("iter     error            convergence of f(n→∞)")
    print("-----------------------------------------------")

    while eProfile<=errorProfile and iter<itermax:
        
        # Hiemenz Equation
        A = [1/dnn + p[i]/(2*dn) for i in range(0, N+1)]
        B = [-2/dnn - f[i] for i in range(0, N+1)]
        C = [1/dnn - p[i]/(2*dn) for i in range(0, N+1)]
        D = [1 for i in range(0, N+1)]

        for i in range(1, N):
            G[i] = - ( C[i] * G[i-1] + D[i] ) / (B[i] + C[i] * F[i-1])
            F[i] = -                 A[i]  /(B[i] + C[i] * F[i-1])

        hold = f.copy()

        for i in range(N-1, 0, -1):
            f[i] = G[i] + F[i] * f[i+1]

        errorProfile = np.max(np.abs(hold - f))
        errorBC = np.abs(f[N] - f[N-1])
        
        for i in range(1, N):
            p[i] = p[i-1] + (f[i] + f[i-1])*dn/2

        iter += 1
        print("{0:4d} {1:16.6e} {2:16.6e}".format(iter, errorProfile, errorBC))
    
    if errorProfile <= eProfile:
        print("")
        print("Solution converged!")
        print("The maximum change between consecutive profiles is less than the error criteria eProfile",eProfile)

    if errorBC <= eBC:
        print("")
        print("Solution for the boundary condition converged!")
        print("The difference between h(N) and h(N+1) is less than the error criteria eBC=",eBC)
        
    return n, f

ntest, ftest = hiemenz(10, 40)

plt.plot(ftest, ntest, label="Δη points", linewidth=2, color='red', marker='o', markerfacecolor='blue')
plt.title("Hiemenz Flow - Convergence of h")
plt.xlabel("$h$")
plt.ylabel("$Δη$")
plt.legend(loc='upper left')

plt.show()

ns = []
fs = []
for n_max in [5, 10, 20]:
    n, f = hiemenz(n_max)
    ns.append(n)
    fs.append(f)

plt.figure()
plt.plot(fs[0], ns[0], label="ηmax = 5", linewidth=1, color='red', marker='v', markersize=5, alpha=0.9, markerfacecolor='red')
plt.plot(fs[1], ns[1], label="ηmax = 10", linewidth=1, color='blue', marker='*', markersize=5, alpha=0.6, markerfacecolor='blue')
plt.plot(fs[2], ns[2], label="ηmax = 20", linewidth=1, color='green', marker='s', markersize=5, alpha=0.6, markerfacecolor='green')
plt.title("Hiemenz Flow - Effect of ηmax")
plt.xlabel("$h$")
plt.ylabel("$Δη$")
plt.legend(loc='upper left')

plt.show()

ns = []
fs = []
for N in [10, 20, 40, 80]:
    n, f = hiemenz(10, N)
    ns.append(n)
    fs.append(f)

plt.figure()
plt.scatter(fs[0], ns[0], label="N = 10", marker='o', s=80, alpha=0.6, color='purple')
plt.scatter(fs[1], ns[1], label="N = 20", marker='^', s=70, alpha=0.6, color='red')
plt.scatter(fs[2], ns[2], label="N = 40", marker='v', s=60, alpha=0.6, color='green')
plt.scatter(fs[3], ns[3], label="N = 80", marker='s', s=10, alpha=0.6, color='blue')
plt.title("Hiemenz Flow - Effect of Grid Resolution")
plt.xlabel("$h$")
plt.ylabel("$Δη$")
plt.legend(loc='upper left')

plt.show()