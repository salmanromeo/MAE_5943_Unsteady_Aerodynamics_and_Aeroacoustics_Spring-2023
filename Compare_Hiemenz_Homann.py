# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 08:58:45 2023

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

def homann(nmax=10, N=40, itermax=40, eProfile=1e-6, eBC=1e-6):
    
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
        
        # Homann Equation
        A = [1/dnn + p[i]/dn for i in range(0, N+1)]
        B = [-2/dnn - f[i] for i in range(0, N+1)]
        C = [1/dnn - p[i]/dn for i in range(0, N+1)]
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
    
ntest_hi, ftest_hi = hiemenz(10, 40)
ntest_ho, ftest_ho = homann(10, 40)

plt.plot(ftest_hi, ntest_hi, label=r'$\Delta{\eta}$ points for Hiemenz flow', linewidth=2, color='red', marker='o', markerfacecolor='blue')
plt.plot(ftest_ho, ntest_ho, label=r'$\Delta{\eta}$ points for Homann flow', linewidth=2, color='green', marker='*', markerfacecolor='yellow')
plt.title(r'Convergence of $\phi^{\prime}$')
plt.xlabel(r'$\phi^{\prime}$')
plt.ylabel(r'$\Delta{\eta}$')
plt.legend(loc='upper left')

plt.show()