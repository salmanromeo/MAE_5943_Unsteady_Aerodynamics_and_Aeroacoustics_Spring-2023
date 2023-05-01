# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:53:15 2023

@author: Shafi Romeo
"""

import numpy as np
import matplotlib.pyplot as plt

def blasius(nmax=10, N=40, itermax=40, eProfile=1e-6, eBC=1e-6):
    
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
    H = np.zeros(N+1)
    
    p = np.zeros(N+1)           # p = f(η)
    h = np.zeros(N+1)           # h = f'(η)
    h_prime = np.zeros(N+1)     # h' = f"(η)
    n = np.zeros(N+1)

    n = np.array([(i-1)*dn for i in range(1, N+2)])

    # BCs
    h[0] = 0.0   # h(n=0) = 0
    h[N] = 1.0   # h(n=∞) = 1
    p[0] = 0.0   # p(n=0) = 0

    G[0] = 0.0
    H[0] = 0.0

    # Solution initialization
    h = np.array([(i-1)/N for i in range(1, N+2)])
    
    for i in range(1, N+1):
        p[i] = p[i-1] + (h[i] + h[i-1])*dn/2
      
    print("iter     error            convergence of f(n→∞)")
    print("-----------------------------------------------")

    while eProfile<=errorProfile and iter<itermax:
        
        # Hiemenz Equation
        A = [2/dnn + p[i]/(2*dn) for i in range(0, N+1)]
        B = [-4/dnn for i in range(0, N+1)]
        C = [2/dnn - p[i]/(2*dn) for i in range(0, N+1)]
        D = [0 for i in range(0, N+1)]

        for i in range(1, N):
            G[i] = - ( C[i] * G[i-1] + D[i] ) / (B[i] + C[i] * H[i-1])
            H[i] = -                 A[i]  /(B[i] + C[i] * H[i-1])

        hold = h.copy()

        for i in range(N-1, 0, -1):
            h[i] = G[i] + H[i] * h[i+1]
            h_prime[i-1] = (h[i+1] - h[i-1]) / (2 * dn) # approximate f" using centered finite difference

        errorProfile = np.max(np.abs(hold - h))
        errorBC = np.abs(h[N] - h[N-1])
        
        for i in range(1, N):
            p[i] = p[i-1] + (h[i] + h[i-1])*dn/2

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
        
    return n, p, h, h_prime

ntest, ptest, htest, h_primetest = blasius(10, 40)

#%%
# Wall shear stress
L = ntest[-1]
ro = 1.204             # density of air at 20°C (kg/m3)
u_inf = 10             # free stream velocity (m/s)
nu = 1.516e-5          # kinematic viscosity viscosity of air at 20°c (m2/s)
print("f''(η=0) =",h_primetest[0])
tau_w = ro*u_inf*np.sqrt(nu*u_inf/ntest)*h_primetest[0]
print("Wall shear stress =", tau_w, "Pa")
plt.plot(ntest, tau_w, label="Wall shear stress", linewidth=2, color='red', marker='o', markerfacecolor='blue')
plt.title("Wall shear stress at different location")
plt.xlabel("$x [m]$")
plt.ylabel("$τ_w [Pa]$")
plt.legend(loc='upper right')
plt.grid(linestyle = '--', linewidth = 0.2)
plt.show()

#%%
# Skin-friction coefficient
re = u_inf*ntest/nu
C_f = 2*h_primetest[0]/(np.sqrt(re))
print("Skin-friction coefficient =", C_f)
plt.plot(ntest, C_f, label="Skin-friction coefficient", linewidth=2, color='red', marker='o', markerfacecolor='blue')
plt.title("Skin-friction coefficient at different location")
plt.xlabel("$x [m]$")
plt.ylabel("$C_f$")
plt.legend(loc='upper right')
plt.grid(linestyle = '--', linewidth = 0.2)
plt.show()

#%%
# Boundary-layer thickness
delta = 5*ntest/(np.sqrt(re))
delta[0] = 0
print("Boundary-layer thickness =", delta)
# Displacement thickness
delta_1 = 1.7208*ntest/(np.sqrt(re))
delta_1[0] = 0
print("Displacement thickness =", delta_1)
# Momentum thickness
delta_2 = 2*h_primetest[0]*ntest/(np.sqrt(re))
delta_2[0] = 0
print("Momentum thickness =", delta_1)
plt.plot(ntest, delta, label="Boundary-layer thickness (δ)", linewidth=2, color='red', marker='o', markerfacecolor='blue')
plt.plot(ntest, delta_1, label="Displacement thickness (δ1)", linewidth=2, color='green', marker='o', markerfacecolor='purple')
plt.plot(ntest, delta_2, label="Momentum thickness (θ)", linewidth=2, color='orange', marker='o', markerfacecolor='gray')
plt.title("Thickness at different location")
plt.xlabel("$x [m]$")
plt.ylabel("$δ^* [m]$")
plt.legend(loc='upper left')
plt.grid(linestyle = '--', linewidth = 0.2)
plt.show()

#%%
plt.plot(htest, ntest, label="Δη points", linewidth=2, color='red', marker='o', markerfacecolor='blue')
plt.title("Blasius Flow - Convergence of h")
plt.xlabel("$h$")
plt.ylabel("$Δη$")
plt.legend(loc='upper left')
plt.grid(linestyle = '--', linewidth = 0.2)
plt.show()

plt.plot(ptest, ntest, label="$f(η)$", linewidth=2, color='red', marker='v', markersize=5, alpha=0.9, markerfacecolor='red')
plt.plot(htest, ntest, label="$f'(η)$", linewidth=2, color='blue', marker='*', markersize=5, alpha=0.6, markerfacecolor='blue')
plt.plot(h_primetest, ntest, label="$f''(η)$", linewidth=2, color='green', marker='s', markersize=5, alpha=0.6, markerfacecolor='green')
plt.title("Solution of Blasius Flow")
plt.xlabel("$f, f'$ and $f''$")
plt.ylabel("$Δη$")
plt.xlim(0,2)
plt.ylim(0,10)
plt.legend(loc='upper left')
plt.grid(linestyle = '--', linewidth = 0.2)
plt.show()

ns = []
hs = []
for n_max in [5, 10, 20]:
    n, p, h, h_prime = blasius(n_max)
    ns.append(n)
    hs.append(h)

plt.figure()
plt.plot(hs[0], ns[0], label="ηmax = 5", linewidth=1, color='red', marker='v', markersize=5, alpha=0.9, markerfacecolor='red')
plt.plot(hs[1], ns[1], label="ηmax = 10", linewidth=1, color='blue', marker='*', markersize=5, alpha=0.6, markerfacecolor='blue')
plt.plot(hs[2], ns[2], label="ηmax = 20", linewidth=1, color='green', marker='s', markersize=5, alpha=0.6, markerfacecolor='green')
plt.title("Blasius Flow - Effect of Viscous Layer Height")
plt.xlabel("$h$")
plt.ylabel("$Δη$")
plt.legend(loc='upper left')
plt.grid(linestyle = '--', linewidth = 0.2)
plt.show()

ns = []
hs = []
for N in [10, 20, 40, 80]:
    n, p, h, h_prime = blasius(10, N)
    ns.append(n)
    hs.append(h)

plt.figure()
plt.scatter(hs[0], ns[0], label="N = 10", marker='o', s=80, alpha=0.6, color='purple')
plt.scatter(hs[1], ns[1], label="N = 20", marker='^', s=70, alpha=0.6, color='red')
plt.scatter(hs[2], ns[2], label="N = 40", marker='v', s=60, alpha=0.6, color='green')
plt.scatter(hs[3], ns[3], label="N = 80", marker='s', s=10, alpha=0.6, color='blue')
plt.title("Blasius Flow - Effect of Grid Resolution")
plt.xlabel("$h$")
plt.ylabel("$Δη$")
plt.legend(loc='upper left')
plt.grid(linestyle = '--', linewidth = 0.2)
plt.show()