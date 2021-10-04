# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:04:56 2020

Forward Solver for the FitzHugh-Nagumo Model
For more details, please check our paper: 
    "Forward Sensitivity Analysis of the FitzHugh–Nagumo System: Parameter Estimation"
    By: Shady E. Ahmed, Omer San, and Sivaramakrishnan Lakshmivarahan 

For coding questions and/or suggestions, please contact Shady Ahmed at shady.ahmed@okstate.edu
"""


#%% Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt


#%% Define Functions
# compute the cubic root of x
def cubic_root(x):
    return math.copysign(math.pow(abs(x), 1.0/3.0), x)


# Right-hand side of model
def rhs(X,a,b,tau,I):
    # V=X[0], W=X[1]
    f = np.zeros(2)
    f[0] = X[0] - (1/3)*(X[0])**3 - X[1] + I
    f[1] = (X[0] + a - b*X[1])/tau
    return f
    
 
# Time Integrator
def Euler(X,a,b,tau,I,dt): #1st order Euler Scheme
    f = rhs(X,a,b,tau,I)
    Xn = X + dt*f
    return Xn


def RK4(X,a,b,tau,I,dt): #4th order RK Scheme
    k1 = rhs(X,a,b,tau,I)
    k2 = rhs(X+k1*dt/2,a,b,tau,I)
    k3 = rhs(X+k2*dt/2,a,b,tau,I)
    k4 = rhs(X+k3*dt,a,b,tau,I)
    Xn = X + (dt/6)*(k1+2*k2+2*k3+k4)
    return Xn


# Jacobian of right-hand side of model
def Jrhs(X,a,b,tau,I):
    # V=X[0], W=X[1]
    df = np.zeros([2,4])
    df[0,0] = 1 - (X[0])**2
    df[0,1] = -1
    df[0,2] = 0
    df[0,3] = 0

    df[1,0] = 1/tau
    df[1,1] = -b/tau
    df[1,2] = 1/tau
    df[1,3] = -X[1]/tau
    return df
    

# Compute model Jacobians
def JEuler(X,a,b,tau,I,dt): #1st order Euler Scheme
    df = Jrhs(X,a,b,tau,I)
    dM = np.eye(2,4) + dt*df
    
    dMx = dM[:,:2]
    dMa = dM[:,2:]

    return dMx, dMa

    
def JRK4(X,a,b,tau,I,dt): #4th order RK Scheme
    k1 = rhs(X,a,b,tau,I)
    k2 = rhs(X+k1*dt/2,a,b,tau,I)
    k3 = rhs(X+k2*dt/2,a,b,tau,I)
    #k4 = rhs(X+k3*dt,a,b,tau,I)
    
    dk1 = Jrhs(X,a,b,tau,I)
    dk2 = Jrhs(X+k1*dt/2,a,b,tau,I) @ (np.eye(4)+(dt/2)*np.vstack([dk1,np.zeros([2,4])]))
    dk3 = Jrhs(X+k2*dt/2,a,b,tau,I) @ (np.eye(4)+(dt/2)*np.vstack([dk2,np.zeros([2,4])]))
    dk4 = Jrhs(X+k3*dt,a,b,tau,I) @ (np.eye(4)+(dt)*np.vstack([dk3,np.zeros([2,4])]))


    dM = np.eye(2,4) + (dt/6)*(dk1 + 2*dk2 + 2*dk3 + dk4)
    dMx = dM[:,:2]
    dMa = dM[:,2:]

    return dMx, dMa
    

#%% Main script

n = 2 #dimension of state u
p = 2 #dimension of parameters
m = 2 #dimension of measurement z

  
tm = 150
dt = 0.01 #timestep
nt = int(tm/dt) #total number of timesteps
t = np.linspace(0,tm,nt+1)

X0 = np.array([0,1])

a = 0.25
b = 0.25
I = 0
tau = 10


sig = 0.1
R = sig**2 * np.eye(m)
Ri = np.linalg.inv(R)

#%% Fixed [equilibrium] points

a = 0.15
b = 0.35
I= 0

# Nullclines
# V-curve [dV/dt=0]
V = np.linspace(-2,2,100)
W1 = V - (1/3)*V**3+I

# W-curve [dV/dt=0]
W2 = (V+a)/b


p = 3*(1/b-1)
q = 3*(a/b-I)
Vs = cubic_root( (-q/2) + np.sqrt(q**2/4 + p**3/27) )  \
   + cubic_root( (-q/2) - np.sqrt(q**2/4 + p**3/27) )
Ws = (Vs+a)/b
   

plt.figure(1)
plt.plot(V,W1)
plt.plot(V,W2)
plt.plot(Vs,Ws,'o')
#plt.ylim([-2,2])
plt.xlabel(r'$V$')
plt.ylabel(r'$W$')


# Check for stability
Xs = np.array([Vs,Ws])
J = Jrhs(Xs,a,b,tau,I)[:,:2]
L = np.linalg.eig(J)[0]

if np.all(L<0):
    print('Stable')
elif np.all(L>0):
    print('Unstable')

#%% Time integration

X = np.zeros([2,nt+1])
X[:,0] = X0
for k in range(nt):
    X[:,k+1] = RK4(X[:,k],a,b,tau,I,dt)

fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,3))
ax = ax.flat

ax[0].plot(t,X[0,:], color='C0')  
ax[0].plot(t[0],X[0,0],'o', color='C1')
ax[0].set_xlabel(r'$t$', fontsize=14)
ax[0].set_ylabel(r'$V(t)$', fontsize=14)

ax[1].plot(t,X[1,:], color='C0')
ax[1].plot(t[0],X[1,0], 'o', color='C1')   
ax[1].set_xlabel(r'$t$', fontsize=14)
ax[1].set_ylabel(r'$W(t)$', fontsize=14)


ax[2].plot(X[0,:],X[1,:], color='C0')  
ax[2].plot(X[0,0],X[1,0], 'o', color='C1')  
ax[2].plot(V,W1,'--k')
ax[2].plot(V,W2,'--k')
ax[2].set_xlabel(r'$V(t)$', fontsize=14)
ax[2].set_ylabel(r'$W(t)$', fontsize=14)
#ax[2].set_ylim([-2,2])

fig.subplots_adjust(wspace=0.45)

plt.show()
#%%%
    
V = np.zeros([2,2,nt+1])

for k in range(nt):
    dMx , dMa = JRK4(X[:,k],a,b,tau,I,dt)
    dMx , dMa = JEuler(X[:,k],a,b,tau,I,dt)

    V[:,:,k+1] = dMx @ V[:,:,k] + dMa

#%%    
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(9,9))
ax = ax.flat

ax[0].plot(t,V[0,0,:]**1)    
ax[0].set_ylabel(r'$V_{11}$', fontsize=14)

ax[1].plot(t,V[0,1,:]**1)  
ax[1].set_ylabel(r'$V_{12}$', fontsize=14)
  
ax[2].plot(t,V[1,0,:]**1)  
ax[2].set_ylabel(r'$V_{21}$', fontsize=14)
  
ax[3].plot(t,V[1,1,:]**1)    
ax[3].set_ylabel(r'$V_{22}$', fontsize=14)

for i in range(4):
    ax[i].set_xlabel(r'$t$', fontsize=14)

fig.subplots_adjust(hspace=0.35, wspace=0.35)


plt.show()

#%%    
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(9,9))
ax = ax.flat

ax[0].plot(t,V[0,0,:]**2)    
ax[0].set_ylabel(r'$V_{11}^2$', fontsize=14)

ax[1].plot(t,V[0,1,:]**2)  
ax[1].set_ylabel(r'$V_{12}^2$', fontsize=14)
  
ax[2].plot(t,V[1,0,:]**2)  
ax[2].set_ylabel(r'$V_{21}^2$', fontsize=14)
  
ax[3].plot(t,V[1,1,:]**2)    
ax[3].set_ylabel(r'$V_{22}^2$', fontsize=14)

for i in range(4):
    ax[i].set_xlabel(r'$t$', fontsize=14)

fig.subplots_adjust(hspace=0.35, wspace=0.35)


plt.show()