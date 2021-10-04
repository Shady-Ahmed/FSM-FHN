# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:04:56 2020

Forward Sensitivity Method for the parameter estimation of the FitzHugh-Nagumo Model with Fixed Input/Excitation
For more details, please check our paper: 
    "Forward Sensitivity Analysis of the FitzHugh–Nagumo System: Parameter Estimation"
    By: Shady E. Ahmed, Omer San, and Sivaramakrishnan Lakshmivarahan 

For coding questions and/or suggestions, please contact Shady Ahmed at shady.ahmed@okstate.edu
"""


#%% Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

np.random.seed(seed=0)
import os


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


# Observational map
def h(X):
    Z = X
    return Z

# Jacobian of observational map
def Dh(X):
    D = np.eye(2)
    return D
    

#%% Main script

n = 2 #dimension of state u
p = 2 #dimension of parameters
m = 2 #dimension of measurement z

  
tm = 100
dt = 0.1 #timestep
nt = int(tm/dt) #total number of timesteps
t = np.linspace(0,tm,nt+1)

X0 = np.array([0.0,1.0])

a = 0.15
b = 0.35
II = (t/tm)*5
tau = 10 


sig = 0.1
R = sig**2 * np.eye(m)
Ri = np.linalg.inv(R)


#%% Twin Experiment

#Time integration
X = np.zeros([2,nt+1])
X[:,0] = X0
for k in range(nt):
    I = II[k]
    X[:,k+1] = RK4(X[:,k],a,b,tau,I,dt)

# Generate measurements
tind = np.arange(200,nt+1,200)

#tind = np.array([125,375,800,1000])

Z = X[:,tind] + np.random.normal(0,sig,X[:,tind].shape)

#%% FSM Parameter Estimation

par0 = np.array([0.2,0.2]) #some initial guess for model's parameter viscosity

c = par0

par = par0

max_iter= 50
for jj in range(max_iter):
    U = np.eye(2,2)
    V = np.zeros((2,2))

    H = np.zeros((1,2))
    e = np.zeros((1,1))
    W = np.zeros((1,1)) #weighting matrix
    k = 0
    X1 = X0
    a1 = par[0]
    b1 = par[1]
    for i in range(tind[-1]):
        I = II[i]
        dMx , dMa = JRK4(X1,a1,b1,tau,I,dt)
        V = dMx @ V + dMa

        X1 = RK4(X1,a1,b1,tau,I,dt)
        
        if i+1 == tind[k]:
            Hk = Dh(X1) @ V
            H = np.vstack((H,Hk))
            ek = (h(X1) - Z[:,k]).reshape(-1,1)
            e = np.vstack((e,ek))
            W = block_diag(W,Ri)
            k = k+1
            
    H = np.delete(H, (0), axis=0)
    e = np.delete(e, (0), axis=0)
    W = np.delete(W, (0), axis=0)
    W = np.delete(W, (0), axis=1)
    
    # solve weighted least-squares
    W1 = np.sqrt(W) 
    dc = np.linalg.lstsq(W1@H, -W1@e, rcond=None)[0]
    print(dc)
    c = c + 1*dc.ravel()#/np.linalg.norm(dc)
    par = c
    if np.linalg.norm(dc) <= 1e-6:
        print(jj)
        print(par)
        a1 = par[0]
        b1 = par[1]
        break
   

#%% Compare
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3

# create plot folder
if os.path.isdir("./Plots"):
    print('Plots folder already exists')
else: 
    print('Creating Plots folder')
    os.makedirs("./Plots")
    
    
#Time integration
X1 = np.zeros([2,nt+1])
X1[:,0] = X0
for k in range(nt):
    I = II[k]
    X1[:,k+1] = RK4(X1[:,k],a1,b1,tau,I,dt)
    

fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,3))
ax = ax.flat

ax[0].plot(t,X[0,:], color='k')  
ax[0].plot(t,X1[0,:], '--', color='C1')  
ax[0].plot(t[0],X[0,0],'o', color='k', markersize=8, markeredgewidth=3, fillstyle='none')
ax[0].plot(t[tind],Z[0,:],'*', color='C0', markersize=8, markeredgewidth=2)
ax[0].set_xlabel(r'$t$', fontsize=22)
ax[0].set_ylabel(r'$v(t)$', fontsize=18, labelpad=-15)

ax[1].plot(t,X[1,:], color='k')
ax[1].plot(t,X1[1,:], '--', color='C1')
ax[1].plot(t[0],X[1,0], 'o', color='k', markersize=8, markeredgewidth=3, fillstyle='none')   
ax[1].plot(t[tind],Z[1,:],'*', color='C0', markersize=8, markeredgewidth=2)
ax[1].set_xlabel(r'$t$', fontsize=22)
ax[1].set_ylabel(r'$w(t)$', fontsize=18, labelpad=0)

# Nullclines
# V-curve [dV/dt=0]
V = np.linspace(-3,3,100)
W1 = V - (1/3)*V**3+I

# W-curve [dV/dt=0]
W2 = (V+a)/b
ax[2].plot(V,W1,'-.', color='gray', label=r'\textbf{Nullclines}')
ax[2].plot(V,W2,'-.', color='gray')#, label=r'$W-$\textbf{Nullcline}')

ax[2].plot(X[0,:],X[1,:], color='k', label=r'\textbf{True}')  
ax[2].plot(X1[0,:],X1[1,:], '--', color='C1', label=r'\textbf{Predicted}')  

ax[2].plot(X[0,0],X[1,0], 'o', color='k', markersize=8, markeredgewidth=3, fillstyle='none')  
ax[2].plot(Z[0,:],Z[1,:],'*', color='C0', markersize=8, markeredgewidth=2, label=r'\textbf{Measurements}')

ax[2].set_xlabel(r'$v(t)$', fontsize=18)
ax[2].set_ylabel(r'$w(t)$', fontsize=18, labelpad=0)
ax[2].set_xlim([-2.5,2.5])
ax[2].set_ylim([-2,6])
#ax[2].set_ylim([np.min(W1), np.max(W1)])

fig.subplots_adjust(wspace=0.45)
ax[2].legend(loc="center", bbox_to_anchor=(-0.95,-0.4), ncol=5, fontsize=16)

fig.subplots_adjust(wspace=0.45)

#plt.savefig('./Plots/Ivary_S.png', dpi = 300, bbox_inches = 'tight')

plt.show()

#%%%
V = np.zeros([2,2,nt+1])
G = np.zeros([2,2,nt+1])

for k in range(nt):
    I = II[k]
    dMx , dMa = JRK4(X[:,k],a,b,tau,I,dt)
    #U = DM_a(u,mu) @ U
    V[:,:,k+1] = dMx @ V[:,:,k] + dMa
    G[:,:,k+1] = (V[:,:,k+1]).T @ V[:,:,k+1]
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

mpl.rc('text', usetex=True)

mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}

mpl.rc('font', **font)


fig, ax = plt.subplots(nrows=1,ncols=4, figsize=(16,3))
ax = ax.flat

ax[0].plot(t,V[0,0,:]**2,linewidth=3)    
ax[0].plot(t[tind],V[0,0,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
ax[0].set_ylabel(r'$V_{11}^2$', fontsize=20)

ax[1].plot(t,V[0,1,:]**2,linewidth=3)  
ax[1].plot(t[tind],V[0,1,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
ax[1].set_ylabel(r'$V_{12}^2$', fontsize=20)

ax[2].plot(t,V[1,0,:]**2,linewidth=3)  
ax[2].plot(t[tind],V[1,0,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
ax[2].set_ylabel(r'$V_{21}^2$', fontsize=20)


ax[3].plot(t,V[1,1,:]**2,linewidth=3)    
ax[3].plot(t[tind],V[1,1,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
ax[3].set_ylabel(r'$V_{22}^2$', fontsize=20)

for i in range(4):
    ax[i].set_xlabel(r'$t$', fontsize=26)

fig.subplots_adjust(wspace=0.5)

#plt.savefig('./Plots/Ivsens.png', dpi = 300, bbox_inches = 'tight')

plt.show()
#%%    

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(8,3))
ax = ax.flat

ax[0].plot(t,V[0,0,:]**2+V[1,0,:]**2,linewidth=2)    
ax[0].plot(t[tind],V[0,0,tind]**2+V[1,0,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
ax[0].set_ylabel(r'$V_{11}^2+V_{21}^2$', fontsize=16)

ax[1].plot(t,V[0,1,:]**2+V[1,1,:]**2,linewidth=2)  
ax[1].plot(t[tind],V[0,1,tind]**2+V[1,1,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
ax[1].set_ylabel(r'$V_{12}^2+V_{22}^2$', fontsize=16)

for i in range(2):
    ax[i].set_xlabel(r'$t$', fontsize=18)

fig.subplots_adjust(hspace=0.35, wspace=0.35)
plt.show()



#%%    
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(9,4))

ax.plot(t,V[0,0,:]*V[0,1,:]+V[1,0,:]*V[1,1,:])    
ax.plot(t[tind],V[0,0,tind]*V[0,1,tind]+V[1,0,tind]*V[1,1,tind],'o')    
ax.set_ylabel(r'$V_{11}V_{12}+V_{21}V_{22}$', fontsize=14)

ax.set_xlabel(r'$t$', fontsize=14)

fig.subplots_adjust(hspace=0.35, wspace=0.35)

plt.show()