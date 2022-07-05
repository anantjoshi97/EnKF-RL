from constants_enkf import *
from LQSys import cLQSys

import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.size'] = 12

def state_cost(m_state):
    return np.dot(C,m_state)

def m_dynamics(m_part,m_con):
    return (np.dot(A,m_part)*STEP + np.dot(B,m_con))

def DiffRicCon(m_C,m_A,m_B,m_Q,m_Rinv):
    Cdot = -np.dot(m_A.T,m_C) - np.dot(m_C,m_A) + np.linalg.multi_dot((m_C,m_B,m_Rinv,m_B.T,m_C)) - m_Q
    return Cdot

rng = np.random.default_rng(seed=SEED0)    

a_S = np.zeros((ITER+1,DIMX,DIMX))
a_S[ITER,:,:] = ST

for k in range(ITER,0,-1):
    m_S = a_S[k,:,:]

    m_k1 = DiffRicCon(m_S,A,B,Q,RINV)

    a_S[k-1,:,:] = m_S - m_k1*STEP


a_c_particles = np.zeros((ITER+1,DIMX,N))
a_c_SigmaN = np.zeros((ITER+1,DIMX,DIMX))
a_c_SN = np.zeros((ITER+1,DIMX,DIMX))
m_c_mean = np.zeros((DIMX,ITER+1))

a_dwp0 = np.zeros((N*ITER,DIMU))
a_dwp = np.zeros((ITER,N,DIMU))



SEED0 = SEED0 + 1
rng = np.random.default_rng(seed=SEED0)

a_c_particles[0,:,0:N] = (rng.multivariate_normal(np.zeros(DIMX), OMEGA0, N)).T

m_c_mean[:,0] = np.mean(a_c_particles[0,:,0:N],axis=1)

a_c_SigmaN[0,:,:] = np.cov(a_c_particles[0,:,0:N])

a_c_SN[ITER,:,:] = np.linalg.inv(a_c_SigmaN[0,:,:])

a_dwp = rng.multivariate_normal(np.zeros(DIMU),RINV*STEP,(ITER,N))        

for k in range(0,ITER):

    a_c_particles[k+1,:,0:N] = ( a_c_particles[k,:,0:N] - m_dynamics(a_c_particles[k,:,0:N],-a_dwp[k,0:N,:].T)  
         - 0.5*STEP*np.dot((state_cost(a_c_SigmaN[k,:,:])).T,state_cost(a_c_particles[k,:,0:N] + m_c_mean[:,k:k+1])) )

    # a_c_particles[k+1,:,0:N] = ( a_c_particles[k,:,0:N] - np.dot(A,a_c_particles[k,:,0:N])*STEP 
    #     + np.dot(B,a_dwp[k,0:N,:].T) - 0.5*STEP*np.dot(a_c_SigmaN[k,:,:],np.dot(Q,(a_c_particles[k,:,0:N] + m_c_mean[:,k:k+1]))) )

    m_c_mean[:,k+1] = np.mean(a_c_particles[k+1,:,0:N],axis=1)
    a_c_SigmaN[k+1,:,:] = np.cov(a_c_particles[k+1,:,0:N])
    a_c_SN[ITER - k - 1,:,:] = np.linalg.inv(a_c_SigmaN[k+1,:,:])

            


f1, axes = plt.subplots(nrows=1, ncols=1)

ax = axes
eig1 = LA.eigvals(A)
eig2 = LA.eigvals(A  - LA.multi_dot((B,RINV,B.T,a_c_SN[0,:,:])))


ax.scatter(eig1.real,eig1.imag,marker="o",facecolor="none",edgecolor="blue",label="OL")
ax.scatter(eig2.real,eig2.imag,marker="x",c="red",label="CL")

ax.axvline(x=0, c="black", label="", ls = "--",lw = 1.2)
ax.axhline(y=0, c="black", label="", ls = "--", lw = 1.2)

ax.grid()

plt.show()


f2, axes = plt.subplots(nrows=DIMX, ncols=DIMX)
j=0
k=0
for row in axes:    
    k = 0 
    for ax in row:
        ax.axhline(y=PINF[j,k], c="red", label="ARE", ls = "solid", lw = 1.2)
        ax.plot(PLOTVEC_ERROR,np.flip(a_c_SN[0:ITER,j,k]),color='green', ls = "dashed", label="EnKF",lw=1.2)
        ax.plot(PLOTVEC_ERROR,np.flip(a_S[0:ITER,j,k]),color='black',linestyle='dashed',label="DRE", lw=1.5)
        ax.set_xticks([0,ITER/2,ITER])
        ax.set_xticklabels([0,int(T/2),T])
        subs = (j+1)*10 + (k+1)
        ax.set_title(r'$P_{%i}$' % subs)   
        k = k + 1
    j = j + 1

ax.legend(loc=0,framealpha = 1)

f2.supxlabel("Time (s)")
f2.supylabel("Value of parameter")
plt.show()
