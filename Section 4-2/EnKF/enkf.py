from constants_enkf import *
from LQSys import cLQSys

import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.size'] = 12

import time
import datetime

def state_cost(m_state):
    return np.dot(C,m_state)

def m_dynamics(m_part,m_con):
    return (np.dot(A,m_part)*STEP + np.dot(B,m_con))


def DiffRicCon(m_C,m_A,m_B,m_Q,m_Rinv):
    Cdot = -np.dot(m_A.T,m_C) - np.dot(m_C,m_A) + np.linalg.multi_dot((m_C,m_B,m_Rinv,m_B.T,m_C)) - m_Q
    return Cdot

tinit = time.process_time()

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

r_mseSN_ct = np.zeros(NSIM)
m_error_K = np.zeros((NSIM,ITER+1))
m_error_cost = np.zeros((NSIM,ITER+1))

counter = 0

m_K1 = np.zeros((DIMU,DIMX))
m_K2 = np.zeros((DIMU,DIMX))

sysINF = cLQSys(A,B,Q,R,KINF,np.zeros((DIMX,1)))
cINF = sysINF.inf_cost(SIGMA0)


for Nlog in NVEC:

    N = int(np.ceil(Nlog))
    

    for j in range(0,int(NAVG)):
        SEED0 = SEED0 + 1
        rng = np.random.default_rng(seed=SEED0)
        
        
        a_c_particles[0,:,0:N] = (rng.multivariate_normal(np.zeros(DIMX), OMEGA0, N)).T
        
        m_c_mean[:,0] = np.mean(a_c_particles[0,:,0:N],axis=1)
        
        a_c_SigmaN[0,:,:] = np.cov(a_c_particles[0,:,0:N])
        
        a_c_SN[ITER,:,:] = np.linalg.inv(a_c_SigmaN[0,:,:])
        
        a_dwp = rng.multivariate_normal(np.zeros(DIMU),RINV*STEP,(ITER,N))

        mseSN_ctj = STEP*((np.linalg.norm(a_c_SN[ITER,:,:] - a_S[ITER,:,:],'fro')/np.linalg.norm(a_S[ITER,:,:],'fro'))**2)
            
        m_K1 = np.linalg.multi_dot((-RINV,B.T,a_c_SN[ITER - k,:,:]))            
        m_K2 = KINF            
        
        

        for k in range(0,ITER):

            a_c_particles[k+1,:,0:N] = ( a_c_particles[k,:,0:N] - m_dynamics(a_c_particles[k,:,0:N],-a_dwp[k,0:N,:].T)  
                    - 0.5*STEP*np.dot((state_cost(a_c_SigmaN[k,:,:])).T,state_cost(a_c_particles[k,:,0:N] + m_c_mean[:,k:k+1])) )

            # a_c_particles[k+1,:,0:N] = ( a_c_particles[k,:,0:N] - np.dot(A,a_c_particles[k,:,0:N])*STEP 
            #     + np.dot(B,a_dwp[k,0:N,:].T) - 0.5*STEP*np.dot(a_c_SigmaN[k,:,:],np.dot(Q,(a_c_particles[k,:,0:N] + m_c_mean[:,k:k+1]))) )

            m_c_mean[:,k+1] = np.mean(a_c_particles[k+1,:,0:N],axis=1)
            a_c_SigmaN[k+1,:,:] = np.cov(a_c_particles[k+1,:,0:N])
            a_c_SN[ITER - k - 1,:,:] = np.linalg.inv(a_c_SigmaN[k+1,:,:])

            mseSN_ctj = mseSN_ctj + STEP*((np.linalg.norm(a_c_SN[ITER - k - 1,:,:] - a_S[ITER - k - 1,:,:],'fro')/np.linalg.norm(a_S[ITER - k - 1,:,:],'fro'))**2)
            
            m_K1 = np.linalg.multi_dot((-RINV,B.T,a_c_SN[ITER - k,:,:]))            
            m_K2 = KINF            
            m_error_K[counter,ITER - k - 1] = ( (m_error_K[counter,ITER - k - 1]*j + 
                (np.linalg.norm(m_K1 - m_K2,'fro')/np.linalg.norm(m_K2,'fro')))/(j+1) )            

        m_K1 = 0*np.linalg.multi_dot((-RINV,B.T,a_c_SN[ITER,:,:]))
        sys0 = cLQSys(A,B,Q,R,m_K1,np.zeros((DIMX,1)))
        c0 = sys0.inf_cost(SIGMA0)
        

        for k in range(0,ITER):
            m_K1 = np.linalg.multi_dot((-RINV,B.T,a_c_SN[ITER - k,:,:]))            
            sysk = cLQSys(A,B,Q,R,m_K1,np.zeros((DIMX,1)))
            ck = sysk.inf_cost(SIGMA0)
            m_error_cost[counter,ITER - k - 1] = (m_error_cost[counter,ITER - k - 1]*j + ((ck - cINF)/(c0 - cINF)))/(j+1)

        

        r_mseSN_ct[counter] = r_mseSN_ct[counter] + (1/NAVG)*mseSN_ctj
        

    counter = counter + 1



tfinal = time.process_time()
telapsed = tfinal - tinit
print("time elapsed = ", telapsed, "seconds")

counter = 0
for Nlog in NVEC:
    print("terminal error in gain for ", int(NVEC[counter]), " particles = ", m_error_K[counter,0])
    print("terminal error in cost for ", int(NVEC[counter]), " particles = ", m_error_cost[counter,0])
    counter = counter + 1

NVEC = np.ceil(NVEC)
q7 = np.power(NVEC,-1)

f1 = plt.figure()

ax = plt.subplot(111)
ax.semilogy(PLOTVEC,r_mseSN_ct)
ax.semilogy(PLOTVEC,q7, '--r')
ax.set_ylabel('MSE')
ax.set_xlabel('Number of particles $(N)$')
plt.xticks([PLOTVEC[0],PLOTVEC[-1]],[int(NVEC[0]),int(NVEC[-1])])
plt.show()




