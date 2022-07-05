from constants_f18 import *
from LQSys import dLQSys


import numpy as np
import numpy.linalg as LA
import scipy.linalg as SLA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
import time
import datetime


def RVSphere(rng):
    # https://dl.acm.org/doi/10.1145/377939.377946
    v_v = rng.multivariate_normal(np.zeros(DIMX*DIMU), np.eye(DIMX*DIMU))
    return v_v/(LA.norm(v_v))

def RVX(rng):
    r_x0 = rng.multivariate_normal(np.zeros(DIMX), OMEGA)
    v_x0 = np.reshape(r_x0,(DIMX,1))
    return v_x0

tinit = time.process_time()


rng = np.random.default_rng(seed=SEED0)    
K = np.zeros((GD_ITER+1,DIMU,DIMX))
v_error_cost = np.zeros(GD_ITER+1)
v_error_K = np.zeros(GD_ITER+1)
v_error_grad = np.zeros(GD_ITER)



v_x0 = RVX(rng)
sys0 = dLQSys(A,B,Q,R,K[0,:,:],v_x0)
c0 = sys0.inf_cost(OMEGA)

# v_x0 = RVX(rng)
sysINF = dLQSys(A,B,Q,R,KINF,v_x0)
cINF = sysINF.inf_cost(OMEGA)

v_error_cost[0] = (c0 - cINF)/(c0 - cINF)
v_error_K[0] = np.linalg.norm(K[0,:,:] - KINF)/KINFNORM



for k in range(0,GD_ITER):
    
    m_grad = np.zeros((DIMU,DIMX))
    m_Sigma = np.zeros((DIMX,DIMX))
    for i in range(0,N):
        
        v_dK = RVSPH*RVSphere(rng)
        m_dK = np.reshape(v_dK,(DIMU,DIMX))        
        v_x0 = RVX(rng)
        
        sys1 = dLQSys(A,B,Q,R,K[k,:,:]+SMOOTH*m_dK,v_x0) 
        c1, m_Sigma1 = sys1.CSigma()
        
        sys2 = dLQSys(A,B,Q,R,K[k,:,:]-SMOOTH*m_dK,v_x0)
        c2, m_Sigma2 = sys2.CSigma()
        
        m_grad = (m_grad*(i) + ((c1-c2)*m_dK)/SMOOTH)/(i+1)
        m_Sigma = (m_Sigma*i + m_Sigma1 + m_Sigma2)/(i+1)
    
    m_grad = (m_grad*DIMX)/(2.0*SMOOTH)
    m_Sigma = m_Sigma/2.0    

    K[k+1,:,:] = K[k,:,:] - ALPHA*m_grad
    
    v_x0 = RVX(rng)
    sysk = dLQSys(A,B,Q,R,K[k+1,:,:],v_x0)
    ck = sysk.inf_cost(OMEGA)    
    v_error_cost[k+1] = (ck - cINF)/(c0 - cINF)
    v_error_K[k+1] = np.linalg.norm(K[k+1,:,:] - KINF)/KINFNORM
    

tfinal = time.process_time()
telapsed = tfinal - tinit
print("time elapsed = ", telapsed, "seconds")
print("terminal error in cost = ", v_error_cost[-1])
print("terminal error in gain = ",v_error_K[-1])

f1 = plt.figure()
ax = plt.subplot(211)
ax.semilogy(PLOTVEC,np.absolute(v_error_cost))
ax.set_ylabel('Error in cost')
ax = plt.subplot(212)
ax.semilogy(PLOTVEC,v_error_K)
ax.set_ylabel('Relative error in gain')
ax.set_xlabel('Number of iterations (k)')
f1.suptitle('Spring Mass Damper')
plt.show()



