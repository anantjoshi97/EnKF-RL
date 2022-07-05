from constants_ivp import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

def state_cost(m_state):
    return np.dot(C,m_state)

def r_B(theta):
	c = np.cos(theta)
	s = np.sin(theta)
	r = 1/(MASS_CART + MASS_BALL*(s**2))

	return np.array([0,r,0,-(r*c/LENGTH_ROD)])

def v_dynamics(s1,s2,s3,s4,con):
	s = np.sin(s3)
	c = np.cos(s3)
	r = 1/(MASS_CART + MASS_BALL*(s**2))

	sdot1 = s2
	sdot3 = s4
	sdot2 = r*(con + MASS_BALL*s*(LENGTH_ROD*(s2**2) + GRAV*c))
	sdot4 = (r/LENGTH_ROD)*(-con*c - MASS_BALL*LENGTH_ROD*(s2**2)*c*s - (MASS_BALL + MASS_CART)*GRAV*s)

	return (sdot1,sdot2,sdot3,sdot4)

def m_DiffRicCon(m_C,m_A,m_B,m_Q,m_Rinv):
    Cdot = -np.dot(m_A.T,m_C) - np.dot(m_C,m_A) + np.linalg.multi_dot((m_C,m_B,m_Rinv,m_B.T,m_C)) - m_Q
    return Cdot


def v_dyn(r_state,con):
	dist = r_state[0]
	vel = r_state[1]
	theta = r_state[2]
	omega = r_state[3]
	# s = np.sin(theta)
	# c = np.cos(theta)
	# r = 1/(MASS_CART + MASS_BALL*(s**2))

	# sdot1 = vel
	# sdot3 = omega
	# sdot2 = r*(con + MASS_BALL*s*(LENGTH_ROD*(vel**2) + GRAV*c))
	# sdot4 = (r/LENGTH_ROD)*(-con*c - MASS_BALL*LENGTH_ROD*(vel**2)*c*s - (MASS_BALL + MASS_CART)*GRAV*s)

	sdot1,sdot2,sdot3,sdot4 = v_dynamics(dist,vel,theta,omega,con)
	
	return np.array([[sdot1],[sdot2],[sdot3],[sdot4]])
	#return np.array([sdot1,sdot2,sdot3,sdot4])

def v_dynfilter(v_inp):
	r_inp = np.reshape(v_inp,(DIMX+DIMU,))
	dist = r_inp[0]
	vel = r_inp[1]
	theta = r_inp[2]
	omega = r_inp[3]
	con = r_inp[4]
	# s = np.sin(theta)
	# c = np.cos(theta)	

	r_b = r_B(theta)

	# sdot1 = vel
	# sdot3 = omega
	# sdot2 = r*(MASS_BALL*s*(LENGTH_ROD*(vel**2) + GRAV*c))
	# sdot4 = (r/LENGTH_ROD)*(- MASS_BALL*LENGTH_ROD*(vel**2)*c*s - (MASS_BALL + MASS_CART)*GRAV*s)

	sdot1,sdot2,sdot3,sdot4 = v_dynamics(dist,vel,theta,omega,0.0)

	ds1 = -sdot1*STEP + r_b[0]*con
	ds2 = -sdot2*STEP + r_b[1]*con
	ds3 = -sdot3*STEP + r_b[2]*con
	ds4 = -sdot4*STEP + r_b[3]*con

	#return np.array([[sdot1],[sdot2],[sdot3],[sdot4]])
	return np.array([ds1,ds2,ds3,ds4])

a_S = np.zeros((ITER+1,DIMX,DIMX))
a_S[ITER,:,:] = ST

for k in range(ITER,0,-1):
    m_S = a_S[k,:,:]

    m_k1 = m_DiffRicCon(m_S,A,B,Q,RINV)
    m_k2 = m_DiffRicCon(m_S - 0.5*STEP*m_k1,A,B,Q,RINV)
    m_k3 = m_DiffRicCon(m_S - 0.5*STEP*m_k2,A,B,Q,RINV)
    m_k4 = m_DiffRicCon(m_S - STEP*m_k3,A,B,Q,RINV)

    a_S[k-1,:,:] = m_S - (1/6.0)*(m_k1 + m_k2 + m_k3 + m_k4)*STEP

rng = np.random.default_rng(seed=SEED0)    

a_c_particles = np.zeros((ITER+1,DIMX,N))
a_c_SigmaN = np.zeros((ITER+1,DIMX,DIMX))
a_c_SigmaNinv = np.zeros((ITER+1,DIMX,DIMX))
m_c_mean = np.zeros((DIMX,ITER+1))

#a_dwp0 = np.zeros((N*ITER,DIMU))
a_dwp = np.zeros((ITER,N,DIMU))

a_c_particles[0,:,:] = (rng.multivariate_normal(np.zeros(DIMX), OMEGA0, N)).T
m_c_mean[:,0] = np.mean(a_c_particles[0,:,0:N],axis=1)
a_c_SigmaN[0,:,:] = np.cov(a_c_particles[0,:,:])
a_c_SigmaNinv[ITER,:,:] = np.linalg.inv(a_c_SigmaN[0,:,:])   
a_dwp = rng.multivariate_normal(np.zeros(DIMU),RINV*STEP,(ITER,N))

# non linear dual enkf propagation

for k in range(0,ITER):

	m_ds = np.apply_along_axis(v_dynfilter,0,np.vstack((a_c_particles[k,:,0:N] + STATE_EQ,a_dwp[k,0:N,:].T)))
	a_c_particles[k+1,:,0:N] = ( a_c_particles[k,:,0:N] + m_ds 
		- 0.5*STEP*np.dot((state_cost(a_c_SigmaN[k,:,:])).T,state_cost(a_c_particles[k,:,0:N] + m_c_mean[:,k:k+1])) )
        #- 0.5*STEP*np.dot(a_c_SigmaN[k,:,:],np.dot(Q,(a_c_particles[k,:,0:N] + m_c_mean[:,k:k+1]))) )

	m_c_mean[:,k+1] = np.mean(a_c_particles[k+1,:,0:N],axis=1)
	a_c_SigmaN[k+1,:,:] = np.cov(a_c_particles[k+1,:,0:N])
	a_c_SigmaNinv[ITER - k - 1,:,:] = np.linalg.inv(a_c_SigmaN[k+1,:,:])

	# if np.mod(k,ITER/20)==0:
	# 	print(k*100/ITER)

# linearised filter propagation
# for k in range(0,ITER):

#     a_c_particles[k+1,:,0:N] = ( a_c_particles[k,:,0:N] - np.dot(A,a_c_particles[k,:,0:N])*STEP 
#         + np.dot(B,a_dwp[k,0:N,:].T) - 0.5*STEP*np.dot(a_c_SigmaN[k,:,:],np.dot(Q,(a_c_particles[k,:,0:N] + m_c_mean[:,k:k+1]))) )

#     m_c_mean[:,k+1] = np.mean(a_c_particles[k+1,:,0:N],axis=1)
#     a_c_SigmaN[k+1,:,:] = np.cov(a_c_particles[k+1,:,0:N])
#     a_c_SigmaNinv[ITER - k - 1,:,:] = np.linalg.inv(a_c_SigmaN[k+1,:,:])

#     if np.mod(k,ITER/20)==0:
#     	print(k*20/ITER)


m_state = np.zeros((DIMX,ITER+1))
m_control = np.zeros((DIMU,ITER))

m_state[:,0] = np.array([dist_init,vel_init,theta_init,omega_init])


m_state_ana = np.zeros((DIMX,ITER+1))
m_control_ana = np.zeros((DIMU,ITER))
m_state_ana[:,0] = m_state[:,0]

v_delta_state = np.zeros([DIMX,1])
v_delta_state_ana = v_delta_state
G = np.zeros((DIMU,DIMX))

for k in range(0,ITER):

	v_delta_state = m_state[:,k:k+1] - STATE_EQ
	G = -np.linalg.multi_dot((RINV,B.T,a_c_SigmaNinv[k,:,:]))
	m_control[:,k:k+1] = np.dot(G,v_delta_state)

	v_delta_state_ana = m_state_ana[:,k:k+1] - STATE_EQ
	G = -np.linalg.multi_dot((RINV,B.T,a_S[k,:,:]))
	m_control_ana[:,k:k+1] = np.dot(G,v_delta_state_ana)

	m_state[:,k+1:k+2] = m_state[:,k:k+1] + v_dyn(m_state[:,k],m_control[0,k])*STEP
	m_state_ana[:,k+1:k+2] = m_state_ana[:,k:k+1] + v_dyn(m_state_ana[:,k],m_control_ana[0,k])*STEP

	# if np.mod(k,ITER/20)==0:
	# 	print(k*100/ITER)
	# 	print(k)


f1 = plt.figure()
ax = plt.subplot(211)
ax.plot(PLOTVEC,m_state[0,:], label = "EnKF")
ax.plot(PLOTVEC,m_state_ana[0,:], '--r', label = "DRE")
ax.set_ylabel('$x$')
ax = plt.subplot(212)
ax.plot(PLOTVEC,m_state[2,:], label = "EnKF")
ax.plot(PLOTVEC,m_state_ana[2,:], '--r', label = "DRE")
ax.set_ylabel('$\\theta$')
ax.set_xlabel('Time (t)')
ax.legend()
plt.show()
f1.suptitle('Cart-Pole')





