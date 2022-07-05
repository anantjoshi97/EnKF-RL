import numpy as np
import scipy.linalg as SLA

T = 10
STEP = 0.02
ITER = int(T/STEP)
N = 2

SEED0 = 1

SMOOTH = 1e-1
GD_ITER = int(50)
ALPHA = 1e-4

MASSES = 2

if MASSES ==0 :

	DIMX = 1
	DIMU = 1
	RVSPH = np.sqrt(DIMU*DIMX)
	
	A = -0.5*np.eye(1)
	B = np.eye(1)
	Q = np.eye(1)
	R = np.eye(1)
	RINV = np.linalg.inv(R)

elif MASSES ==1 :

	DIMX = 2*MASSES
	DIMU = MASSES
	RVSPH = np.sqrt(DIMU*DIMX)
	
	A = np.array([[0,1],[-1,-1]])
	B = np.array([[0],[1]])
	Q = np.eye(DIMX)
	R = np.eye(DIMU)
	

	A = np.eye(DIMX) + A*STEP
	B = B*STEP
	Q = Q*STEP
	R = R*STEP

	RINV = np.linalg.inv(R)

else:
	DIMX = 2*MASSES
	DIMU = MASSES
	RVSPH = np.sqrt(DIMU*DIMX)

	TOEP_VEC = np.zeros(MASSES)
	TOEP_VEC[0] = 2
	TOEP_VEC[1] = -1
	TOEP = SLA.toeplitz(TOEP_VEC)
	A = np.vstack((np.hstack((np.zeros((MASSES,MASSES)),np.eye(MASSES))),np.hstack((-TOEP,-TOEP))))
	B = np.vstack((np.zeros((MASSES,MASSES)),np.eye(MASSES)))
	Q = np.eye(DIMX)
	R = np.eye(DIMU)
	RINV = np.linalg.inv(R)

	A = np.eye(DIMX) + A*STEP
	B = B*STEP
	Q = Q*STEP
	R = R*STEP

	RINV = np.linalg.inv(R)


OMEGA = np.eye(DIMX)

PINF = SLA.solve_discrete_are(A,B,Q,R)
KINF = -np.dot(np.linalg.inv(np.dot(B.T,np.dot(PINF,B)) + R),np.dot(B.T,np.dot(PINF,A)))
KINFNORM = np.linalg.norm(KINF)

# A_ARE = (A - np.eye(DIMX))/STEP
# B_ARE = B/STEP
# Q_ARE = Q/STEP
# R_ARE = R/STEP
# RINV_ARE = np.linalg.inv(R_ARE)

# PINF = SLA.solve_continuous_are(A_ARE,B_ARE,Q_ARE,R_ARE)
# KINF = -np.dot(RINV_ARE,np.dot(B_ARE.T,PINF))
# KINFNORM = np.linalg.norm(KINF)

PLOTVEC = np.arange(0,GD_ITER+1)

RVSPH = 1.0

