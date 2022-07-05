import numpy as np
import scipy.linalg as SLA

T = 10.0
STEP = 0.02
ITER = int(T/STEP)
NAVG = 100.0
NSIM = 3
NVEC = np.logspace(1,3,NSIM)
N = int(np.amax(NVEC))

SEED0 = 2

MASSES = 1

if MASSES ==1 :

	DIMX = 2*MASSES
	DIMU = MASSES
	RVSPH = np.sqrt(DIMU*DIMX)
	
	A = np.array([[0,1],[-1,-1]])
	B = np.array([[0],[1]])
	C = np.sqrt(5)*np.eye(DIMX)
	Q = np.dot(C.T,C)
	R = np.eye(DIMU)
	RINV = np.linalg.inv(R)
	ST = np.eye(DIMX)
	OMEGA0 = np.linalg.inv(ST)

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
	C = np.eye(DIMX)
	Q = np.dot(C.T,C)
	R = np.eye(DIMU)
	RINV = np.linalg.inv(R)
	ST = np.eye(DIMX)
	OMEGA0 = np.linalg.inv(ST)


PINF = SLA.solve_continuous_are(A,B,Q,R)
KINF = -np.dot(RINV,np.dot(B.T,PINF))
KINFNORM = np.linalg.norm(KINF)

PLOTVEC = np.arange(1,NSIM+1)
PLOTVEC_ERROR = np.arange(1,ITER+1)

SIGMA0 = np.eye(DIMX)
