import numpy as np
import scipy.linalg as SLA

T = 10
STEP = 0.02
ITER = int(T/STEP)
N = 1000

SEED0 = 2 #for d=10, and SEED0 = 3 for d=2

DIMX = 2
	
DIMU = 1

rng = np.random.default_rng(seed=SEED0 + 6)    	


if DIMX == 1:
	#DIMX = 1
	A = np.zeros((DIMX,DIMX))
	B = np.zeros((DIMX,DIMU))
	A = rng.multivariate_normal(np.zeros(DIMX), np.eye(DIMX), 1)
	B[0,0] = 1.0
else:
	#DIMX = 2*MASSES
	A = np.zeros((DIMX,DIMX))
	B = np.zeros((DIMX,DIMU))
	A[0:DIMX-1,1:DIMX] = np.eye(DIMX - 1)
	A[DIMX - 1:DIMX,0:DIMX] = rng.multivariate_normal(np.zeros(DIMX), np.eye(DIMX), 1)
	B[DIMX-1,0] = 1.0

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

