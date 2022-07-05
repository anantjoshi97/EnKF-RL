import numpy as np
import scipy.linalg as SLA

PI = np.pi
N = 500


T = 20
STEP = 0.0002

MASS_BALL = 0.08
LENGTH_ROD = 0.70/1.0
GRAV = 9.81
MASS_CART = 1.0
MOM_INERTIA = (MASS_BALL)*(LENGTH_ROD**2);

ITER = int(T/STEP)

SEED0 = 2

DIMX = 4
DIMU = 1

A = np.zeros((4,4)) # states are (x,v,theta,omega)
A[0,1] = 1.0
A[2,3] = 1.0
A[1,2] = MASS_BALL*GRAV/MASS_CART
A[3,2] = ((MASS_BALL + MASS_CART)*GRAV)/(MASS_CART*LENGTH_ROD)
B = np.array([[0],[1.0/(MASS_CART)],[0],[1.0/(MASS_CART*LENGTH_ROD)]])

STATE_EQ = np.array([[0],[0],[PI],[0]])

C = np.diag(np.array([np.sqrt(10),1,np.sqrt(10),1]))
Q = np.dot(C.T,C)
R = 10*np.eye(DIMU);
ST = np.eye(DIMX);
RINV = np.linalg.inv(R);
STINV = np.linalg.inv(ST);

x0 = np.zeros((DIMX,1));
SIGMA0 = 0.1*np.eye(DIMX);
OMEGA0 = STINV;

# PINF = SLA.solve_discrete_are(A,B,Q,R)
# KINF = np.dot(np.linalg.inv(np.dot(B.T,np.dot(PINF,B)) + R),np.dot(B.T,np.dot(PINF,A)))
# KINFNORM = np.linalg.norm(KINF)

PLOTVEC = np.arange(1,ITER+2)

theta_init =(1.25)*PI;
dist_init = -0.1;
vel_init = 0*0.001;
omega_init = 0*0.001;

