from constants_f18 import ITER, DIMX, STEP

import numpy as np
import numpy.linalg as LA
import scipy.linalg as SLA


class dLQSys(object):
    """docstring for LQR"""
    def __init__(self, A, B, Q, R, K, X0):     
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = K
        self.x0 = X0
        self.Phi = (A + np.dot(B,K))
        self.Ac = (A + np.dot(B,K))
        self.PINF = SLA.solve_discrete_are(self.A,self.B,self.Q,self.R)

    def CSigma(self):
        m_Sigma = np.zeros((DIMX,DIMX))
        v_x = self.x0
        v_u = np.dot(self.K,v_x)        
        c = 0.0
        for i in range(0,ITER):
            m_Sigma = (m_Sigma*i + np.dot(v_x,v_x.T))/(i+1)
            c = c + (np.dot(v_x.T,np.dot(self.Q,v_x)) + np.dot(v_u.T,np.dot(self.R,v_u)))
            #print("cost1=",np.dot(v_x.T,np.dot(Q,v_x))*STEP," cost2=",np.dot(v_u.T,np.dot(R,v_u))*STEP)
            v_x = np.dot(self.Phi,v_x)
            v_u = np.dot(self.K,v_x)
        #c = c + np.dot(v_x.T,np.dot(Q,v_x))

        return c.item(0), m_Sigma

    def cost_lyap(self):
        PK = self.getPK()
        xinit = self.x0

        return np.dot(xinit.T,np.dot(PK,xinit))

    def opt_cost(self,OMEGA):
        m_ic = np.trace(np.dot(OMEGA,PINF))
        return m_ic.item(0)    

    def inf_cost(self,OMEGA): #check
        m_dM = self.Q + np.dot(self.K.T,np.dot(self.R,self.K))
        m_M = m_dM

        #print(m_dM)

        for i in range(1,ITER):            

            m_dM = np.dot(self.Phi.T,np.dot(m_dM,self.Phi))
            m_M = m_M + m_dM
            #print(m_M)

        m_dM = np.dot(self.Phi.T,np.dot(m_dM,self.Phi))
        m_M = m_M + m_dM

        m_c = np.trace(np.dot(OMEGA,m_M))

        return m_c.item(0)

    def getPK(self):
        PK = SLA.solve_discrete_lyapunov((self.Phi).T, self.Q + np.dot(self.K.T,np.dot(self.R,self.K)))
        return PK

    def ana_grad(self,OMEGA):
        SK = np.zeros((DIMX,DIMX))
        #dS = np.zeros((DIMX,DIMX))
        #EK = np.zeros((DIMU,DIMX))

        dS = LA.multi_dot((self.Phi.T,OMEGA,self.Phi))

        for k in range(0,ITER):

            SK = SK + dS
            dS = LA.multi_dot((self.Phi,dS,self.Phi.T))

        PK = self.getPK()

        EK = np.dot(self.R + LA.multi_dot((self.B.T,PK,self.B)),self.K)
        EK = EK + LA.multi_dot((self.B.T,PK,self.A))
        #print(EK)

        grad = 2*np.dot(EK,SK)

        return grad

class cLQSys(object):
    """docstring for LQR"""
    def __init__(self, A, B, Q, R, K, X0):     
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = K
        self.x0 = X0
        self.Phi = SLA.expm((A + np.dot(B,K))*STEP)
        self.Ac = (A + np.dot(B,K))
        self.PINF = SLA.solve_continuous_are(self.A,self.B,self.Q,self.R)

    def cost(self):
        v_x = self.x0
        v_u = np.dot(self.K,v_x)        
        c = 0.0
        for i in range(0,ITER):
            c = c + (np.dot(v_x.T,np.dot(self.Q,v_x)) + np.dot(v_u.T,np.dot(self.R,v_u)))*STEP
            #print("cost1=",np.dot(v_x.T,np.dot(Q,v_x))*STEP," cost2=",np.dot(v_u.T,np.dot(R,v_u))*STEP)
            v_x = np.dot(self.Phi,v_x)
            v_u = np.dot(self.K,v_x)
        #c = c + np.dot(v_x.T,np.dot(Q,v_x))

        return c.item(0)

    def opt_cost(self,OMEGA):
        m_ic = np.trace(np.dot(OMEGA,PINF))
        return m_ic.item(0)

    def cost_trap(self):
        v_x = self.x0
        v_u = np.dot(self.K,v_x)
        c = 0.0
        c = c + (np.dot(v_x.T,np.dot(self.Q,v_x)) + np.dot(v_u.T,np.dot(self.R,v_u)))*STEP*0.5

        for i in range(0,ITER):
            k1 = np.dot(self.Ac,v_x)
            #print(k1)
            #print(self.Ac)
            #print(v_x)
            k2 = np.dot(self.Ac,v_x + 0.5*STEP*k1)
            k3 = np.dot(self.Ac,v_x + 0.5*STEP*k2)    
            k4 = np.dot(self.Ac,v_x + STEP*k3)

            v_x = v_x + (STEP*(k1 + k2 + k3 + k4))/6.0
            v_u = -np.dot(self.K,v_x)
            c = c + (np.dot(v_x.T,np.dot(self.Q,v_x)) + np.dot(v_u.T,np.dot(self.R,v_u)))*STEP

        v_x = np.dot(self.Phi,v_x)
        v_u = np.dot(self.K,v_x)
        c = c + (np.dot(v_x.T,np.dot(self.Q,v_x)) + np.dot(v_u.T,np.dot(self.R,v_u)))*STEP*0.5
        #c = c + np.dot(v_x.T,np.dot(Q,v_x))

        return c.item(0)

    def inf_cost(self,OMEGA): #check
        m_dM = (self.Q + np.dot(self.K.T,np.dot(self.R,self.K)))*STEP
        m_M = m_dM/2.0        
        #print(m_dM)

        for i in range(1,ITER):            

            m_dM = np.dot(self.Phi.T,np.dot(m_dM,self.Phi))
            m_M = m_M + m_dM
            #print(m_M)

        m_dM = np.dot(self.Phi.T,np.dot(m_dM,self.Phi))*STEP
        m_M = m_M + m_dM*STEP/2.0

        m_c = np.trace(np.dot(OMEGA,m_M))

        return m_c.item(0)

    def getPK(self):
        PK = SLA.solve_continuous_lyapunov((self.Ac).T,-(self.Q + np.dot(self.K.T,np.dot(self.R,self.K))))
        return PK

    def cost_lyap(self):
        PK = self.getPK()
        xinit = self.x0

        return np.dot(xinit.T,np.dot(PK,xinit))
