from model import dynamics, cost
import numpy as np
import scipy
from scipy import linalg


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor
TOLERANCE = 1e-12

total_costs = []
        
# Riccati recursion 
def Riccati(A,B,Q,R):
        
    # TODO implement infinite horizon riccati recursion
        
    P = np.zeros((4,4))
        
    for i in range(N):
        L_next = (np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A)
        P_next = Q + A.T @ P @ (A - B @ L_next)
   
        if np.max(np.abs(P_next - P)) < TOLERANCE:
           break
        
        P = P_next
        
    L = L_next 
    P = P_next 
        
    return L,P

def Riccati2(A,B,Q,R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    L = np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A

    return L, P

A = np.random.rand(4, 4)
B = np.random.rand(4, 2)
Q = np.eye(4)
R = np.eye(2)

for n in range(N):
    costs = []

    if n == 0:

       Q_hat = Q
       R_hat = R
       A_hat = A
       B_hat = B

       C = np.concatenate([A,B], axis = 1)
       P_prev = np.eye(6)
       C_prev = np.concatenate([A,B], axis = 1)

       P_prev2 = np.eye(20)
       F_prev = np.random.rand(20,1)

    
    x = dynfun.reset()

    for t in range(T):
        
        # TODO compute policy
        
        L,P_Ricatti = Riccati(A_hat, B_hat, Q_hat, R_hat)
        print("L = " + str(L))
        
        # compute action
        u = (-L @ x)

        print("u = " + str(u))

        z = np.concatenate((x.T, u.T))
        z = z.T
        z = z.reshape((6,1))
        print("z = " + str(z))
        print("z shape = " + str(z.shape))

        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
        
        # implement recursive least squares update

        P = P_prev - (P_prev @ z @ z.T @ P_prev) / (1 + z.T @ P_prev @ z)
        C = C_prev + ((P_prev @ z) @ (xp.T - z.T @ C_prev.T) / (1 + z.T @ P_prev @ z)).T
        P_prev = P
        C_prev = C

        print("P = " + str(P))
        print("C = " + str(C))

        A_hat = C[0:4, 0:4]
        B_hat = C[0:4, 4:6]

        u = u.reshape((2,1))

        x2 = np.outer(x, x)
        u2 = np.outer(u, u)

        print("x2 shape = " + str(x2.shape))
        print("x2 = " + str(x2))
        print("u2 shape = " + str(u2.shape))

        z2 = np.concatenate((x2.flatten(), u2.flatten()))
        z2 = z2.reshape(z2.shape[0],1)
        print("z2 shape = " + str(z2.shape))

        P2 = P_prev2 - (P_prev2 @ z2 @ z2.T @ P_prev2) / (1 + z2.T @ P_prev2 @ z2)
        F = F_prev + (P_prev2 @ z2) @ (c - z2.T @ F_prev) / (1 + z2.T @ P_prev2 @ z2)
        P_prev2 = P2
        F_prev = F
       
        print("F shape = " + str(F.shape))

        Q_hat = F[0:16].reshape((4,4))
        R_hat = F[16:20].reshape((2,2))

        #Q_hat = 0.5 * (Q_hat + Q_hat.T)
        #R_hat = 0.5 * (R_hat + R_hat.T)

        print("Q_hat shape = " + str(Q_hat.shape))
        print("R_hat shape = " + str(R_hat.shape))

        #print("F = " + str(F))
        #print("Q_hat = " + str(Q_hat))
        #print("R_hat = " + str(R_hat))

        x = xp.copy()
        
    total_costs.append(sum(costs))
    print(np.mean(total_costs))
