from model import dynamics, cost
import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt


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
        L_next = gamma * (np.linalg.pinv(R + gamma * B.T @ P @ B) @ B.T @ P @ A)
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

L_star = np.array([[2.51210992,-1.03523418, 3.10840684,0.11485763], [0.12845042,0.95608089,0.07756693, 1.17061578]])

norm_diffs = []

for n in range(N):
    costs = []

    if n == 0:

       Q_hat = Q
       R_hat = R
       A_hat = A
       B_hat = B

       C = np.concatenate([A,B], axis = 1)
       P_prev = 200*np.eye(6)
       C_prev = np.concatenate([A,B], axis = 1)

       P_prev2 = 200*np.eye(20)
       F_prev = np.random.rand(20,1)

    
    x = dynfun.reset()

    if n > 1:
       #print("L - L_star = " + str(np.linalg.norm(L_star - L, 2)))

       norm_diffs.append(np.linalg.norm(L - L_star, 2))

    for t in range(T):
        
        # TODO compute policy
        
        L,P_Ricatti = Riccati(A_hat, B_hat, Q_hat, R_hat)
        
        # compute action

        cov = np.eye(2)
        u = np.random.multivariate_normal(-L @ x, cov, size=1)
        u = u.reshape((2,))

        z = np.concatenate((x.T, u.T))
        z = z.T
        z = z.reshape((6,1))

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

        A_hat = C[0:4, 0:4]
        B_hat = C[0:4, 4:6]

        u = u.reshape((2,1))

        x2 = np.outer(x, x)
        u2 = np.outer(u, u)

        z2 = np.concatenate((x2.flatten(), u2.flatten()))
        z2 = z2.reshape(z2.shape[0],1)

        P2 = P_prev2 - (P_prev2 @ z2 @ z2.T @ P_prev2) / (1 + z2.T @ P_prev2 @ z2)
        F = F_prev + (P_prev2 @ z2) @ (c - z2.T @ F_prev) / (1 + z2.T @ P_prev2 @ z2)
        P_prev2 = P2
        F_prev = F
       
        Q_hat = F[0:16].reshape((4,4))
        R_hat = F[16:20].reshape((2,2))

        Q_hat = 0.5 * (Q_hat + Q_hat.T)
        R_hat = 0.5 * (R_hat + R_hat.T)

        x = xp.copy()
        
    total_costs.append(sum(costs))
    print(np.mean(total_costs))

plt.xlabel('Episode Number')
plt.ylabel('L Norm Difference')

plt.plot(norm_diffs)
plt.show()

plt.xlabel('Episode Number')
plt.ylabel('Cost')

plt.plot(total_costs)
plt.show()
