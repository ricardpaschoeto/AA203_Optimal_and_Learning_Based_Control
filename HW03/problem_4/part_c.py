from model import dynamics, cost
import numpy as np
import scipy
from scipy import linalg
from tqdm import tqdm
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
    L = gamma * np.linalg.pinv(R + gamma * B.T @ P @ B) @ B.T @ P @ A

    return L, P

A = np.random.rand(4, 4)
B = np.random.rand(4, 2)
Q = np.eye(4)
R = np.eye(2)
L_iter = np.zeros((N,T))

L_star = np.array([[2.51210992,-1.03523418, 3.10840684,0.11485763], [0.12845042,0.95608089,0.07756693, 1.17061578]])

Q_hat = Q
R_hat = R
A_hat = A
B_hat = B

#C = np.concatenate([A,B], axis = 1)
P_prev = np.eye(6)
C = np.concatenate([A,B], axis = 1)
C_prev = C
C_list = [C]

P_prev2 = np.eye(20)
F_prev = np.random.rand(20,1)
L,_ = Riccati(A_hat, B_hat, Q_hat, R_hat) # U1

for n in tqdm(range(N)):
    costs = []
    P_prev = np.eye(6)
    P_prev2 = np.eye(20)
    x = dynfun.reset()
    
    for t in range(T):
        
        L_iter[n][t] = linalg.norm(L_star - L)

        # compute action
        u = np.random.multivariate_normal(-L @ x, np.eye(2))

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
        C = C_list[-1] + ((P_prev @ z) @ (xp.T - z.T @ C_list[-1].T) / (1 + z.T @ P_prev @ z)).T
        P_prev = P
        C_list[-1] = C

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

        x = xp.copy()
        
    total_costs.append(sum(costs))
    # Policy Improvement
    L,_ = Riccati(A_hat, B_hat, Q_hat, R_hat) # Uk+1
    C_list.append(C)


print(np.mean(total_costs))

_, axes = plt.subplots(1)
axes.grid(True)
axes.semilogy(np.arange(0, N), total_costs)
plt.savefig('problem_04_c_cost.pdf', bbox_inches='tight')
plt.show()

_, axes = plt.subplots(1)
axes.grid(True)
axes.plot(L_iter[:,:])
plt.savefig('problem_04_c_norm.pdf', bbox_inches='tight')
plt.show()