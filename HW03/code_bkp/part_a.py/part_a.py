from model import dynamics, cost
import numpy as np

dynfun = dynamics(stochastic=False)
# dynfun = dynamics(stochastic=True) # uncomment for stochastic dynamics

costfun = cost()


T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor
TOLERANCE = 1e-12

# Riccati recursion
def Riccati(A,B,Q,R):

    # TODO implement infinite horizon riccati recursion
    
    P = np.zeros((4,4))

    for i in range(N):
        L_next = (np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A)
        P_next = Q + A.T @ P @ (A + B @ L_next)

        if np.max(np.abs(P_next - P)) < TOLERANCE:
           break

        P = P_next
    
    L = L_next
    P = P_next

    return L,P

A = dynfun.A
B = dynfun.B
Q = costfun.Q
R = costfun.R

L,P = Riccati(A,B,Q,R)

total_costs = []

for n in range(N):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):
        
        # policy 
        u = (-L @ x)
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
    
        # dynamics step
        x = dynfun.step(u)
        
    total_costs.append(sum(costs))
    
print(np.mean(total_costs))
