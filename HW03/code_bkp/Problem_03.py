import control
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from scipy import linalg
import cvxpy as cvx
from tqdm import tqdm

# These code lines id for the part (a), I am based on lecture_12 slide 13. A is not asymptotically stable (eigenvalues greater than zero)
def compute_riccati_gain(A, B, Q, R):
    Pinf = linalg.solve_discrete_are(A,B,Q,R)
    Finf = - linalg.inv(R + np.transpose(B) @ Pinf @ B) @ (np.transpose(B) @ Pinf @ A)

    return Pinf, Finf

# letter(a)
def compute_Xf(A, B, Finf):
    Xf = []
    for x1 in np.linspace(-10, 10, int(10/0.1)):
        for x2 in np.linspace(-10, 10, int(10/0.1)):
            # (A + B*Finf)x(t) - x(t) E X and Finf*x(t) E U
            x = (A + B @ Finf) @ np.array([x1,x2])
            u = Finf @ np.array([x1,x2])
            x_norm = linalg.norm(x, 2)
            u_norm = linalg.norm(u, 2)
            if x_norm <= 5 and u_norm <= 1:
                Xf.append([x, x_norm]) 
    
    Xf1 = [item[0][0] for item in Xf]
    Xf2 = [item[0][1] for item in Xf] 

    _, axes = plt.subplots(1)
    axes.set_title('Xf')
    axes.grid(True)
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    axes.set_ylim([-10,10])
    axes.set_xlim([-10,10])
    plt.plot(Xf1,Xf2, 'bo')
    plt.savefig('problem_03_a_Xf.pdf', bbox_inches='tight')
    plt.show()

    return Xf

# letter(b)
def compute_Xf_ellipsoid(A, M):
    Xf = []
    for x1 in np.linspace(-10, 10, 20):
        for x2 in np.linspace(-10, 10, 20):
            x = np.array([x1,x2])
            x_norm = linalg.norm(x)
            if x_norm <= 5:
                k = x @ (A.T @ M) @ A @ x
                if k <= 1:
                    Xf.append(x)

    Xf = np.asarray(Xf)
    _, axes = plt.subplots(1)
    axes.set_title('Xf')
    axes.grid(True)
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    axes.set_ylim([-10,10])
    axes.set_xlim([-10,10])
    plt.plot(Xf[:,0],Xf[:,1], 'bo')
    plt.savefig('problem_03_b_Xf.pdf', bbox_inches='tight')
    plt.show()
    
    return Xf

def recent_horizon(A,B, Q, R, P, Xf, item,x0=np.array([-3.,-2.5]), N=4, uUB=1, xUB=5):
    n = Q.shape[0]
    m = R.shape[0]
    X = {}
    U = {}
    u = []
    x = []
 
    cost_terms = []
    constraints = []
    status = ""
    T = 10
    x_0 = x0
    for t in range(T):
        for k in range(N):

            X[k] = cvx.Variable(n)
            U[k] = cvx.Variable(m)

            if item == 'i' or item == 'iii':
                cost_terms.append(cvx.quad_form(X[k] - Xf, Q))
            elif item == 'ii' or item == 'iv':
                cost_terms.append(cvx.quad_form(X[k], Q))

            cost_terms.append(cvx.quad_form(U[k], R))

            constraints.append(cvx.norm(U[k]) <= uUB)
            constraints.append(cvx.norm(X[k]) <= xUB)

            if k == 0:
                constraints.append(X[k] == x_0)

            if k > 0:
                constraints.append(A @ X[k - 1] + B @ U[k - 1] == X[k])

        X[k+1] = cvx.Variable(n)
        if item == 'i':
            ##################################################
            constraints.append(A @ X[k] + B @ U[k] == X[k + 1])
            cost_terms.append(cvx.quad_form(X[k + 1] - Xf, P))
            ##################################################
        elif item == 'ii':
            ##################################################
            constraints.append(A @ X[k] + B @ U[k] == X[k + 1])
            cost_terms.append(cvx.quad_form(X[k + 1] - Xf, P))
            ##################################################
        elif item == 'iii' or item == 'iv':
            ##################################################
            constraints.append(A @ X[k] + B @ U[k] == X[k + 1])
            cost_terms.append(cvx.quad_form(X[k + 1], P))
            ##################################################

        obj = cvx.Minimize(cvx.sum(cost_terms))
        problem = cvx.Problem(obj, constraints)
        problem.solve()

        status = problem.status
        if status in ["infeasible", "unbounded"]:
            break
        else:
            for k in range(N):
                u.append(U[k].value)
                x.append(X[k].value)            
            x_0 = A @ X[0].value + B @ U[0].value

            x.append(X[k+1].value)
    return x, u

def problem_3(Xf, item):
    trajectories = []
    controls = []
    for xf in tqdm(Xf):
        x, u = recent_horizon(A,B,Q,R, P, xf, item)
        trajectories.append(x)
        controls.append(u)
    
    plot_cases(item,trajectories,controls, Xf)
    

def plot_cases(letter, traj, ctrls, Xf):
    _, axes = plt.subplots(2)
    for x in traj:
        x = np.asarray(x)
        axes[0].plot(x[:,0], x[:,1])
    axes[0].grid(True)
    axes[0].plot(Xf[:,0], Xf[:,1], 'x')
    axes[1].grid(True)
    for u in ctrls:
        axes[1].plot(u)
    plt.savefig('problem_03_c_' + letter + '_v2_.pdf', bbox_inches='tight')
    plt.show()



A = np.array([[0.95, 0.5],[0, 0.95]])
B = np.array([[0], [1]])
Q = np.eye(2)
R = np.eye(1)*10
M = np.array([[0.04, 0.],[0., 1.06]])
P_, F = compute_riccati_gain(A,B,Q,R)
P = np.eye(2)

Xf = compute_Xf_ellipsoid(A, M)
problem_3(Xf, 'iv')
#Xf = compute_Xf(A, B, F)


#print(Xf)
#problem_3(Xf_,Xf_set,'iv')
#print(Xf)
#print(P_)
# print(F)