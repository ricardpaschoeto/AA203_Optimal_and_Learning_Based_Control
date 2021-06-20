from cvxpy.reductions.solvers import solver
from cvxpy.settings import OSQP
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm

def recent_horizon(A,B, Q, R, P, x0, N, uLB=-1, uUB=1, xLB0=-10, xUB0=10, xLB1=-10, xUB1=10):
    n = Q.shape[0]
    m = R.shape[0]
    X = {}
    U = {}
    u = []
    x = []
 
    cost_terms = []
    constraints = []
    list_of_costs = []
    status = ""
    T = 20
    x_0 = x0
    for t in range(T):
        # Optimization problem ################################
        for k in range(N):

            X[k] = cvx.Variable(n)
            U[k] = cvx.Variable(m)

            cost_terms.append(0.5*cvx.quad_form(X[k], Q))
            cost_terms.append(0.5*cvx.quad_form(U[k], R))

            constraints.append(U[k] <= uUB)
            constraints.append(U[k] >= uLB)

            constraints.append(X[k][0] <= xUB0)
            constraints.append(X[k][0] >= xLB0)

            constraints.append(X[k][1] <= xUB1)
            constraints.append(X[k][1] >= xLB1)

            if k == 0:
                constraints.append(X[k] == x_0)

            if k > 0:
                constraints.append(A @ X[k - 1] + B @ U[k - 1] == X[k])

        X[k+1] = cvx.Variable(n)
        constraints.append(A @ X[k] + B @ U[k] == X[k + 1])

        cost_terms.append(0.5*cvx.quad_form(X[k + 1], P))

        obj = cvx.Minimize(cvx.sum(cost_terms))
        problem = cvx.Problem(obj, constraints)
        problem.solve()
        ##########################################################
        status = problem.status
        if status in ["infeasible", "unbounded"]:
            break
        else:
            for k in range(N):
                u.append(list(U[k].value))
                x.append(list(X[k].value))
            x.append(list(X[k+1].value))

            # Next initial State for Iteration
            x_0 = A @ X[0].value + B @ U[0].value
            # Cost values to plot
            list_of_costs.append(cvx.sum(cost_terms))
            
    return x, list_of_costs, status

def compute_riccati(A, B, Q, R):
    return linalg.solve_discrete_are(A,B,Q,R)

def discrete_ss(dt=0.33, xc=10):
    space = []
    l = int(xc/dt)
    for x1 in np.linspace(-xc, xc, int(xc/dt)):
        for x2 in np.linspace(-xc, xc, int(xc/dt)):
            x = np.array([x1,x2])
            space.append(x)
    space.append(np.array([0.,0.]))

    space = np.asarray(space)
    plt.plot(space[:,0],space[:,1], 'bo')
    plt.savefig('grid.pdf', bbox_inches='tight')
    plt.show()

    return space

def get_attraction_set(grid,A,B,Q,R,P,R2,N, uLB, uUB, xLB0, xUB0, xLB1, xUB1, letter):
    result = []
    x0_set = []
    for x0 in tqdm(grid):
        x, _, status = recent_horizon(A,B,Q,R,P,x0,N, uLB, uUB, xLB0, xUB0, xLB1, xUB1)
        if letter != 'b':
            if status not in ["infeasible", "unbounded"]:
                if R2:
                    result.append(x)
                    x0_set.append(x0)
                elif linalg.norm(x[-1]) <= 1e-4:
                    result.append(x)
                    x0_set.append(x0)
        else:
            result.append(x)
            x0_set.append(x0)

    return np.asarray(result), np.asarray(x0_set)

# This for question (h)
def iterate_N(A,B,Q,R,P, uLB, uUB, xLB0, xUB0, xLB1, xUB1):
    N = range(2, 7)
    x0 = np.array([-2.4137931,  1.03448276])
    trajectory = []
    list_costs = []
    for n in tqdm(N):
        x, costs, _ = recent_horizon(A,B,Q,R,P,x0,n, uLB, uUB, xLB0, xUB0, xLB1, xUB1)
        if linalg.norm(x[-1]) <= 1e-4:
            trajectory.append(x)
            list_costs.append(costs)

    plot_traj(np.asarray(trajectory), len(N))
    plot_cost(list_costs)

def plot_traj(trajectory, N):

    _, axes = plt.subplots(N)
    axes[0].set_title('Receding Horizon Problem - Trajectory')
    axes[N-1].set_xlabel('x1')
    axes[N-1].set_ylabel('x2')
    for ii, data in enumerate(trajectory):
        axes[ii].grid(True)
        data = np.asarray(data)
        axes[ii].plot(data[:,0],data[:,1])

    plt.savefig('problem_02_h_Trajectory.pdf', bbox_inches='tight')
    plt.show()

def plot_cost(costs):
    _, axes = plt.subplots(1)
    axes.set_title('Receding Horizon Problem - Cost')
    axes.set_xlabel('time')
    axes.set_ylabel('Cost')
    axes.grid(True)
    N = 2
    labels = []
    for cost in costs:
        axes.plot([c.value for c in cost])
        labels.append(str(N))
        N = N + 1
    axes.legend(labels)
    plt.savefig('problem_02_h_costs.pdf', bbox_inches='tight')
    plt.show()

def plot_ss(result, x_set, letter):
    _, axes = plt.subplots(1)
    axes.set_title('Receding Horizon Problem')
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    axes.grid(True)
    for data in result:
        data = np.asarray(data)
        axes.plot(data[:,0], data[:,1])    
    axes.plot(x_set[:,0], x_set[:,1], 'ko')
    plt.savefig('problem_02_' + letter + '.pdf', bbox_inches='tight')
    plt.show()

# This is for questions (b - h)
def problem_2(letter, A,B,Q,R, P):
    P_ = compute_riccati(A,B,Q,R)
    if letter == 'b':
        grid_ = [np.array([-4.5, 2]), np.array([-4.5, 3])]
        result, x_set = get_attraction_set(grid_,A,B,Q,R,P,True,3, uLB=-0.5,uUB=0.5,xLB0=-5,xUB0=5,xLB1=-5,xUB1=5, letter='b')
    elif letter == 'c':
        #grid_ = [np.array([0, 0]), np.array([1., 1.]), np.array([-1., 1.]), np.array([1., -1.]), np.array([-1., -1.])]        
        result, x_set = get_attraction_set(grid,A,B,Q,R,P_,False,2, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10, letter='')
        print(x_set)
    elif letter == 'd':
        result, x_set = get_attraction_set(grid,A,B,Q,R,P_,False,6, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10, letter='')
        print(x_set)
    elif letter == 'e':
        result, x_set = get_attraction_set(grid,A,B,Q,R,P_,True,2, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10, letter='')
    elif letter == 'f':
        result, x_set = get_attraction_set(grid,A,B,Q,R,P_,True,6, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10, letter='')
    elif letter == 'h':
        iterate_N(A,B,Q,R,P_, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10)
        return
    else:
        print('Choose valid: b - f')
        return

    plot_ss(result, x_set, letter)

A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
Q = np.eye(2)
R = 10*np.eye(1)
P = np.eye(2)
grid = discrete_ss()

problem_2('h', A,B,Q,R, P)

