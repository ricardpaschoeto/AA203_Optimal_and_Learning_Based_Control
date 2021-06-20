import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from scipy import linalg

# Have a look in my comments to try to understand the points, I am based on lecture_11 slide 13
def recent_horizon(A,B, Q, R, P, x0, N, R2, uLB=-1, uUB=1, xLB0=-10, xUB0=10, xLB1=-10, xUB1=10):
    n = Q.shape[0]
    m = R.shape[0]
    X = {}
    U = {}
    u = []
    x = []
    
    cost_terms = []
    constraints = []

    # I believe that here need a while(True), with stop condition when the problem is not feasible or the origin is reached.

    T = 10

    for t in range(T):
    	for k in range(N):
            X[k] = cvx.Variable(n)
            U[k] = cvx.Variable(m)

            cost_terms.append(cvx.quad_form(X[k], Q))
            cost_terms.append(cvx.quad_form(U[k], R))

            constraints.append(U[k] <= uUB)
            constraints.append(U[k] >= uLB)

            constraints.append(X[k][0] <= xUB0)
            constraints.append(X[k][0] >= xLB0)

            constraints.append(X[k][1] <= xUB1)
            constraints.append(X[k][1] >= xLB1)

            if k == 0:
              constraints.append(X[k] == x0)
        
            if k > 0:
              constraints.append(A @ X[k - 1] + B @ U[k - 1] == X[k])

        #X[k+1] = cvx.Variable(n)

        if not R2:
            constraints.append(np.zeros(2) == X[k+1])
        else:
            constraints.append(A @ X[k] + B @ U[k] == X[k+1])
            
        cost_terms.append(cvx.quad_form(X[k+1], P))

        # this goes to outside the loop
        obj = cvx.Minimize(cvx.sum(cost_terms))
        problem = cvx.Problem(obj, constraints)
        problem.solve()

        # If NOT feasible or origin reached stop the loop and return the valid data (Stop condition)
        status = problem.status
        if status in ["infeasible", "unbounded"]:
            break
        else:

            for k in range(N):
                u.append(U[k].value)
                x.append(X[k].value)

            x.append(X[k+1].value)
            x0 = A * x[t] + B * u[0]

    return np.asarray(x), np.asarray(u), status, cost_terms

def compute_riccati(A, B, Q, R):
    return linalg.solve_discrete_are(A,B,Q,R)

def discrete_ss(dt=0.25, xc=10):
    space = []
    for x1 in np.linspace(-xc, xc, int(xc/dt)):
        for x2 in np.linspace(-xc, xc, int(xc/dt)):
            x = np.array([x1,x2])
            space.append(x)
    
    return space

def get_attraction_set(grid,A,B,Q,R,P,R2,N, uLB, uUB, xLB0, xUB0, xLB1, xUB1, letter):
    result = []
    for x0 in grid:
        x, _, status, _ = recent_horizon(A,B,Q,R,P,x0,N,R2, uLB, uUB, xLB0, xUB0, xLB1, xUB1)
        if letter != 'b':
            if np.abs(x[-1][0]) <= 0.1 and np.abs(x[-1][1]) <= 0.1  and status not in ["infeasible", "unbounded"]:
               result.append((x,x0))
        else:
            result.append((x,x0))

    return result

# This for question (h)
def iterate_N(A,B,Q,R,P,R2, uLB, uUB, xLB0, xUB0, xLB1, xUB1):
    N = range(2, 11)
    x0 = np.array([10., -2.3])
    trajectory = []
    for n in N:
        x, u, status, cost = recent_horizon(A,B,Q,R,P,x0,n,R2, uLB, uUB, xLB0, xUB0, xLB1, xUB1)
        if status not in ["infeasible", "unbounded"]:
            x_news = []
            for x_, cost_ in zip(x,cost):
                x_new = A @ x_ + B * u[0]
                x_news.append(x_new)
            trajectory.append(x_news)

    plot_traj_cost(trajectory)

def plot_traj_cost(trajectory):

    _, axes = plt.subplots(1)
    axes.set_title('Receding Horizon Problem')
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    axes.grid(True)
    for data in trajectory:
        arr = np.asarray(data)
        axes.plot(arr[:,0],arr[:,1])
    plt.savefig('problem_02_h.pdf', bbox_inches='tight')
    plt.show()
        
def plot_ss(result, letter):
    _, axes = plt.subplots(1)
    axes.set_title('Receding Horizon Problem')
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    axes.grid(True)
    set_initial = []
    for data in result:
        axes.plot(data[0][:,0],data[0][:,1])
        set_initial.append('[' + str(np.round(data[1][0],2)) + ',' + str(np.round(data[1][1],2)) + ']')
    
    print(set_initial)
    plt.savefig('problem_02_' + letter + '.pdf', bbox_inches='tight')
    plt.show()

# This is for questions (b - h)
def problem_2(letter):
    result = []
    if letter == 'b':
        grid_ = [np.array([-4.5, 2]), np.array([-4.5, 3])]
        result = get_attraction_set(grid_,A,B,Q,R,P,True,10, uLB=-0.5,uUB=0.5,xLB0=-5,xUB0=5,xLB1=-5,xUB1=5, letter='b')
    elif letter == 'c':
        result = get_attraction_set(grid,A,B,Q,R,P,False,2, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10, letter='')
    elif letter == 'd':
        result = get_attraction_set(grid,A,B,Q,R,P,False,6, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10, letter='')
    elif letter == 'e':
        result = get_attraction_set(grid,A,B,Q,R,P,True,2, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10, letter='')
    elif letter == 'f':
        result = get_attraction_set(grid,A,B,Q,R,P,True,6, uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10, letter='')
    elif letter == 'h':
        iterate_N(A,B,Q,R,P, False,uLB=-1,uUB=1,xLB0=-10,xUB0=10,xLB1=-10,xUB1=10)
        return
    else:
        print('Choose valid: b - f')
        return

    plot_ss(result, letter)

A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
Q = np.eye(2)
R = 0.01*np.eye(1)
P = compute_riccati(A,B,Q,R)
grid = discrete_ss()

problem_2('b')

