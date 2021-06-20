import numpy as np
from scipy.integrate import odeint
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy as cvx

from animations import animate_cartpole

@jax.partial(jax.jit, static_argnums=(0,))
def linearize(f, x, u):
    ####################################################################################
    # WRITE YOUR CODE HERE
    A, B, c = jax.jacfwd(lambda x:f(x, u))(x), jax.jacfwd(lambda u:f(x, u))(u), f(x, u)
    ####################################################################################  
    return A, B, c

def scp(f,Q,R,Q_N,s_star,s0,N,dt,rho,uLB,uUB):

    # Outer loop of scp.
    # Implement the inner loop in the function scp_iteration. 

    n = Q.shape[0] # state dimension
    m = R.shape[0] # control dimension
    eps = 0.01 # termination threshold for scp

    # initialize reference rollout s_bar,u_bar
    #u_bar = np.random.uniform(-3,3,(N,m))
    u_bar = np.zeros((N,m))

    s_bar = np.zeros((N+1,n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = f(s_bar[k],u_bar[k])


    # Compute new state and control via scp.
    s, u = scp_iteration(f,Q,R,Q_N,s_bar,u_bar,s_star,s0,N,dt,rho,uLB,uUB)

    # run scp until u converges
    round = 0
    while(np.linalg.norm(u-u_bar,np.inf) > eps):
        print("round: %s, u update: %s" % (round, np.linalg.norm(u-u_bar,np.inf)))
        round = round+1
        se_bar = s
        u_bar = u
        s,u = scp_iteration(f,Q,R,Q_N,s_bar,u_bar,s_star,s0,N,dt,rho,uLB,uUB)
    
    return s,u

def scp_iteration(f,Q,R,Q_N,s_bar,u_bar,s_star,s0,N,dt,rho,uLB,uUB):

    ###########################################################################
    # WRITE YOUR CODE HERE
    # implement one iteration of scp
    # HINT: See slides 34-38 of Recitation 1. 

    n = Q.shape[0]
    m = R.shape[0]
    S = {}
    U = {}
    u = np.zeros((N, m))
    s = np.zeros((N+1, n))

    cost_terms = []
    constraints = []

    for t in range(N):

        S[t] = cvx.Variable(n)
        U[t] = cvx.Variable(m)

        cost_terms.append(cvx.quad_form(S[t], Q)) # State cost
        cost_terms.append(cvx.quad_form(U[t], R)) # Control cost

        if t == 0: # S[0] == s0 # Initial condition
            constraints.append(S[t] == s0)

        if t > 0 and t < (N - 1):
           A, B, c = linearize(f, s_bar[t - 1], u_bar[t - 1])
           constraints.append(A[t - 1] @ (S[t - 1] - s_bar[t - 1]) + B[t - 1] @ (U[t - 1] - u_bar[t - 1]) + c[t - 1] == S[t])

        constraints.append(U[t] <= uUB)
        constraints.append(U[t] >= uLB)

        constraints.append(cvx.norm(U[t] - u_bar[t], "inf") <= rho) # Box constraint 1
        constraints.append(cvx.norm(S[t] - s_bar[t], "inf") <= rho) # Box constraint 2

    S[t + 1] = cvx.Variable(n)
    constraints.append(A[t] @ (S[t] - s_bar[t]) + B[t] @ (U[t] - u_bar[t]) + c[t] == S[t + 1])
    cost_terms.append(cvx.quad_form(S[t + 1] - s_star, Q_N))

    obj = cvx.Minimize(cvx.sum(cost_terms))
    problem = cvx.Problem(obj, constraints)
    problem.solve()

    for k in range(len(U)):
       u[k, :] = U[k].value

    for t in range(len(S)):
       s[t, :] = S[t].value

    ###########################################################################
    return s,u


def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 2.     # pendulum mass
    mc = 10.    # cart mass
    ℓ = 1.      # pendulum length
    g = 9.81    # gravitational acceleration

    x, θ, dx, dθ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    h = mc + mp*(sinθ**2)
    ds = jnp.array([
        dx,
        dθ,
        (mp*sinθ*(ℓ*(dθ**2) + g*cosθ) + u[0]) / h,
        -((mc + mp)*g*sinθ + mp*ℓ*(dθ**2)*sinθ*cosθ + u[0]*cosθ) / (h*ℓ)
    ])
    return ds


if __name__ == '__main__':
    # cartpole swing-up simulation parameters
    n = 4                                   # state dimension
    m = 1                                   # control dimension 
    goal_state = np.array([0,np.pi,0,0])    # desired upright pendulum state
    start_state = np.array([0,0,0,0])       # initial downright pendulum state
    dt = 0.1                                # discrete time resolution
    T = 8                                   # total simulation time    

    # specify cost function
    Qf = 1000.*np.eye(4)                    # terminal state cost matrix
    Q = np.diag(np.array([10,10,2,2]))      # state cost matrix
    R = 2.5*np.eye(1)                       # control cost matrix

    # specify cartpole dynamics
    f = jax.jit(cartpole)
    f_discrete = jax.jit(lambda s, u, dt=dt: s + dt*f(s, u))

    # scp parameters 
    rho = 0.5                               # trust region parameter
    uLB = -5.                               # control effort lower bound
    uUB = 3.                                # control effort upper bound

    # solve swing-up with scp
    print('Computing SCP solution ... ', end='')
    t = np.arange(0., T, dt)                
    N = t.size - 1                          
    s,u = scp(f_discrete,Q,R,Qf,goal_state,start_state,N,dt,rho,uLB,uUB) # scp; yours to implement
    print('done!')

    print('Simulating ...')
    for k in tqdm(range(N)):
        s[k+1] = f_discrete(s[k],u[k])
        

    # Plot
    fig, axes = plt.subplots(1, n+1, dpi=100, figsize=(12, 2))
    plt.subplots_adjust(wspace=0.35)
    ylabels = (r'$x(t)$', r'$\theta(t)$',
               r'$\dot{x}(t)$', r'$\dot{\theta}(t)$', r'$u(t)$')
    for i in range(n):
        axes[i].plot(t, s[:, i])
        axes[i].set_xlabel(r'$t$')
        axes[i].set_ylabel(ylabels[i])
    axes[n].plot(t[0:N], u)
    axes[n].set_xlabel(r'$t$')
    axes[n].set_ylabel(ylabels[n])
    plt.savefig('cartpole_scp_swingup.pdf', bbox_inches='tight')
    plt.show()

    # animate the solution
    fig,ani = animate_cartpole(t, s[:,0], s[:,1])
    ani.save('cartpole_scp_swingup.mp4', writer='ffmpeg')
