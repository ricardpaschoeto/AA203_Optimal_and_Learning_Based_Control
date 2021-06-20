from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

import animations as an

def cartpole(s,t,u):
    mp = 2.0
    mc = 10.0
    l = 1.0
    g = 9.81
    _, teta, x_dot, teta_dot = s
    x_ddot = (mp*np.sin(teta)*(l*teta_dot**2 + g*np.cos(teta)) + u)/(mc + mp*(np.sin(teta))**2)
    teta_ddot = -((mc + mp)*g*np.sin(teta) + mp*l*(teta_dot**2)*np.sin(teta)*np.cos(teta) + u*np.cos(teta))/((mc + mp*(np.sin(teta))**2)*l)
    dsdt = [x_dot, teta_dot, x_ddot, teta_ddot]
    return dsdt

def noise(mean, cov):
    cov_matrix = np.diag(cov)
    w = np.random.multivariate_normal(mean, cov_matrix, 1)
    return w

def simulate(s0, animated, add_noise):
    tf = 30.0
    dt = 0.1
    t = np.linspace(0, tf, int(tf/dt) + 1)
    kinf = [0.7291397,  -231.85419281,    4.21967188,  -68.24742825]
    w =  noise(np.array([0,0,0,0]), np.array([0, 0 ,10**(-4),10**(-4)]))
    s_star = np.array([0, np.pi, 0, 0])
    if add_noise:
        s = [s0 + w[0]]
    else:
        s = [s0]
    u = [kinf @ (s[0] - s_star)]

    for k in range(len(t)-1):
        if add_noise:
            w =  noise(np.array([0,0,0,0]), np.array([0, 0 ,10**(-4),10**(-4)]))
            s.append((odeint(cartpole, s[k], t[k:k+2],(u[k],))[1] + w[0]))
            u.append(kinf @ (s[k] - s_star))            
        else:
            s.append(odeint(cartpole, s[k], t[k:k+2],(u[k],))[1])
            u.append(kinf @ (s[k] - s_star))
        
    _, ax = plt.subplots()
    ax.plot(t, np.asanyarray(s)[:,0], 'k--', label='Displacement')
    ax.plot(t, np.asanyarray(s)[:,1], 'g:', label='Angle')
    ax.plot(t, np.asanyarray(s)[:,2], 'b', label='Velocity')
    ax.plot(t, np.asanyarray(s)[:,3], 'r', label='Agular Velocity')
    plt.grid(True)
    ax.legend()
    if animated:
        _, _ = an.animate_cartpole(t, np.asanyarray(s)[:,0], np.asanyarray(s)[:,1])
    
    plt.show()

simulate(np.array([0, 3*np.pi/4, 0, 0]), False, True)
