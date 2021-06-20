import numpy as np
import matplotlib.pyplot as plt

alpha = -1
beta = 3
alpha_m = 4
beta_m = 4
dt = 0.01
gamma = 2.0
tval = np.linspace(0,10,int((10/dt)))
y = 0
ym = 0
kr = 0
ky = 0

y_history = []
ym_history = []
kr_history = []
ky_history = []
error_history = []
delta_r_hist = []
delta_y_hist = []

y_history.append(y)
ym_history.append(ym)
kr_history.append(kr)
ky_history.append(ky)

kr_star = beta_m/beta
ky_star = (alpha - alpha_m)/beta
kr_star_hist = kr_star*np.ones(len(tval))
ky_star_hist = ky_star*np.ones(len(tval))

error = 0
error_history.append(error)
dr = 0 - kr_star
delta_r_hist.append(dr)
dy = 0 - ky_star
delta_y_hist.append(dy)

ref_in = 1

def ref(t):
    if ref_in == 0:
        r = 4
    else:
        r = 4*np.sin(3*t)
    return r

def y_dot(y, r, ky, kr):
    return 3*control(y, r, ky, kr) + y

def control(y, r, ky, kr):
    return kr*r + ky*y

def ym_dot(ym, r):
    return 4*(r - ym)

def kr_dot(gamma, error, r):
    return -gamma*error*r

def ky_dot(gamma, error, y):
    return -gamma*error*y

def plot_outputs(y_history, ym_history):
    plt.figure
    plt.plot(tval, y_history)
    plt.plot(tval, ym_history)
    plt.title('True Model and Reference Model')
    plt.legend('y')
    plt.legend('ym')
    plt.xlabel('time')
    plt.ylabel('Outputs')
    plt.grid(True)
    plt.show()

def plot_gains(kr_history, ky_history, kr_star_hist, ky_star_hist):
    plt.figure
    plt.plot(tval, kr_history,'b', 'LineWidth',1)
    plt.plot(tval, kr_star_hist,'b--', 'LineWidth',2)
    plt.plot(tval, ky_history,'r', 'LineWidth',1)
    plt.plot(tval, ky_star_hist,'r--', 'LineWidth',2)
    plt.title('Gains Time history')
    plt.xlabel('time')
    plt.ylabel('Gains')
    plt.legend('kr') 
    plt.legend('kr*')
    plt.legend('ky')
    plt.legend('ky*')
    plt.grid(True)
    plt.show()

for t in tval[1:]:
    r = ref(t)
    y = y + np.dot(y_dot(y, r, ky, kr), dt)
    y_history.append(y)

    ym = ym + np.dot(ym_dot(ym, r), dt)
    ym_history.append(ym)
    error = y - ym
    error_history.append(error)
    kr = kr + np.dot(kr_dot(gamma, error, r), dt)
    kr_history.append(kr)
    ky = ky + np.dot(ky_dot(gamma, error, y), dt)
    ky_history.append(ky)

    delta_r = kr - kr_star
    delta_r_hist.append(delta_r)
    delta_y = ky - ky_star
    delta_y_hist.append(delta_y)

plot_outputs(y_history, ym_history)
plot_gains(kr_history, ky_history, kr_star_hist, ky_star_hist)
