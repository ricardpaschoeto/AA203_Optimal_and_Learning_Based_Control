import numpy as np

def riccati():
    dt = 0.1
    mp = 2.0
    mc = 10.0
    l = 1.0
    g = 9.81
    Pk = np.zeros((4,4))
    Kk = np.zeros((1,4))
    Q = np.eye(4,4)
    R = 1
    A = np.matrix([[1,0,dt,0],[0,1,0,dt],[0,dt*mp*g/mc,1,0],[0,dt*(mc+mp)*g/(mc*l),0,1]])
    B = np.array([0,0,dt/mc,dt/(mc*l)]).reshape(4,1)
    while True:
        Pk_adv = Pk            
        Kk = (np.linalg.inv(-(R + (B.T)@Pk_adv@B))*(B.T)@Pk_adv@A)
        Pk = (Q + A.transpose()@Pk_adv@(A + B@Kk))    
        print(Kk.round(2))
        if np.linalg.norm(Pk_adv - Pk) < 10**(-4):
            break
    
    return Kk

K = riccati()