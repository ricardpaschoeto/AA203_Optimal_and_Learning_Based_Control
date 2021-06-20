import numpy as np
import matplotlib.pyplot as plt
from grid import Grid as G
import seaborn as sb
import pandas as pd
import math as m
from matplotlib.patches import Rectangle

n=20
env = G(n)

x_eye=(15,15)
x_goal=(19,9)

env.actions_space(x_goal)
env.set_rewards(x_goal)

V = {}
for x in env.all_states:
    V[x] = 0
V[x_goal] = 0

policy = {}
for x in env.aS.keys():
    policy[x] = np.random.choice(env.aS[x])

def value_iteration(env, epsilon=0.0001, discount=0.95, sigma=10):

    def next_step(a, x):
        x1, x2 = x
        x1_eye, x2_eye = x_eye
        w = np.exp(-((x1 - x1_eye)**2 + (x2 - x2_eye)**2)/(2*(sigma**2)))

        v = np.zeros(4)
        for a in env.aS[x]:
            x_next = env.prob(a, x)
            for r_act in env.aS[x]:
                if (r_act == a):
                    p = (1 - w + w/len(env.aS[x]))
                    r = env.rewards[x_next]
                    v_next = V[x_next]
                else:
                    x_next_rand = env.prob(r_act, x)
                    p = w/len(env.aS[x])
                    r = env.rewards[x_next_rand]
                    v_next = V[x_next_rand]

                v[a] += p*(r + discount*v_next)

        return v

    iteration = 0
    while True :
        delta = 0
        for x in env.all_states:            
            Q = next_step(V, x)
            best_v_action = max(Q)
            delta = max(delta, np.abs(V[x] - best_v_action))
            V[x] = best_v_action
            policy[x] = np.argmax(Q)

        if delta < epsilon:
            break
        iteration += 1

    print(iteration)

    return V, policy

def plot_heatmap(data, ant):
    m = np.array([data[key] for key in data.keys()]).reshape((n, n)).T
    ax = sb.heatmap(np.round(m, 3), annot=ant)
    ax.invert_yaxis()
    p = path(data, (0,19))
    for t in p:
        ax.add_patch(Rectangle(t, 1, 1, fill=False, edgecolor='blue', lw=3))
    plt.show()

def path(policy, x_start):
    p = []
    x = x_start
    N = 100
    p.append(x_start)
    iteration = 0
    while iteration <= N:
        if x == x_goal:
            break
        x = env.prob(policy[x], x)
        p.append(x)
        iteration += 1
    return p

V, policy = value_iteration(env)
plot_heatmap(V, False)
plot_heatmap(policy, True)








