from problem1_q_learning_env import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
sim = simulator()

T = 5*365 # simulation duration
gamma = 0.95 # discount factor
epsilon = 0.1
alpha = 0.2

# get historical data
data = generate_historical_data(sim)
# historical dataset: 
# shape is 3*365 x 4
# k'th row contains (x_k, u_k, r_k, x_{k+1})
N = 10

x0_hist = []
x1_hist = []
x2_hist = []
x3_hist = []
x4_hist = []
x5_hist = []
r_hist = []

# TODO: write Q-learning to yield Q values,
# use Q values in policy (below)

def policy(state, Q):
    return np.argmax(Q[state])*2

def e_greedy(state,Q):
    if np.random.random() < epsilon:
        a = random_policy()
        index = int(a/2)
        return random_policy(), index
    else:         
        index = np.argmax(Q[state])
        a = index*2 
        return a, index

def q_learning():
    q_values = np.zeros((len(sim.valid_states),len(sim.valid_actions)))
    for n in range(N):
        s = sim.reset()
        for t in range(len(data)):
            x0_hist.append(copy.deepcopy(q_values[0]))
            x1_hist.append(copy.deepcopy(q_values[1]))
            x2_hist.append(copy.deepcopy(q_values[2]))
            x3_hist.append(copy.deepcopy(q_values[3]))
            x4_hist.append(copy.deepcopy(q_values[4]))
            x5_hist.append(copy.deepcopy(q_values[5]))

            a, index = e_greedy(s,q_values)
            sp,r = sim.step(a)
            td = r + gamma*np.max(q_values[sp]) - q_values[s,index]
            q_values[s,index] += alpha*td
            s = sp

    return q_values

def simulation(Q):
    s = sim.reset()
    r_hist.append(0)
    for t in range(T):
        a = policy(s,Q)
        sp, r = sim.step(a)
        r_hist.append(r)
        s = sp

# TODO: write value iteration to compute true Q values
# use functions:
# - sim.transition (dynamics)
# - sim.get_reward 
# plus sim.demand_probs for the probabilities associated with each demand value
iteration = 0
def value_iteration(sim, epsilon=0.001):
    V = np.zeros((len(sim.valid_states)))
    q_values = np.zeros((len(sim.valid_states),len(sim.valid_actions)))
    
    def next_step(V, x):
        nv = 0
        policy = 0
        for a in sim.valid_actions:
            for d, prob in enumerate(sim.demand_probs):
                next_state = sim.transition(x,a, d)
                r = sim.get_reward(x, a, d)
                v = sim.demand_probs[next_state] * (r + gamma * V[next_state])

                if nv < v:
                    nv = v
                    policy = a
        return nv, policy             

    while True:
        delta = 0
        for x in sim.valid_states:
            prev_v = V[x]
            best_v, best_a = next_step(V, x)
            V[x] = best_v
            delta = max(delta, np.abs(prev_v - V[x]))
            q_values[x][int(best_a/2)] = V[x]

        print(delta)
        if delta < epsilon:
            break

    return q_values

def plot_q():
    fig, axs = plt.subplots(6)
    fig.suptitle('Q values for actions to each state')
    
    axs[0].plot(x0_hist)
    axs[0].legend(['action 0', 'action 2', 'action 4'])
    axs[0].set_title('state 0')
    axs[0].get_xaxis().set_visible(False)
    axs[0].grid(True)

    axs[1].plot(x1_hist)
    axs[1].legend(['action 0', 'action 2', 'action 4'])
    axs[1].set_title('State 1')
    axs[1].get_xaxis().set_visible(False)
    axs[1].grid(True)

    axs[2].plot(x2_hist)
    axs[2].legend(['action 0', 'action 2', 'action 4'])
    axs[2].set_title('State 2')
    axs[2].get_xaxis().set_visible(False)
    axs[2].grid(True)

    axs[3].plot(x3_hist)
    axs[3].legend(['action 0', 'action 2', 'action 4'])
    axs[3].set_title('State 3')
    axs[3].get_xaxis().set_visible(False)
    axs[3].grid(True)

    axs[4].plot(x4_hist)
    axs[4].legend(['action 0', 'action 2', 'action 4'])
    axs[4].set_title('State 4')
    axs[4].get_xaxis().set_visible(False)
    axs[4].grid(True)

    axs[5].plot(x5_hist)
    axs[5].legend(['action 0', 'action 2', 'action 4'])
    axs[5].set_title('State 5')
    axs[5].grid(True)
    plt.show()

Qiv = value_iteration(sim)
#Q = q_learning()
# plot_q()
simulation(Qiv)
plt.plot(np.cumsum(r_hist))
plt.title('Iteration Value Simulation for 5 years')
plt.xlabel('time')
plt.ylabel('Aggregate reward')
plt.grid(True)
plt.show()