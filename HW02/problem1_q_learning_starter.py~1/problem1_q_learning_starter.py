from problem1_q_learning_env import *
import matplotlib.pyplot as plt

import numpy as np
sim = simulator()

T = 5*365 # simulation duration
gamma = 0.95 # discount factor


# get historical data
data = generate_historical_data(sim)
# historical dataset: 
# shape is 3*365 x 4
# k'th row contains (x_k, u_k, r_k, x_{k+1})

# TODO: write Q-learning to yield Q values,
# use Q values in policy (below)

Q = {}
alpha = 0.7
gamma = 0.95
epsilon = 0.1

num_episodes = 10000
num_timesteps = 3*365

def policy(state,Q):

    # Epsillon greedy
    if np.random.uniform(0, 1) < epsilon:
       action = np.random.choice(3)*2
    else:
       Q_keys = np.array(list(Q.keys()))
       Q_values = np.array(list(Q.values()))
       action = Q_keys[np.argmax(Q_values)][1]

    return action


def policy_inference(state, Q):

    Q_keys = np.array(list(Q.keys()))
    Q_values = np.array(list(Q.values()))
    action = Q_keys[np.argmax(Q_values)][1]

    return action

def q_learning():
    for state in sim.valid_states:
        for action in sim.valid_actions:
            Q[(state, action)] = 0.0

    # Q-Learning
    num_iterations = 0
    for i in range(num_episodes):
        state = sim.reset()

        # Sample action based on epsilon-greedy policy
        action = policy(state, Q)

        for t in range(num_timesteps):
            s_next, reward = sim.step(action)

            opt_action_index = np.argmax([Q[(s_next, a)] for a in sim.valid_actions])
            Q_S_values =  [(s_next, a) for a in sim.valid_actions]
            opt_action = Q_S_values[opt_action_index][1]
            Q[(state, action)] += alpha * (reward + gamma * Q[(s_next, opt_action)] - Q[(state, action)])
            state = s_next

            num_iterations += 1
            if num_iterations % 100000 == 0:
               print("Q value funciton: " + str(Q))
               print("num_iterations = " + str(num_iterations / 100000))

    print("Q-value table after convergence =" + str(Q))
    return Q


#Q = q_learning()
#print("Q-value table after convergence =" + str(Q))

# Forward simulating the system 
def simulate(Q):
    print("in simulate")
    s = sim.reset()
    rewards = {}
    for t in range(T):
        a = policy(s,Q)
        sp,r = sim.step(a)
        s = sp
        # TODO add logging of rewards for plotting
        if s not in rewards:
           rewards[s] = r
        else:
           rewards[s] += r

    plt.plot(rewards.keys(), rewards.values())

# TODO: write value iteration to compute true Q values
# use functions:
# - sim.transition (dynamics)
# - sim.get_reward 
# plus sim.demand_probs for the probabilities associated with each demand value

Q = {0: 4, 1: 2,  2: 2, 3: 2, 4: 0, 5: 0}
simulate(Q)

V = {}
policy = {}

def value_iteration():
    # Initialize Value function to zero
    for state in sim.valid_states:
        V[state] = 0.0

    iterations = 0
    converged = False
    converge_epsilon = 0.001

    state = sim.reset()

    # Value Iteration
    while (not converged):
       max_change = 0
       for state in sim.valid_states:
           prev_v = V[state]
           new_v = 0
           for action in sim.valid_actions:
              sim.check_inputs(None, action, None)
              demand = sim.get_demand()
              next_state = sim.transition(state, action, demand)
              reward = sim.get_reward(state, action, demand)
              v = sim.demand_probs[next_state] * (reward + gamma * V[next_state])

              if new_v < v:
                 new_v = v
                 policy[state] = action

           V[state] = new_v
           max_change = max(max_change, np.abs(prev_v - V[state]))

       iterations += 1
       if iterations % 1000 == 0:
          print("iterations = " + str(iterations))
          print("max_change = " + str(max_change))

       if max_change < converge_epsilon:
          print("max_change at time of convergence = " + str(max_change))
          converged = True

    return policy, V 

#state = sim.reset()

Q = q_learning()
simulate(Q)
#policy, V = value_iteration()

#print("policy = " + str(policy))
#print("V = " + str(V))
