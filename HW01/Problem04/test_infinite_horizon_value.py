import numpy as np
from lib.envs.gridworld import GridworldEnv


env = GridworldEnv()

def value_iteration(env, epsilon=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    def one_step_lookahead(V, a, s):

    	[(prob, next_state, reward, done)] = env.P[s][a]
    	v = prob * (reward + discount_factor * V[next_state])

    	return v
    
    #start with inital value function and intial policy
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    #while not the optimal policy
    while True:
      #for stopping condition
    	delta = 0

    	#loop over state space
    	for s in range(env.nS):

    		actions_values = np.zeros(env.nA)

    		#loop over possible actions
    		for a in range(env.nA):

    			#apply bellman eqn to get actions values
    			actions_values[a] = one_step_lookahead(V, a, s)

    		#pick the best action
    		best_action_value = max(actions_values)

    		#get the biggest difference between best action value and our old value function
    		delta = max(delta, abs(best_action_value - V[s]))

    		#apply bellman optimality eqn
    		V[s] = best_action_value

    		#to update the policy
    		best_action = np.argmax(actions_values)

    		#update the policy
    		policy[s] = np.eye(env.nA)[best_action]


    	#if optimal value function
    	if(delta < epsilon):
    		break
    
    return policy, V


policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")