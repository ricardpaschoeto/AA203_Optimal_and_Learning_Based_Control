import numpy as np

class simulator:
    def  __init__(self):
        
        self.sell_price = 1.2
        self.buy_price = 1.
        self.hold_price = 0.05
        self.rent = 1.
        
        self.init_state = 5
        self.valid_states = [0,1,2,3,4,5]
        self.valid_actions = [0,2,4]
        self.valid_demands = [0,1,2,3,4,5]
        
        self.demand_probs = [0.1,
                             0.3,
                             0.3,
                             0.2,
                             0.1,
                             0.0]
        
    def reset(self):
        self.state = self.init_state
        return self.state
        
    def check_inputs(self, state, action, demand):
        # make sure only legal inputs are provided 
        checks = [state,action,demand]
        checklists = [self.valid_states, self.valid_actions, self.valid_demands]
        
        for i,c in enumerate(checks):
            if c is None:
                pass
            elif c not in checklists[i]:
                raise ValueError('Input must be in ' + str(checklists[i]))
        
    def step(self,action):
        self.check_inputs(None, action, None)
        
        demand = self.get_demand()
        next_state = self.transition(self.state,action,demand)
        rew = self.get_reward(self.state,action,demand)

        self.state = next_state
        
        return next_state, rew
        
    def transition(self,state,action,demand):
        self.check_inputs(state, action, demand)
        
        return min(max(state + action - demand, np.min(self.valid_states)), np.max(self.valid_states))
        
    def get_demand(self):
        return np.random.choice(len(self.demand_probs),p=self.demand_probs)
    
    def get_reward(self, state, action, demand):
        self.check_inputs(state, action, demand)

        # demand satisfaction reward 
        satisfied_demand = min(demand, state + action)
        demand_rew = self.sell_price*satisfied_demand
        
        # inventory holding cost
        hold_cost = self.hold_price*state
        
        # purchase cost 
        buy_cost = self.buy_price*np.sqrt(action)
        
        return -hold_cost - buy_cost + demand_rew - self.rent

    # def evaluate_policy(self, policy, gamma=0.95):
    # # `policy` is a vector that maps states to actions, e.g., [4, 4, 2, 0, 0, 0].
    #     num_states = len(.valid_states)
    #     transition_matrix = np.zeros((num_states, num_states))
    #     expected_single_step_reward = np.zeros(num_states)
    #     for state in self.valid_states:
    #         for demand, prob in enumerate(self.demand_probs):
    #             expected_single_step_reward[state] += self.get_reward(state, policy[state], demand) * prob
    #             transition_matrix[state, self.transition(state, policy[state], demand)] += prob
        
    #     return np.linalg.solve(np.eye(num_states) - gamma * transition_matrix, expected_single_step_reward)
        
def random_policy():
    return np.random.choice(3)*2
    
def generate_historical_data(sim):
    np.random.seed(0)
    T = 3*365
    
    data = np.zeros((T,4))
    
    s = sim.reset()
    for t in range(T):
        a = random_policy()
        sp, r = sim.step(a)
        
        data[t,:] = [s,a,r,sp]
        s = sp
        
    return data


        