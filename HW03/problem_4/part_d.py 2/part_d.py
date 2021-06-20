from model import dynamics, cost
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import gym
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from tqdm import tqdm



stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100
#N = 10000 
N = 1000
gamma = 0.95 # discount factor

total_costs = []

# Define the policy Network
state_space_size = 4
output_size = 5 # 2-D Mu vector, 3 paramters for covariance matrix
num_hidden_layer = 32
model = torch.nn.Sequential(torch.nn.Linear(state_space_size, num_hidden_layer), 
                            torch.nn.ReLU(), 
                            torch.nn.Linear(num_hidden_layer, output_size),
                            torch.nn.Softmax())

optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train()

losses = []
mean_rewards = []



for n in tqdm(range(N)):
    costs = []
    log_probs = []
    rewards = []
    entropies = []
    
    x = dynfun.reset()
    for t in range(T):

        # TODO compute action
        outputs = model(Variable(torch.FloatTensor(x)))
        mu_vector = outputs[0:2]

        L = torch.eye(2)
        L[0][0] = outputs[4]
        L[1][1] = outputs[3]
        L[1][1] = outputs[2]

        epsilon = 0.01
        cov_matrix = L @ L.T + epsilon * torch.eye(2)
        cov_matrix *= 0.1
        
        #print("mu_vector = " + str(mu_vector))
        #print("mu_vector shape = " + str(mu_vector.shape))
        #print("cov_matrix = " + str(cov_matrix))
        #print("cov_matrix shape = " + str(cov_matrix.shape))

        dist = MultivariateNormal(mu_vector, cov_matrix)
        u = dist.sample()
        entropies.append(dist.entropy())
        #print("u = " + str(u))

        log_prob = dist.log_prob(u)
        #print("log probability = " + str(log_prob))
        log_probs.append(log_prob)

        # get reward
        #c = costfun.evaluate(x,u)

        x1 = Variable(torch.FloatTensor(x))

        Q = torch.eye(4)
        Q[2,2] *= 0.1
        Q[3,3] *= 0.1
        R = 0.01 * torch.eye(2)
        c = torch.matmul(torch.matmul(x1, Q), x1) + torch.matmul(torch.matmul(u, R), u)
        rewards.append(-c)
        
        # dynamics step
        xp = dynfun.step(u.detach().numpy())

        x = xp.copy()

    # TODO update policy
    R = torch.zeros(1, 1)
    
    loss = 0

    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        #print("rewards[i] = " + str(rewards[i]))
        loss = loss - log_probs[i] * Variable(R) - 0.0001 * entropies[i]

    loss = loss / len(rewards)
    #print("Mean reward = " + str(np.mean(rewards)))
    #print("Loss = " + str(loss))
    losses.append(loss.item())
    mean_rewards.append(np.mean(rewards))

    optimizer.zero_grad()

    loss.backward()
    utils.clip_grad_norm(model.parameters(), 40)
    optimizer.step()

    total_costs.append(sum(costs))

plt.xlabel('Episode Number')
plt.ylabel('Loss')

plt.plot(losses)
plt.show()

plt.xlabel('Episode Number')
plt.ylabel('Mean reward')

plt.plot(mean_rewards)
plt.show()

