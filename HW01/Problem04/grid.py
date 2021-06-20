import numpy as np


class Grid:

    # action (0=up, 1=down, 2=left, 3=right)
   
    def __init__(self, n):
        self.actions = [0, 1, 2, 3]
        self.n = n
        self.all_states = self.grid()
        self.ns = n*n
        self.na = len(self.actions)
        self.rewards = {}
        self.aS = {}

    def grid(self):
        states = []
        for x1 in range(self.n):
            for x2 in range(self.n):
                states.append((x1,x2))    
        return states

    def actions_space(self, x_goal):
        for x in self.all_states:
            act = ()
            for a in self.actions:
                x_next = self.execute(a, x)
                if self.validate(x_next):
                    act = act + (a,)
            self.aS[x] = act

    def set_rewards(self, x_goal):
        for x in self.all_states:
            if x == x_goal:
                self.rewards[x] = 1
            else:
                self.rewards[x] = 0

    def prob(self, a, x):  
        x_next = self.execute(a, x)
        if not(self.validate(x_next)):
                x_next = x

        return x_next

    def validate(self, x):
        x1, x2 = x
        if (x1 >= self.n or x1 < 0) or (x2 >= self.n or x2 < 0):
            return False
        return True

    def execute(self, a,x):
        x1, x2 = x
        if a == 0:
            x_next = (x1, x2 + 1)
        elif a == 1:
            x_next = (x1, x2 - 1)
        elif a == 2:
            x_next = (x1 - 1, x2)
        else:# a == 3
            x_next = (x1 + 1, x2)

        return x_next
