import numpy as np
import os

class GridWorld:
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.actions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        self.state = np.array([0, 0])
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.state = np.array([0, 0])

    def step(self, action):
        reward = 0
        if self.grid[self.state[1], self.state[0]] == 'F' or self.steps > 1000:
            return self.state, reward, True
        self.steps += 1
        reward -= 0.1
        action = self.actions[action]
        newstate = np.add(self.state, action)
        if newstate[0] < 0 or newstate[0] >= self.grid.shape[1] or newstate[1] < 0 or newstate[1] >= self.grid.shape[0]:
            return self.state, reward, False
        field = self.grid[newstate[1], newstate[0]]
        if field == 'O':
            self.state = newstate
        elif field == 'T':
            reward -= 1
            self.state = newstate
        elif field == 'F':
            reward += 100
            self.state = newstate
            return self.state, reward, True
        return self.state, reward, False
    
    def action_space(self):
        return self.actions.shape[0]
    
    def observation_space(self):
        return self.grid.shape

    def visualize(self):
        os.system('cls' if os.name=='nt' else 'clear')
        print('-'*(self.grid.shape[1]*2+1))
        for i in range(self.grid.shape[1]):
            printrow = '|'
            for j in range(self.grid.shape[0]):
                field = self.grid[i, j]
                if self.state[0] == j and self.state[1] == i:
                    printrow += 'x'
                elif field == '':
                    printrow += ' '
                else:
                    printrow += field
                printrow += '|'
            print(printrow)
            print('-'*(self.grid.shape[1]*2+1))
