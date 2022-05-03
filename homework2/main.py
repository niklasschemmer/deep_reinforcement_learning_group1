from turtle import shape
from gridWorld import GridWorld
import numpy as np
from queue import Queue
import os
import time

def main():
    #world = GridWorld([['O', 'O', 'O', 'O', 'O', 'O'],
    #                   ['O', 'O', '', '', 'T', 'O'],
    #                   ['O', 'O', '', 'O', 'T', 'O'],
    #                   ['', '', 'O', 'O', 'O', 'T'],
    #                   ['O', 'O', '', 'O', 'O', 'O'],
    #                   ['F', 'O', 'O', 'O', 'T', 'O']])
    world = GridWorld([['O', 'O', 'O', 'O', 'O'],
                       ['O', 'O', 'O', 'O', 'O'],
                       ['O', 'O', 'O', 'O', 'O'],
                       ['O', 'O', 'O', 'F', 'O'],
                       ['O', 'O', 'O', 'O', 'O']])

    Q_values = np.ones(world.observation_space() + (4,))
    episodes = 1000
    sarsa_n_values = 3
    discountfactor = 0.9
    learningRate = 0.1
    visualizeSteps = False
    action_space = world.action_space()

    for i in range(episodes):
        world.reset()
        state, reward, done = [0, 0], 0, False
        actEpsilon = 1-i/episodes
        laststeps = np.empty(shape=(0,3))

        while not done:
            if visualizeSteps:
                world.visualize()
                time.sleep(0.001)
            action = None
            if np.random.rand() > actEpsilon:
                selectQ = Q_values[state[1], state[0]]
                action = np.argmax(selectQ)
            else:
                action = np.random.randint(action_space)
            state, reward, done = world.step(action)

            if len(laststeps) >= sarsa_n_values - 1:
                summedRewards = 0
                for j in range(sarsa_n_values-1):
                    summedRewards += laststeps[j][2] * (discountfactor ** j)
                newQVal = summedRewards + Q_values[state[1], state[0], action]
                oldQVal = Q_values[laststeps[0][0][1], laststeps[0][0][0], laststeps[0][1]]
                Q_values[laststeps[0][0][1], laststeps[0][0][0], laststeps[0][1]] = oldQVal * (1 - learningRate) + learningRate * newQVal
                laststeps = laststeps[1:]
            laststeps = np.vstack([laststeps, [state, action, reward]])
        
        if i%500 == 499:
            visualize_qvalues(Q_values)

def visualize_qvalues(qValues):
    os.system('cls' if os.name=='nt' else 'clear')
    print(2*'\n')
    for i in range(qValues.shape[1]*3):
        printRow = ''
        for j in range(qValues.shape[0]*3):
            printRow += ' '
            if i%3==0:
                if j%3==1:
                    printRow += '%05.0f' % qValues[int(i/3)][int(j/3)][1]
                else:
                    printRow += ' '*4
            elif i%3==1:
                if j%3==0:
                    printRow += '%05.0f' % qValues[int(i/3)][int(j/3)][0]
                elif j%3==2:
                    printRow += '%05.0f' % qValues[int(i/3)][int(j/3)][2]
                else:
                    printRow += ' '*3
            elif i%3==2:
                if j%3==1:
                    printRow += '%05.0f' % qValues[int(i/3)][int(j/3)][3]
                else:
                    printRow += ' '*4
        print(printRow)

if __name__ == "__main__":
    main()