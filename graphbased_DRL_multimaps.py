import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

dimcells_x = 6
dimcells_y = 10

def num2word(action = -1):
    if action == 0:
        return "NORTH"
    elif action == 1:
        return "EAST"
    elif action == 2:
        return "SOUTH"
    elif action == 3:
        return "WEST"
    elif action == -1:
        return "INIT_POINT"
    else:
        print("Wrong action code passed!")
        return

def print_map(map_state, action = -1):

    print("Action:\t", num2word(action))
    print(map_state)

def set_map(map_type):

    if map_type == "empty":

        map_state = np.zeros((dimcells_y, dimcells_x))

        freecells = dimcells_y*dimcells_x

        init_y = 0
        init_x = 0

        return map_state, freecells, init_y, init_x
    
    elif map_type == "L":

        map_state = np.zeros((dimcells_y, dimcells_x))
        map_state[:7, 3:] = 2

        freecells = dimcells_y*dimcells_x - 7*3

        init_y = 0
        init_x = 0

        return map_state, freecells, init_y, init_x
    
    elif map_type == "triangle":

        map_state = np.array([[0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 2],
                                    [0, 0, 0, 0, 0, 2],
                                    [0, 0, 0, 0, 2, 2],
                                    [0, 0, 0, 0, 2, 2],
                                    [0, 0, 0, 2, 2, 2],
                                    [0, 0, 0, 2, 2, 2],
                                    [0, 0, 2, 2, 2, 2],
                                    [0, 0, 2, 2, 2, 2],
                                    [0, 2, 2, 2, 2, 2]])
        
        freecells = dimcells_y*dimcells_x - (1 + 3 + 5 + 7 + 9)

        init_y = 0
        init_x = 0

        return map_state, freecells, init_y, init_x
    
    elif map_type == "cross":

        map_state = np.zeros((dimcells_y, dimcells_x))
        map_state[:3, :2] = 2
        map_state[7:, :2] = 2
        map_state[7:, 4:] = 2
        map_state[:3, 4:] = 2

        freecells = dimcells_y*dimcells_x - 6*4

        init_y = 3
        init_x = 0

        return map_state, freecells, init_y, init_x
    
    elif map_type == "hole":

        map_state = np.zeros((dimcells_y, dimcells_x))
        map_state[3:7, 2:4] = 2

        freecells = dimcells_y*dimcells_x - 8

        init_y = 0
        init_x = 0

        return map_state, freecells, init_y, init_x

class GraphBasedPPEnv(Env):

    def __init__(self, map_type):
        super().__init__()
        
        self.observation_space = MultiDiscrete(np.append(np.array([dimcells_y, dimcells_x, dimcells_y*dimcells_x]), 3*np.ones((dimcells_y, dimcells_x)).ravel()), start=np.append(np.array([0, 0, 1]), -np.ones((dimcells_y, dimcells_x)).ravel()))

        self.action_space = Discrete(4)

        self.map_type = map_type

        self.map_state, self.freecells, init_y, init_x = set_map(self.map_type)

        self.map_state[init_y, init_x] = 1

        self.position = np.array([init_y, init_x])

        self.visited_cells = 1

        self.previous_visited_cells = 1

        self.previous_action = -1

        self.full_state = np.append(np.append(self.position, self.visited_cells), self.map_state.ravel())

        self.Nstep = 0

    def step(self, action):
        
        # 0 = north
        # 1 = east
        # 2 = south
        # 3 = west

        reward = 0

        terminated = False

        truncated = False

        self.Nstep += 1

        reward -= 0.1 * self.Nstep

        if action == 0:

            if self.position[0] < dimcells_y - 1:

                # print(self.position[1])
                if self.map_state[self.position[0] + 1, self.position[1]] != 2:
                    # moving on a free cell
                    self.position[0] += 1
                    reward += 0.01
                    if self.map_state[self.position[0], self.position[1]] == 0:
                        # rewarded if moving on an unvisited cell
                        self.map_state[self.position[0], self.position[1]] = 1
                        self.visited_cells += 1
                        reward += 0.1
                else:
                    # collision with obstacles
                    reward -= 10

            else:
                # out of boundaries
                reward -= 10
        
        elif action == 1:

            if self.position[1] < dimcells_x - 1:

                if self.map_state[self.position[0], self.position[1] + 1] != 2:
                    # moving on a free cell
                    self.position[1] += 1
                    reward += 0.01
                    if self.map_state[self.position[0], self.position[1]] == 0:
                        # rewarded if moving on an unvisited cell
                        self.map_state[self.position[0], self.position[1]] = 1
                        self.visited_cells += 1
                        reward += 0.1
                else:
                    # collision with obstacles
                    reward -= 10

            else:
                # out of boundaries
                reward -= 10
        
        elif action == 2:

            if self.position[0] > 0:

                if self.map_state[self.position[0] - 1, self.position[1]] != 2:
                    # moving on a free cell
                    self.position[0] -= 1
                    reward += 0.01
                    if self.map_state[self.position[0], self.position[1]] == 0:
                        # rewarded if moving on an unvisited cell
                        self.map_state[self.position[0], self.position[1]] = 1
                        self.visited_cells += 1
                        reward += 0.1
                else:
                    # collision with obstacles
                    reward -= 10

            else:
                # out of boundaries
                reward -= 10
        
        elif action == 3:

            if self.position[1] > 0:

                if self.map_state[self.position[0], self.position[1] - 1] != 2:
                    # moving on a free cell
                    self.position[1] -= 1
                    reward += 0.01
                    if self.map_state[self.position[0], self.position[1]] == 0:
                        # rewarded if moving on an unvisited cell
                        self.map_state[self.position[0], self.position[1]] = 1
                        self.visited_cells += 1
                        reward += 0.1
                else:
                    # collision with obstacles
                    reward -= 10

            else:
                # out of boundaries
                reward -= 10
        
        if self.previous_action != action and self.previous_action != -1:
            reward -= 0.1
        else:
            reward += 0.01

        self.previous_action = action

        self.previous_visited_cells = self.visited_cells

        if self.visited_cells == self.freecells:
            reward += 10
        
        if self.Nstep == 400:
            truncated = True
        
        if self.visited_cells == self.freecells:
            terminated = True

        info = {}

        self.full_state = np.append(np.append(self.position, self.visited_cells), self.map_state.ravel())

        return self.full_state, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.map_state, _, init_y, init_x = set_map(self.map_type)

        self.map_state[init_y, init_x] = 1

        self.position = np.array([init_y, init_x])

        self.visited_cells = 1

        self.previous_visited_cells = 1

        self.previous_action = -1

        self.Nstep = 0
        
        self.full_state = np.append(np.append(self.position, self.visited_cells), self.map_state.ravel())

        info = {}

        return self.full_state, info

def make_env(map_type):
    def _init():
        env = GraphBasedPPEnv(map_type)
        return Monitor(env)
    return _init

def train():

    map_list = ["empty", "L", "triangle", "cross", "hole"]

    env_fns = [make_env(m) for m in map_list]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy",
                vec_env,
                verbose=1,
                normalize_advantage=False,
                ent_coef=0.001,
                vf_coef=0.05,
                tensorboard_log= "Training/tensorboard_log/PPO_GraphBased/",
    )

    model.learn(total_timesteps=10000000, progress_bar=True)

    model.save("Training/saved_models/PPO_GraphBased/PPO_2M_5maps.zip")

def inference():
    env = DummyVecEnv([lambda: GraphBasedPPEnv(map_type="empty")])

    model = PPO.load("Training/saved_models/PPO_GraphBased/PPO_2M_rect5x3_diffdirpenalty_randinitpt.zip")

    episodes = 1
    for episode in range(episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        count = 0
        print(count)
        print_map(obs[0, 3:].reshape(dimcells_y, dimcells_x))

        while not terminated:
            count += 1
            print(count)
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, *info = env.step(action)
            # print("Reward:\t", reward)
            print_map(obs[0, 3:].reshape(dimcells_y, dimcells_x), action)


if __name__ == '__main__':
        
    train()
    # inference()
        