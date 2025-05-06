import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

dimcells_x = 8
dimcells_y = 12

maps = ["empty", "L", "cross", "hole", "triangle"]

def get_area(name = "empty"):

    area = np.zeros((dimcells_y, dimcells_x))
    area[0, :] = 2
    area[:, 0] = 2
    area[-1, :] = 2
    area[:, -1] = 2

    if name == "empty":
        return area
    
    elif name == "L":
        area[4:, 4:] = 2
        return area

    elif name == "cross":
        area[1:4, 1:3] = 2
        area[8:12, 1:3] = 2
        area[8:12, 5:8] = 2
        area[1:4, 5:8] = 2
        return area
    
    elif name == "hole":
        area[4:8, 3:5] = 2
        return area
    
    elif name == "triangle":
        area = np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 0, 0, 0, 0, 0, 0, 2],
                         [2, 0, 0, 0, 0, 0, 2, 2],
                         [2, 0, 0, 0, 0, 0, 2, 2],
                         [2, 0, 0, 0, 0, 2, 2, 2],
                         [2, 0, 0, 0, 0, 2, 2, 2],
                         [2, 0, 0, 0, 2, 2, 2, 2],
                         [2, 0, 0, 0, 2, 2, 2, 2],
                         [2, 0, 0, 2, 2, 2, 2, 2],
                         [2, 0, 0, 2, 2, 2, 2, 2],
                         [2, 0, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2]], dtype=float)
        return area
    
    else:
        print("Wrong map_code passed! Returned empty by default")
        return area

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

# print_map(get_area("empty"))

class GraphBasedPPEnv(Env):

    def __init__(self):
        super().__init__()
        
        self.observation_space = MultiDiscrete(np.append(np.array([dimcells_y, dimcells_x, dimcells_y*dimcells_x]), 3*np.ones((dimcells_y, dimcells_x)).ravel()), start=np.append(np.array([0, 0, 1]), np.zeros((dimcells_y, dimcells_x)).ravel()))

        self.action_space = Discrete(4)

        self.current_step = 0

        self.map_state = [None] * len(maps)
        self.freecells = [None] * len(maps)
        coords = [None] * len(maps)
        init_y = [None] * len(maps)
        init_x = [None] * len(maps)
        self.position = [None] * len(maps)
        self.visited_cells = [None] * len(maps)
        self.previous_visited_cells = [None] * len(maps)
        self.previous_action = [None] * len(maps)
        self.full_state = [None] * len(maps)
        self.Nstep = [None] * len(maps)

        self.rewards = [None] * len(maps)
        self.terminated = [None] * len(maps)
        self.truncated = [None] * len(maps)

        self.map_state[0] = get_area(maps[0])
        self.map_state[1] = get_area(maps[1])
        self.map_state[2] = get_area(maps[2])
        self.map_state[3] = get_area(maps[3])
        self.map_state[4] = get_area(maps[4])

        self.freecells[0] = np.count_nonzero(self.map_state[0] == 0)
        coords[0] = np.column_stack(np.where(self.map_state[0] == 0))
        self.freecells[1] = np.count_nonzero(self.map_state[1] == 0)
        coords[1] = np.column_stack(np.where(self.map_state[1] == 0))
        self.freecells[2] = np.count_nonzero(self.map_state[2] == 0)
        coords[2] = np.column_stack(np.where(self.map_state[2] == 0))
        self.freecells[3] = np.count_nonzero(self.map_state[3] == 0)
        coords[3] = np.column_stack(np.where(self.map_state[3] == 0))
        self.freecells[4] = np.count_nonzero(self.map_state[4] == 0)
        coords[4] = np.column_stack(np.where(self.map_state[4] == 0))


        init_y[0], init_x[0] = coords[0][np.random.choice(coords[0].shape[0])]
        init_y[1], init_x[1] = coords[1][np.random.choice(coords[1].shape[0])]
        init_y[2], init_x[2] = coords[2][np.random.choice(coords[2].shape[0])]
        init_y[3], init_x[3] = coords[3][np.random.choice(coords[3].shape[0])]
        init_y[4], init_x[4] = coords[4][np.random.choice(coords[4].shape[0])]

        self.map_state[0][init_y[0], init_x[0]] = 1
        self.map_state[1][init_y[1], init_x[1]] = 1
        self.map_state[2][init_y[2], init_x[2]] = 1
        self.map_state[3][init_y[3], init_x[3]] = 1
        self.map_state[4][init_y[4], init_x[4]] = 1

        self.position[0] = np.array([init_y[0], init_x[0]])
        self.position[1] = np.array([init_y[1], init_x[1]])
        self.position[2] = np.array([init_y[2], init_x[2]])
        self.position[3] = np.array([init_y[3], init_x[3]])
        self.position[4] = np.array([init_y[4], init_x[4]])

        self.visited_cells[0] = 1
        self.visited_cells[1] = 1
        self.visited_cells[2] = 1
        self.visited_cells[3] = 1
        self.visited_cells[4] = 1

        self.previous_visited_cells[0] = 1
        self.previous_visited_cells[1] = 1
        self.previous_visited_cells[2] = 1
        self.previous_visited_cells[3] = 1
        self.previous_visited_cells[4] = 1

        self.previous_action[0] = -1
        self.previous_action[1] = -1
        self.previous_action[2] = -1
        self.previous_action[3] = -1
        self.previous_action[4] = -1

        self.full_state[0] = np.append(np.append(self.position[0], self.visited_cells[0]), self.map_state[0].ravel())
        self.full_state[1] = np.append(np.append(self.position[1], self.visited_cells[1]), self.map_state[1].ravel())
        self.full_state[2] = np.append(np.append(self.position[2], self.visited_cells[2]), self.map_state[2].ravel())
        self.full_state[3] = np.append(np.append(self.position[3], self.visited_cells[3]), self.map_state[3].ravel())
        self.full_state[4] = np.append(np.append(self.position[4], self.visited_cells[4]), self.map_state[4].ravel())

        self.Nstep[0] = 0
        self.Nstep[1] = 0
        self.Nstep[2] = 0
        self.Nstep[3] = 0
        self.Nstep[4] = 0

    def step(self, action):
        
        # 0 = north
        # 1 = east
        # 2 = south
        # 3 = west

        reward = 0

        terminated = False

        truncated = False

        a = self.current_step % len(maps)

        self.current_step += 1

        self.Nstep[a] += 1

        # reward -= 0.1 * self.Nstep
        reward -= 0.1

        if action == 0:

            if self.position[a][0] < dimcells_y - 1:

                if self.map_state[a][self.position[a][0] + 1, self.position[a][1]] != 2:
                    # moving on a free cell
                    self.position[a][0] += 1
                    reward += 0.01
                    if self.map_state[a][self.position[a][0], self.position[a][1]] == 0:
                        # rewarded if moving on an unvisited cell
                        self.map_state[a][self.position[a][0], self.position[a][1]] = 1
                        self.visited_cells[a] += 1
                        reward += 0.1
                else:
                    # collision with obstacles
                    reward -= 10

            else:
                # out of boundaries
                reward -= 10
        
        elif action == 1:

            if self.position[a][1] < dimcells_x - 1:

                if self.map_state[a][self.position[a][0], self.position[a][1] + 1] != 2:
                    # moving on a free cell
                    self.position[a][1] += 1
                    reward += 0.01
                    if self.map_state[a][self.position[a][0], self.position[a][1]] == 0:
                        # rewarded if moving on an unvisited cell
                        self.map_state[a][self.position[a][0], self.position[a][1]] = 1
                        self.visited_cells[a] += 1
                        reward += 0.1
                else:
                    # collision with obstacles
                    reward -= 10

            else:
                # out of boundaries
                reward -= 10
        
        elif action == 2:

            if self.position[a][0] > 0:

                if self.map_state[a][self.position[a][0] - 1, self.position[a][1]] != 2:
                    # moving on a free cell
                    self.position[a][0] -= 1
                    reward += 0.01
                    if self.map_state[a][self.position[a][0], self.position[a][1]] == 0:
                        # rewarded if moving on an unvisited cell
                        self.map_state[a][self.position[a][0], self.position[a][1]] = 1
                        self.visited_cells[a] += 1
                        reward += 0.1
                else:
                    # collision with obstacles
                    reward -= 10

            else:
                # out of boundaries
                reward -= 10
        
        elif action == 3:

            if self.position[a][1] > 0:

                if self.map_state[a][self.position[a][0], self.position[a][1] - 1] != 2:
                    # moving on a free cell
                    self.position[a][1] -= 1
                    reward += 0.01
                    if self.map_state[a][self.position[a][0], self.position[a][1]] == 0:
                        # rewarded if moving on an unvisited cell
                        self.map_state[a][self.position[a][0], self.position[a][1]] = 1
                        self.visited_cells[a] += 1
                        reward += 0.1
                else:
                    # collision with obstacles
                    reward -= 10

            else:
                # out of boundaries
                reward -= 10
        
        if self.previous_action[a] != action and self.previous_action[a] != -1:
            reward -= 0.1
        else:
            reward += 0.01

        self.previous_action[a] = action

        self.previous_visited_cells[a] = self.visited_cells[a]

        if self.visited_cells[a] == self.freecells[a]:
            reward += 10
        
        if self.Nstep[a] == 500:
            truncated = True
        
        if self.visited_cells[a] == self.freecells[a]:
            terminated = True

        info = {}

        self.full_state[a] = np.append(np.append(self.position[a], self.visited_cells[a]), self.map_state[a].ravel())

        return self.full_state[a], reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.map_state = [None] * 5
        self.freecells = [None] * 5
        coords = [None] * 5
        init_y = [None] * 5
        init_x = [None] * 5
        self.position = [None] * 5
        self.visited_cells = [None] * 5
        self.previous_visited_cells = [None] * 5
        self.previous_action = [None] * 5
        self.full_state = [None] * 5
        self.Nstep = [None] * 5

        self.map_state[0] = get_area(maps[0])
        self.map_state[1] = get_area(maps[1])
        self.map_state[2] = get_area(maps[2])
        self.map_state[3] = get_area(maps[3])
        self.map_state[4] = get_area(maps[4])

        self.freecells[0] = np.count_nonzero(self.map_state[0] == 0)
        coords[0] = np.column_stack(np.where(self.map_state[0] == 0))
        self.freecells[1] = np.count_nonzero(self.map_state[1] == 0)
        coords[1] = np.column_stack(np.where(self.map_state[1] == 0))
        self.freecells[2] = np.count_nonzero(self.map_state[2] == 0)
        coords[2] = np.column_stack(np.where(self.map_state[2] == 0))
        self.freecells[3] = np.count_nonzero(self.map_state[3] == 0)
        coords[3] = np.column_stack(np.where(self.map_state[3] == 0))
        self.freecells[4] = np.count_nonzero(self.map_state[4] == 0)
        coords[4] = np.column_stack(np.where(self.map_state[4] == 0))


        init_y[0], init_x[0] = coords[0][np.random.choice(coords[0].shape[0])]
        init_y[1], init_x[1] = coords[1][np.random.choice(coords[1].shape[0])]
        init_y[2], init_x[2] = coords[2][np.random.choice(coords[2].shape[0])]
        init_y[3], init_x[3] = coords[3][np.random.choice(coords[3].shape[0])]
        init_y[4], init_x[4] = coords[4][np.random.choice(coords[4].shape[0])]

        self.map_state[0][init_y[0], init_x[0]] = 1
        self.map_state[1][init_y[1], init_x[1]] = 1
        self.map_state[2][init_y[2], init_x[2]] = 1
        self.map_state[3][init_y[3], init_x[3]] = 1
        self.map_state[4][init_y[4], init_x[4]] = 1

        self.position[0] = np.array([init_y[0], init_x[0]])
        self.position[1] = np.array([init_y[1], init_x[1]])
        self.position[2] = np.array([init_y[2], init_x[2]])
        self.position[3] = np.array([init_y[3], init_x[3]])
        self.position[4] = np.array([init_y[4], init_x[4]])

        self.visited_cells[0] = 1
        self.visited_cells[1] = 1
        self.visited_cells[2] = 1
        self.visited_cells[3] = 1
        self.visited_cells[4] = 1

        self.previous_visited_cells[0] = 1
        self.previous_visited_cells[1] = 1
        self.previous_visited_cells[2] = 1
        self.previous_visited_cells[3] = 1
        self.previous_visited_cells[4] = 1

        self.previous_action[0] = -1
        self.previous_action[1] = -1
        self.previous_action[2] = -1
        self.previous_action[3] = -1
        self.previous_action[4] = -1

        self.full_state[0] = np.append(np.append(self.position[0], self.visited_cells[0]), self.map_state[0].ravel())
        self.full_state[1] = np.append(np.append(self.position[1], self.visited_cells[1]), self.map_state[1].ravel())
        self.full_state[2] = np.append(np.append(self.position[2], self.visited_cells[2]), self.map_state[2].ravel())
        self.full_state[3] = np.append(np.append(self.position[3], self.visited_cells[3]), self.map_state[3].ravel())
        self.full_state[4] = np.append(np.append(self.position[4], self.visited_cells[4]), self.map_state[4].ravel())

        self.Nstep[0] = 0
        self.Nstep[1] = 0
        self.Nstep[2] = 0
        self.Nstep[3] = 0
        self.Nstep[4] = 0

        info = {}

        return self.full_state[0], info

def train():
    env = Monitor(GraphBasedPPEnv())

    model = PPO("MlpPolicy",
                env,
                verbose=1,
                normalize_advantage=False,
                ent_coef=0.001,
                vf_coef=0.05,
                # tensorboard_log= "Training/tensorboard_log/PPO_GraphBased/",
    )

    model.learn(total_timesteps=500000, progress_bar=True)

    # model.save("Training/saved_models/PPO_GraphBased/PPO_500k_rect10x6_framed_randinitpt.zip")

def inference():
    env = DummyVecEnv([lambda: GraphBasedPPEnv()])

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

    # train()
    # inference()
    pass