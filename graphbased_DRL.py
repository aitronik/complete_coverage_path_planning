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

from PIL import Image, ImageDraw
import imageio.v2 as imageio
import os
import shutil

dimcells_x = 8
dimcells_y = 12

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

class GraphBasedPPEnv(Env):

    def __init__(self):
        super().__init__()
        
        self.observation_space = MultiDiscrete(np.append(np.array([dimcells_y, dimcells_x, dimcells_y*dimcells_x]), 3*np.ones((dimcells_y, dimcells_x)).ravel()), start=np.append(np.array([0, 0, 1]), np.zeros((dimcells_y, dimcells_x)).ravel()))

        self.action_space = Discrete(4)

        self.map_state = np.zeros((dimcells_y, dimcells_x))
        self.map_state[0, :] = 2
        self.map_state[:, 0] = 2
        self.map_state[dimcells_y - 1, :] = 2
        self.map_state[:, dimcells_x - 1] = 2

        freecells = np.count_nonzero(self.map_state == 0)
        coords = np.column_stack(np.where(self.map_state == 0))

        self.freecells = freecells

        init_y, init_x = coords[np.random.choice(coords.shape[0])]

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

        # reward -= 0.1 * self.Nstep
        reward -= 0.1

        if action == 0:

            if self.position[0] < dimcells_y - 1:

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
        
        if self.Nstep == 500:
            truncated = True
        
        if self.visited_cells == self.freecells:
            terminated = True

        info = {}

        self.full_state = np.append(np.append(self.position, self.visited_cells), self.map_state.ravel())

        # if terminated == True:
        #     get_output(self.map_state, 0, self.Nstep, self.position)

        return self.full_state, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.map_state = np.zeros((dimcells_y, dimcells_x))
        self.map_state[0, :] = 2
        self.map_state[:, 0] = 2
        self.map_state[dimcells_y - 1, :] = 2
        self.map_state[:, dimcells_x - 1] = 2

        freecells = np.count_nonzero(self.map_state == 0)
        coords = np.column_stack(np.where(self.map_state == 0))

        self.freecells = freecells

        init_y, init_x = coords[np.random.choice(coords.shape[0])]

        self.map_state[init_y, init_x] = 1

        self.position = np.array([init_y, init_x])

        self.visited_cells = 1

        self.previous_visited_cells = 1

        self.previous_action = -1

        self.Nstep = 0
        
        self.full_state = np.append(np.append(self.position, self.visited_cells), self.map_state.ravel())

        info = {}

        return self.full_state, info

def get_output(matrix=2*np.ones((dimcells_y, dimcells_x)), episode=0, step=0, position=np.zeros(2)):

    cols = dimcells_x
    rows = dimcells_y
    cell_size = 40

    os.makedirs("output_pngs/", exist_ok=True)

    img = Image.new("RGB", (cols * cell_size, rows * cell_size), "white")
    draw = ImageDraw.Draw(img)

    color_map = {
        0: (255, 255, 255),
        1: (0, 255, 0),
        2: (128, 128, 128)
    }

    for i in range(rows):
        for j in range(cols):
            value = matrix[i][j]
            color = color_map.get(value, (255, 0, 0))
            x0 = j * cell_size
            y0 = i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(200, 200, 200))
    
    x0 = position[1] * cell_size
    y0 = position[0] * cell_size
    x1 = x0 + cell_size
    y1 = y0 + cell_size
    draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0), outline=(200, 200, 200))

    img.save("output_pngs/step_" + str(step) + ".png")

def get_video():

    png_files = [f for f in os.listdir("output_pngs/") if f.endswith('.png')]

    images = [imageio.imread(f'output_pngs/step_{i}.png') for i in range(len(png_files))]

    i = 0
    for image in images:
        img = Image.fromarray(image)
        
        new_width = img.width * 10
        new_height = img.height * 10
        
        images[i] = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        i += 1

    imageio.mimsave('output_video.mp4', images, fps=2, macro_block_size=None)

    shutil.rmtree("output_pngs/")

def train():
    env = Monitor(GraphBasedPPEnv())

    model = PPO("MlpPolicy",
                env,
                verbose=1,
                # gamma=0.999,
                # normalize_advantage=False,
                # ent_coef=0.001,
                # vf_coef=0.05,
                tensorboard_log= "Training/tensorboard_log/PPO_GraphBased/",
    )

    model.learn(total_timesteps=3_000_000, progress_bar=True)

    model.save("Training/saved_models/PPO_GraphBased/PPO_3M_rect10x6_framed_randinitpt_ALLdefault.zip")

def inference():
    env = DummyVecEnv([lambda: GraphBasedPPEnv()])

    model = PPO.load("Training/saved_models/PPO_GraphBased/PPO_1M_rect10x6_framed_randinitpt_ALLdefault.zip")

    episodes = 1
    for episode in range(episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        count = 0
        # print(count)
        # print_map(obs[0, 3:].reshape(dimcells_y, dimcells_x))
        get_output(obs[0, 3:].reshape(dimcells_y, dimcells_x), episode, count, obs[0, 0:2])

        while not terminated:
            count += 1
            # print(count)
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, *info = env.step(action)
            # print("Reward:\t", reward)
            # print_map(obs[0, 3:].reshape(dimcells_y, dimcells_x), action)
            if terminated == False:
                get_output(obs[0, 3:].reshape(dimcells_y, dimcells_x), episode, count, obs[0, 0:2])
        
        get_video()

def train_loadedmodel():
    env = Monitor(GraphBasedPPEnv())

    model = PPO.load("Training/saved_models/PPO_GraphBased/PPO_500k_rect10x6_framed_randinitpt_ALLdefault.zip", env=env)

    model.learn(total_timesteps=500000,
                progress_bar=True,
                tb_log_name="PPO_1M_rect10x6_framed_randinitpt_ALLdefault",
                reset_num_timesteps=False,
    )

    model.save("Training/saved_models/PPO_GraphBased/PPO_1M_rect10x6_framed_randinitpt_ALLdefault.zip")

if __name__ == '__main__':

    train()
    # inference()
    # train_loadedmodel()
    pass
        