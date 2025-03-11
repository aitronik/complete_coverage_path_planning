import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

from stable_baselines3 import TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

acc = 0.2
dec = 0.2
vmax = 0.55

k_cov = 1
k_time = 1

fp_width = 0.5

def segmentTime(l, a=acc, d=dec, vmax=vmax):
    t_acc = vmax / a
    
    t_dec = vmax / d
    
    l_acc = (vmax * vmax) / (2 * a)
    
    l_dec = (vmax * vmax) / (2 * d)
    
    l_max = l_acc + l_dec
    
    if l_max >= l:
        return np.sqrt((2 * l) / a) + np.sqrt((2 * l) / d)
    else:
        t_costante = (l - l_max) / vmax        
        return t_acc + t_dec + t_costante

def deg2rad(alpha):
    return alpha * np.pi / 180

with open("sw_input_logs/input_per_1.txt", "r") as f:
    header = list(map(float, f.readline().strip().split("\t")))

area = header.pop()
min_time = header.pop()
header = np.array(header).astype(int)

data = np.genfromtxt("sw_input_logs/input_per_1.txt", delimiter='\t', skip_header=1)

y_bbox = np.mean(np.sum(np.abs(data[0:header[1], :]), axis=1)[np.sum(np.abs(data[0:header[1], :]), axis=1) != 0])
x_bbox = np.mean(np.sum(np.abs(data[int(header[0]*header[1]/2):int(header[0]*header[1]/2 + header[1]), :]), axis=1)[np.sum(np.abs(data[int(header[0]*header[1]/2):int(header[0]*header[1]/2 + header[1]), :]), axis=1) != 0])

# print(y_bbox)
# print(x_bbox)

# print(not np.all(data[100, :] == 0))

tmp_data = np.sum(np.abs(data), axis=1)

max_sw_len = max(tmp_data)

input_data = data.ravel()

max_len = max(input_data)
min_len = min(input_data)

#print(data.shape)

n_input = np.prod(header)
n_output = 4000

init_pts = np.ones((header[0] * header[1], 2))*-1

for i in range(header[0]):
    
    for j in range(header[1]):

        if i == 0:

            if not np.all(data[header[1] * i + j, :] == 0):
                init_pts[header[1] * i + j, :] = np.array([fp_width/2 + j * fp_width, y_bbox])
            
            # else:
            #     init_pts[header[1] * i + j, :] = np.array([-1, -1])
        
        elif i == header[0]/2:

            if not np.all(data[header[1] * i + j, :] == 0):
                init_pts[header[1] * i + j, :] = np.array([0, fp_width/2 + j * fp_width])
            
            # else:
            #     init_pts[header[1] * i + j, :] = np.array([-1, -1])
        
        elif i > 0 and i < header[0]/2:

            if not np.all(data[header[1] * i + j, :] == 0):

                if (fp_width/2 + j * fp_width)/np.sin(deg2rad(i * 180 / header[0])) <= y_bbox:
                    init_pts[header[1] * i + j, :] = np.array([0, (fp_width/2 + j * fp_width)/np.sin(deg2rad(i * 180 / header[0]))])

                else:
                    init_pts[header[1] * i + j, :] = np.array([np.tan(deg2rad(i * 180 / header[0])) * ((fp_width/2 + j * fp_width)/np.sin(deg2rad(i * 180 / header[0])) - y_bbox), y_bbox])
            
            # else:
            #     init_pts[header[1] * i + j, :] = np.array([-1, -1])
        
        elif i > header[0]/2 and i < header[0]:

            if not np.all(data[header[1] * i + j, :] == 0):

                if (fp_width/2 + j * fp_width)/np.cos(np.pi - deg2rad(i * 180 / header[0])) <= x_bbox:
                    init_pts[header[1] * i + j, :] = np.array([x_bbox - (fp_width/2 + j * fp_width)/np.cos(np.pi - deg2rad(i * 180 / header[0])), 0])

                else:
                    init_pts[header[1] * i + j, :] = np.array([0, np.tan(deg2rad(i * 180 / header[0]) - np.pi/2) * ((fp_width/2 + j * fp_width)/np.cos(np.pi - deg2rad(i * 180 / header[0])) - x_bbox)])
                        
            # else:
            #     init_pts[header[1] * i + j, :] = np.array([-1, -1])

# plt.scatter(init_pts[:, 0], init_pts[:, 1])
# plt.show()

def check_intersections(action_sample, action, x_bbox=x_bbox, y_bbox=y_bbox):

    inter_count = 0

    for i in range(action.shape[1]):
        pass

    return

def plot_action(action_sample, init_pts=init_pts):

    action_sample = action_sample.T

    if action_sample[0] == 0:
        if init_pts[header[1] * action_sample[0] + action_sample[1]] is not -np.ones(2):
            x = [init_pts[header[1] * action_sample[0] + action_sample[1], 0], init_pts[header[1] * action_sample[0] + action_sample[1], 0]]
            y = [init_pts[header[1] * action_sample[0] + action_sample[1], 1] - np.abs(action_sample[2]), init_pts[header[1] * action_sample[0] + action_sample[1], 1] - np.abs(action_sample[2]) - action_sample[3]]
            # plt.plot(x, y)

    elif action_sample[0] == header[0]/2:
        if init_pts[header[1] * action_sample[0] + action_sample[1]] is not -np.ones(2):
            x = [init_pts[header[1] * action_sample[0] + action_sample[1], 0] + np.abs(action_sample[2]), init_pts[header[1] * action_sample[0] + action_sample[1], 0] + np.abs(action_sample[2]) + action_sample[3]]
            y = [init_pts[header[1] * action_sample[0] + action_sample[1], 1], init_pts[header[1] * action_sample[0] + action_sample[1], 1]]
            # plt.plot(x, y)

    else:
        if init_pts[header[1] * action_sample[0] + action_sample[1]] is not -np.ones(2):
            x = [init_pts[header[1] * action_sample[0] + action_sample[1], 0] + np.cos(deg2rad(action_sample[0] * 180 / header[0])) * np.abs(action_sample[2]), init_pts[header[1] * action_sample[0] + action_sample[1], 0] + np.cos(deg2rad(action_sample[0] * 180 / header[0])) * (np.abs(action_sample[2] + action_sample[3]))]
            y = [init_pts[header[1] * action_sample[0] + action_sample[1], 1] + np.sin(deg2rad(action_sample[0] * 180 / header[0])) * np.abs(action_sample[2]), init_pts[header[1] * action_sample[0] + action_sample[1], 1] + np.sin(deg2rad(action_sample[0] * 180 / header[0])) * (np.abs(action_sample[2] + action_sample[3]))]
            # plt.plot(x, y)

    return

class PathPlanningNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Space, features_dim: int):
        super(PathPlanningNN, self).__init__(observation_space, features_dim)

        self.fc1 = nn.Linear(n_input, 4096)
        self.fc2 = nn.Linear(4096, features_dim)
        self.flatten = nn.Flatten()
    
    def forward(self, x):

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        output = F.tanh(x)

        return output

class PathPlanningEnv(Env):

    def __init__(self):
        super().__init__()
        
        self.action_space = Box(low=-1, high=1, shape=(4, int(n_output//4)))
        
        self.observation_space = Box(low=min_len, high=max_len, shape=(input_data.shape))

        self.observation = input_data
        
        self.current_action = None

        self.Nstep = 1000

    def step(self, action):

        # print(self.Nstep)

        action[0, :] = np.floor(action[0, :] * header[0]/2 * 0.999 + header[0]/2).astype(int)
        action[1, :] = np.floor(action[1, :] * header[1]/2 * 0.999 + header[1]/2).astype(int)
        action[2, :] = action[2, :] * max_sw_len/2 - max_sw_len/2
        action[3, :] = action[3, :] * max_len/2 + max_len/2

        self.current_action = action

        # print(action[:, 0:5])

        self.Nstep -= 1

        reward = 0
        reward_time = 0

        terminated = False
        
        truncated = False

        flag_stop = False
        
        info = {}

        for i in range(self.action_space.shape[1]):
            
            if action[3, i] == 0:
                flag_stop = True
                break
            
            row = int(action[0, i] * header[0] + action[1, i])

            # if not np.all(data[row, :] == 0):

            if np.abs(action[2, i]) <= np.sum(np.abs(data[row, :])):

                cont = 0

                for j in np.where(data[row, :] > 0)[0]:

                    if np.sum(np.abs(data[row, j if j == 0 else slice(0, j)])) <= np.abs(action[2, i]) and np.sum(np.abs(data[row, 0:(j+1)])) >= np.abs(action[2, i]) + np.abs(action[3, i]):
                        # reward += k_cov * (1 - np.abs((1 - fp_width * action[3, i] / area)))
                        # reward += k_cov * np.exp(- (1 - fp_width * action[3, i] / area)**2)
                        # reward += fp_width * action[3, i] / area
                        reward += 10 * k_cov
                        reward_time += segmentTime(action[3, i])
                    
                    else:
                        cont += 1
                
                if cont == np.where(data[row, :] > 0)[0].shape[0]:
                    reward += -k_cov
            
            else:
                reward += -2*k_cov
            
            # else:
            #     reward += -10 * k_cov
        
        
        # reward = k_cov * (1 - np.abs((1 - reward)))
        reward_time = k_time * (min_time - reward_time)/min_time
        # reward += reward_time
                        
        if self.Nstep == 0 or flag_stop == True:
            terminated = True
        else:
            terminated = False
        
        # print(reward)

        return self.observation, reward, terminated, truncated, info
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.observation = input_data

        self.current_action = None

        self.Nstep = 1000

        info = {}

        return self.observation, info

    def render(self, mode="human", close=False):

        img = cv2.imread("immagini/perimeters/per_1.png", cv2.IMREAD_GRAYSCALE)

        bbox_center = [x_bbox/2, y_bbox/2]

        img_scalefactor = int(1000/(1.2*np.max(x_bbox, y_bbox)))
            
        for i in range(self.current_action.shape[1]):
            if self.current_action[0, i] == 0:
                if init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i]] is not -np.ones(2):
                    x = [init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 0], init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 0]]
                    y = [init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 1] - np.abs(self.current_action[2, i]), init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 1] - np.abs(self.current_action[2, i]) - self.current_action[3, i]]
                    x_img = int(img_scalefactor * (x + (500/img_scalefactor - bbox_center[0])))
                    y_img = 1000 - int(img_scalefactor * (y + (500/img_scalefactor - bbox_center[1])))
                    cv2.line(img, tuple(x_img[0], y_img[0]), tuple(x_img[1], y_img[1]), color=128, thickness=2)

            elif self.current_action[0, i] == header[0]/2:
                if init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i]] is not -np.ones(2):
                    x = [init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 0] + np.abs(self.current_action[2, i]), init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 0] + np.abs(self.current_action[2, i]) + self.current_action[3, i]]
                    y = [init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 1], init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 1]]
                    x_img = int(img_scalefactor * (x + (500/img_scalefactor - bbox_center[0])))
                    y_img = 1000 - int(img_scalefactor * (y + (500/img_scalefactor - bbox_center[1])))
                    cv2.line(img, tuple(x_img[0], y_img[0]), tuple(x_img[1], y_img[1]), color=128, thickness=2)

            else:
                if init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i]] is not -np.ones(2):
                    x = [init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 0] + np.cos(deg2rad(self.current_action[0, i] * 180 / header[0])) * np.abs(self.current_action[2, i]), init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 0] + np.cos(deg2rad(self.current_action[0, i] * 180 / header[0])) * (np.abs(self.current_action[2, i] + self.current_action[3, i]))]
                    y = [init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 1] + np.sin(deg2rad(self.current_action[0, i] * 180 / header[0])) * np.abs(self.current_action[2, i]), init_pts[header[1] * self.current_action[0, i] + self.current_action[1, i], 1] + np.sin(deg2rad(self.current_action[0, i] * 180 / header[0])) * (np.abs(self.current_action[2, i] + self.current_action[3, i]))]
                    x_img = int(img_scalefactor * (x + (500/img_scalefactor - bbox_center[0])))
                    y_img = 1000 - int(img_scalefactor * (y + (500/img_scalefactor - bbox_center[1])))
                    cv2.line(img, tuple(x_img[0], y_img[0]), tuple(x_img[1], y_img[1]), color=128, thickness=2)
        
        cv2.imshow("Action", img)
        cv2.destroyAllWindows()


policy_kwargs = dict(
    # activation_fn= nn.Tanh,
    net_arch = [1500, 1500],
    # features_extractor_class=PathPlanningNN,
    # features_extractor_kwargs=dict(features_dim=512),
)

env = DummyVecEnv([lambda: PathPlanningEnv()])

# model = TD3("MlpPolicy",
#             env,
#             #policy_kwargs=policy_kwargs,
#             verbose=1,
#             buffer_size=10000,
#             #policy_delay=1,
#             #batch_size=128,
#             #learning_starts=200,
#             #tensorboard_log= "Training/tensorboard_log/",
#             #action_noise=NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=1 * np.ones(env.action_space.shape))
# )

# con output = 4000: massimo buffer_size = 10000
model = SAC("MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            buffer_size=10000,
            #batch_size=128,
            #learning_starts=200,
            tensorboard_log= "Training/tensorboard_log/",
)

model.learn(total_timesteps=10000,
            progress_bar=True,
)

model.save("Training/saved_models/SAC/SAC_10k")

# del model

# model = SAC.load("Training/saved_models/SAC/SAC_10k.zip")

# episodes = 2
# for episode in range(1, episodes+1):
#     obs = env.reset()
#     terminated = False

#     while not terminated:
#         action, _ = model.predict(obs)
#         obs, reward, terminated, truncated, *info = env.step(action)
#         env.render(mode="human")

env.close()