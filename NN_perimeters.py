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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import update_learning_rate

from sb3_contrib import TQC

acc = 0.2
dec = 0.2
vmax = 0.55

k_cov = 1
k_time = 1

fp_width = 0.5

n_per = "triangle"
angle_step = "30"

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

with open("sw_input_logs/input_per_" + n_per + "_anglestep" + angle_step + "°_onlymask0°.txt", "r") as f:
    header = list(map(float, f.readline().strip().split("\t")))

x_bbox = header.pop()
y_bbox = header.pop()
area = header.pop()
min_time = header.pop()
header = np.array(header).astype(int)

data = np.genfromtxt("sw_input_logs/input_per_" + n_per + "_anglestep" + angle_step + "°_onlymask0°.txt", delimiter='\t', skip_header=1)

# y_bbox = np.mean(np.sum(np.abs(data[0:header[1], :]), axis=1)[np.sum(np.abs(data[0:header[1], :]), axis=1) != 0])
# x_bbox = np.mean(np.sum(np.abs(data[int(header[0]*header[1]/2):int(header[0]*header[1]/2 + header[1]), :]), axis=1)[np.sum(np.abs(data[int(header[0]*header[1]/2):int(header[0]*header[1]/2 + header[1]), :]), axis=1) != 0])

tmp_data = np.sum(np.abs(data), axis=1)

max_sw_len = max(tmp_data)

input_data = data.ravel()

max_len = max(input_data)
min_len = min(input_data)


n_input = np.prod(header)
n_output = 60

init_pts = np.ones((header[0] * header[1], 2))*-1

all_goodacts = np.array([])

for i in range(header[0]):
    
    for j in range(header[1]):

        if i == 0:
            if not np.all(data[header[1] * i + j, :] == 0):
                init_pts[header[1] * i + j, :] = np.array([fp_width/2 + j * fp_width, y_bbox])
        
        elif i == header[0]/2:
            if not np.all(data[header[1] * i + j, :] == 0):
                init_pts[header[1] * i + j, :] = np.array([0, fp_width/2 + j * fp_width])
                   
        elif i > 0 and i < header[0]/2:
            if not np.all(data[header[1] * i + j, :] == 0):
                if (fp_width/2 + j * fp_width)/np.sin(deg2rad(i * 180 / header[0])) <= y_bbox:
                    init_pts[header[1] * i + j, :] = np.array([0, (fp_width/2 + j * fp_width)/np.sin(deg2rad(i * 180 / header[0]))])

                else:
                    init_pts[header[1] * i + j, :] = np.array([np.tan(deg2rad(i * 180 / header[0])) * ((fp_width/2 + j * fp_width)/np.sin(deg2rad(i * 180 / header[0])) - y_bbox), y_bbox])
        
        elif i > header[0]/2 and i < header[0]:
            if not np.all(data[header[1] * i + j, :] == 0):
                if (fp_width/2 + j * fp_width)/np.cos(np.pi - deg2rad(i * 180 / header[0])) <= x_bbox:
                    init_pts[header[1] * i + j, :] = np.array([x_bbox - (fp_width/2 + j * fp_width)/np.cos(np.pi - deg2rad(i * 180 / header[0])), 0])

                else:
                    init_pts[header[1] * i + j, :] = np.array([0, np.tan(deg2rad(i * 180 / header[0]) - np.pi/2) * ((fp_width/2 + j * fp_width)/np.cos(np.pi - deg2rad(i * 180 / header[0])) - x_bbox)])

def check_intersections(action):

    cont = 0

    for i in range(action.shape[1]):

        for j in range(i + 1, action.shape[1]):

            if action[0, j] == action[0, i]:
                if not (np.abs(action[1, j]) + action[2, j] < np.abs(action[1, i]) or np.abs(action[1, i]) + action[2, i] < np.abs(action[1, j])):
                    # plot_action(action)
                    cont += 5

            elif action[0, j] // header[1] != action[0, i] // header[1]:

                # getting points of action i
                if action[0, i] // header[1] == 0:
                    if init_pts[int(action[0, i])] is not -np.ones(2):
                        xi = [init_pts[int(action[0, i]), 0], init_pts[int(action[0, i]), 0]]
                        yi = [init_pts[int(action[0, i]), 1] - np.abs(action[1, i]), init_pts[int(action[0, i]), 1] - np.abs(action[1, i]) - action[2, i]]

                elif action[0, i] // header[1] == header[0]/2:
                    if init_pts[int(action[0, i])] is not -np.ones(2):
                        xi = [init_pts[int(action[0, i]), 0] + np.abs(action[1, i]), init_pts[int(action[0, i]), 0] + np.abs(action[1, i]) + action[2, i]]
                        yi = [init_pts[int(action[0, i]), 1], init_pts[int(action[0, i]), 1]]

                else:
                    if init_pts[int(action[0, i])] is not -np.ones(2):
                        alpha = -np.pi/2
                        # alpha = 0
                        xi = [init_pts[int(action[0, i]), 0] + np.cos(deg2rad((action[0, i] // header[1]) * 180 / header[0]) + alpha) * np.abs(action[1, i]), init_pts[int(action[0, i]), 0] + np.cos(deg2rad((action[0, i] // header[1]) * 180 / header[0]) + alpha) * (np.abs(action[1, i]) + action[2, i])]
                        yi = [init_pts[int(action[0, i]), 1] + np.sin(deg2rad((action[0, i] // header[1]) * 180 / header[0]) + alpha) * np.abs(action[1, i]), init_pts[int(action[0, i]), 1] + np.sin(deg2rad((action[0, i] // header[1]) * 180 / header[0]) + alpha) * (np.abs(action[1, i]) + action[2, i])]

                # getting points of action j
                if action[0, j] // header[1] == 0:
                    if init_pts[int(action[0, j])] is not -np.ones(2):
                        xj = [init_pts[int(action[0, j]), 0], init_pts[int(action[0, j]), 0]]
                        yj = [init_pts[int(action[0, j]), 1] - np.abs(action[1, j]), init_pts[int(action[0, j]), 1] - np.abs(action[1, j]) - action[2, j]]

                elif action[0, j] // header[1] == header[0]/2:
                    if init_pts[int(action[0, j])] is not -np.ones(2):
                        xj = [init_pts[int(action[0, j]), 0] + np.abs(action[1, j]), init_pts[int(action[0, j]), 0] + np.abs(action[1, j]) + action[2, j]]
                        yj = [init_pts[int(action[0, j]), 1], init_pts[int(action[0, j]), 1]]

                else:
                    if init_pts[int(action[0, j])] is not -np.ones(2):
                        alpha = -np.pi/2
                        # alpha = 0
                        xj = [init_pts[int(action[0, j]), 0] + np.cos(deg2rad((action[0, j] // header[1]) * 180 / header[0]) + alpha) * np.abs(action[1, j]), init_pts[int(action[0, j]), 0] + np.cos(deg2rad((action[0, j] // header[1]) * 180 / header[0]) + alpha) * (np.abs(action[1, j]) + action[2, j])]
                        yj = [init_pts[int(action[0, j]), 1] + np.sin(deg2rad((action[0, j] // header[1]) * 180 / header[0]) + alpha) * np.abs(action[1, j]), init_pts[int(action[0, j]), 1] + np.sin(deg2rad((action[0, j] // header[1]) * 180 / header[0]) + alpha) * (np.abs(action[1, j]) + action[2, j])]

                A = np.array([[xi[1] - xi[0], -(xj[1] - xj[0])], 
                            [yi[1] - yi[0], -(yj[1] - yj[0])]])
                
                b = np.array([xj[0] - xi[0], 
                            yj[0] - yi[0]])
                
                try:
                    t, s = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    print("Rette parallele")
                
                if 0 < t < 1 and 0 < s < 1:
                    # plot_action(action)
                    cont += 1

    # plot_action(action)
    return cont
                

def plot_action(action, strng=None, init_pts=init_pts):

    img = cv2.imread("immagini/perimeters/per_" + n_per + ".png", cv2.IMREAD_GRAYSCALE)

    bbox_center = [x_bbox/2, y_bbox/2]

    img_scalefactor = np.round(1000/(1.2*int(max(x_bbox, y_bbox))))
        
    for i in range(action.shape[1]):
        if action[0, i] // header[1] == 0:
            if init_pts[int(action[0, i])] is not -np.ones(2):
                x = [init_pts[int(action[0, i]), 0], init_pts[int(action[0, i]), 0]]
                y = [init_pts[int(action[0, i]), 1] - np.abs(action[1, i]), init_pts[int(action[0, i]), 1] - np.abs(action[1, i]) - action[2, i]]
                x_img = [int(img_scalefactor * (xi + ((500 / img_scalefactor) - bbox_center[0]))) for xi in x]
                y_img = [(1000 - int(img_scalefactor * (yi + ((500 / img_scalefactor) - bbox_center[1])))) for yi in y]
                cv2.line(img, tuple(map(int, (x_img[0], y_img[0]))), tuple(map(int, (x_img[1], y_img[1]))), color=128, thickness=2)

        elif action[0, i] // header[1] == header[0]/2:
            if init_pts[int(action[0, i])] is not -np.ones(2):
                x = [init_pts[int(action[0, i]), 0] + np.abs(action[1, i]), init_pts[int(action[0, i]), 0] + np.abs(action[1, i]) + action[2, i]]
                y = [init_pts[int(action[0, i]), 1], init_pts[int(action[0, i]), 1]]
                x_img = [int(img_scalefactor * (xi + ((500 / img_scalefactor) - bbox_center[0]))) for xi in x]
                y_img = [(1000 - int(img_scalefactor * (yi + ((500 / img_scalefactor) - bbox_center[1])))) for yi in y]
                cv2.line(img, tuple(map(int, (x_img[0], y_img[0]))), tuple(map(int, (x_img[1], y_img[1]))), color=128, thickness=2)

        else:
            if init_pts[int(action[0, i])] is not -np.ones(2):
                alpha = -np.pi/2
                # alpha = 0
                x = [init_pts[int(action[0, i]), 0] + np.cos(deg2rad((action[0, i] // header[1]) * 180 / header[0]) + alpha) * np.abs(action[1, i]), init_pts[int(action[0, i]), 0] + np.cos(deg2rad((action[0, i] // header[1]) * 180 / header[0]) + alpha) * (np.abs(action[1, i]) + action[2, i])]
                y = [init_pts[int(action[0, i]), 1] + np.sin(deg2rad((action[0, i] // header[1]) * 180 / header[0]) + alpha) * np.abs(action[1, i]), init_pts[int(action[0, i]), 1] + np.sin(deg2rad((action[0, i] // header[1]) * 180 / header[0]) + alpha) * (np.abs(action[1, i]) + action[2, i])]
                x_img = [int(img_scalefactor * (xi + ((500 / img_scalefactor) - bbox_center[0]))) for xi in x]
                y_img = [(1000 - int(img_scalefactor * (yi + ((500 / img_scalefactor) - bbox_center[1])))) for yi in y]
                cv2.line(img, tuple(map(int, (x_img[0], y_img[0]))), tuple(map(int, (x_img[1], y_img[1]))), color=128, thickness=2)
    
    # for i in range(init_pts.shape[0]):
    #     if init_pts[i, :] is not -np.ones(2):
    #         x_img = int(img_scalefactor * (init_pts[i, 0] + ((500 / img_scalefactor) - bbox_center[0])))
    #         y_img = 1000 - int(img_scalefactor * (init_pts[i, 1] + ((500 / img_scalefactor) - bbox_center[1])))
    #         cv2.circle(img, (x_img, y_img), radius=2, color=128, thickness=-1)

    # x_img = [int(img_scalefactor * (((500 / img_scalefactor) - bbox_center[0]))), int(img_scalefactor * (((500 / img_scalefactor) - bbox_center[0])))]
    # y_img = [1000 - int(img_scalefactor * (((500 / img_scalefactor) - bbox_center[1]))), 1000 - int(img_scalefactor * (y_bbox + ((500 / img_scalefactor) - bbox_center[1])))]
    # cv2.line(img, tuple(map(int, (x_img[0], y_img[0]))), tuple(map(int, (x_img[1], y_img[1]))), color=128, thickness=2)
    # x_img = [int(img_scalefactor * (((500 / img_scalefactor) - bbox_center[0]))), int(img_scalefactor * (x_bbox + ((500 / img_scalefactor) - bbox_center[0])))]
    # y_img = [1000 - int(img_scalefactor * (y_bbox + ((500 / img_scalefactor) - bbox_center[1]))), 1000 - int(img_scalefactor * (y_bbox + ((500 / img_scalefactor) - bbox_center[1])))]
    # cv2.line(img, tuple(map(int, (x_img[0], y_img[0]))), tuple(map(int, (x_img[1], y_img[1]))), color=128, thickness=2)
    # x_img = [int(img_scalefactor * (x_bbox + ((500 / img_scalefactor) - bbox_center[0]))), int(img_scalefactor * (x_bbox + ((500 / img_scalefactor) - bbox_center[0])))]
    # y_img = [1000 - int(img_scalefactor * (y_bbox + ((500 / img_scalefactor) - bbox_center[1]))), 1000 - int(img_scalefactor * (((500 / img_scalefactor) - bbox_center[1])))]
    # cv2.line(img, tuple(map(int, (x_img[0], y_img[0]))), tuple(map(int, (x_img[1], y_img[1]))), color=128, thickness=2)
    # x_img = [int(img_scalefactor * (x_bbox + ((500 / img_scalefactor) - bbox_center[0]))), int(img_scalefactor * (((500 / img_scalefactor) - bbox_center[0])))]
    # y_img = [1000 - int(img_scalefactor * (((500 / img_scalefactor) - bbox_center[1]))), 1000 - int(img_scalefactor * (((500 / img_scalefactor) - bbox_center[1])))]
    # cv2.line(img, tuple(map(int, (x_img[0], y_img[0]))), tuple(map(int, (x_img[1], y_img[1]))), color=128, thickness=2)
    
    # cv2.circle(img, (500, 500), radius=3, color=128, thickness=5)
    # cv2.line(img, (0, 0), (1000, 1000), color=128, thickness=2)
    # cv2.line(img, (0, 1000), (1000, 0), color=128, thickness=2)

    cv2.imshow("Action " + str(strng), img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
        
        self.action_space = Box(low=-1, high=1, shape=(3, int(n_output//3)))
        
        self.observation_space = Box(low=min_len, high=max_len, shape=(input_data.shape))

        self.observation = input_data
        
        self.current_action = None

        self.Nstep = 100

    def step(self, action):

        global all_goodacts

        # print(self.Nstep)

        action[0, :] = np.floor(action[0, :] * (header[0]*header[1])/2 * 0.999 + (header[0]*header[1])/2).astype(int)
        action[1, :] = action[1, :] * max_sw_len/2 - max_sw_len/2
        action[2, :] = action[2, :] * (max_len/2 - fp_width/2) + max_len/2 + fp_width/2

        self.current_action = action

        # print(action[:, 0:5])

        goodacts = np.array([])

        self.Nstep -= 1

        reward = 0
        reward_time = 0

        terminated = False
        
        truncated = False

        flag_stop = False
        
        info = {}

        for i in range(self.action_space.shape[1]):
            
            if action[2, i] == 0:
                flag_stop = True
                break
            
            row = int(action[0, i])

            if np.abs(action[1, i]) + np.abs(action[2, i]) <= np.sum(np.abs(data[row, :])):

                cont = 0

                for j in np.where(data[row, :] > 0)[0]:

                    if np.sum(np.abs(data[row, j if j == 0 else slice(0, j)])) <= np.abs(action[1, i]) and np.sum(np.abs(data[row, 0:(j+1)])) >= np.abs(action[1, i]) + np.abs(action[2, i]):
                        # reward += k_cov * (1 - np.abs((1 - fp_width * action[3, i] / area)))
                        # reward += k_cov * np.exp(- (1 - fp_width * action[3, i] / area)**2)
                        # reward += fp_width * action[3, i] / area
                        # print(j, '\t', np.sum(np.abs(data[row, j if j == 0 else slice(0, j)])), '\t', np.sum(np.abs(data[row, 0:(j+1)])), '\t', action[:, i])
                        if goodacts.size == 0:
                            goodacts = action[:, i].reshape(-1, 1)
                            # print(goodacts)
                        else:
                            goodacts = np.concatenate((goodacts, action[:, i].reshape(-1, 1)), axis=1)
                        reward += 10 * k_cov
                        reward_time += segmentTime(action[2, i])
                    
                    else:
                        cont += 1
                
                if cont == np.where(data[row, :] > 0)[0].shape[0]:
                    reward += -k_cov
            
            # else:
            #     reward += -2*k_cov        
        
        # reward = k_cov * (1 - np.abs((1 - reward)))
        reward_time = k_time * (min_time - reward_time)/min_time
        # reward += reward_time
        reward += -check_intersections(action)

        # print(goodacts)

        if goodacts.size != 0:
            if all_goodacts.size == 0:
                all_goodacts = goodacts
            else:
                if goodacts.ndim == 1:
                    all_goodacts = np.concatenate((all_goodacts, goodacts.reshape(-1,1)), axis=1)
                elif goodacts.ndim > 1:
                    all_goodacts = np.concatenate((all_goodacts, goodacts), axis=1)
            
        
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

        self.Nstep = 100

        info = {}

        return self.observation, info

class CustomSAC(SAC):

    def __init__(self, policy, env, *args, actor_lr=3e-4, critic_lr=3e-4, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
    
    def _update_learning_rate(self, optimizers):
        
        actor_opt, critic_opt, _ = optimizers

        update_learning_rate(actor_opt, self.actor_lr)
        update_learning_rate(critic_opt, self.critic_lr)

class CustomTQC(TQC):

    def __init__(self, policy, env, *args, actor_lr=3e-4, critic_lr=3e-4, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
    
    def _update_learning_rate(self, optimizers):
        
        actor_opt, critic_opt, _ = optimizers

        update_learning_rate(actor_opt, self.actor_lr)
        update_learning_rate(critic_opt, self.critic_lr)

policy_kwargs = dict(
    # activation_fn= nn.Tanh,
    net_arch = dict(pi=[400, 200, 100], qf=[40, 20, 10]),
    # features_extractor_class=PathPlanningNN,
    # features_extractor_kwargs=dict(features_dim=512),
)

env = DummyVecEnv([lambda: PathPlanningEnv()])
# env = PathPlanningEnv()
# env = Monitor(env)

# model = SAC("MlpPolicy",
#             env,
#             policy_kwargs=policy_kwargs,
#             verbose=1,
#             #buffer_size=10000,
#             #batch_size=128,
#             #learning_starts=200,
#             #tensorboard_log= "Training/tensorboard_log/SAC/",
# )


# model = CustomSAC("MlpPolicy",
#                   env,
#                   actor_lr=1e-4,
#                   critic_lr=3e-4,
#                   policy_kwargs=policy_kwargs,
#                   verbose=1,
#                   tensorboard_log= "Training/tensorboard_log/SAC/",
# )

# model = TQC("MlpPolicy",
#             env,
#             verbose=1,
#             policy_kwargs=policy_kwargs,
#             learning_rate=0.001,
#             tensorboard_log="Training/tensorboard_log/TQC/",
# )

# model = CustomTQC("MlpPolicy",
#                   env,
#                   actor_lr=1e-4,
#                   critic_lr=3e-4,
#                   policy_kwargs=policy_kwargs,
#                   verbose=1,
#                   tensorboard_log= "Training/tensorboard_log/TQC/",
# )

# model.learn(total_timesteps=500000,
#             progress_bar=True,
# )

# print(all_goodacts.T)
# print(all_goodacts.shape)
# plot_action(all_goodacts)


# with open("Training/logs/all_goodacts.txt", "w") as f:
#     for riga in all_goodacts.T:
#         f.write(" ".join(map(str, riga)) + "\n")

# model.save("Training/saved_models/SAC/SAC_500k_per_" + n_per + "_anglestep" + angle_step + "°_actor_lr_1e-4_critic_lr_3e-4_interhandling_smallcriticnet.zip")

# model.save("Training/saved_models/TQC/TQC_100k_per_" + n_per + "_anglestep" + angle_step + "°_actor_lr_1e-4_critic_lr_3e-4_interhandling_smallcriticnet.zip")

# del model

model = SAC.load("Training/saved_models/SAC/SAC_100k_per_triangle_anglestep30°_actor_lr_1e-3_critic_lr_3e-3_interhandling_onlymask0°.zip")

# model = TQC.load("Training/saved_models/TQC/TQC_100k_per_rotated_square_anglestep30°_lr_1e-3_interhandling_onlymask0°.zip")

episodes = 3
for episode in range(1, episodes+1):
    
    best_state = np.zeros([1, 3, 20])

    obs = env.reset()

    terminated = False

    best_reward = -1e6

    good_actions = []

    while not terminated:
        action, _ = model.predict(obs)             
        obs, reward, terminated, truncated, *info = env.step(action)
        if reward > best_reward:
            best_reward = reward
            best_state = action
        # action = action.squeeze(0)
        # for i in range(action.shape[1]):
        #     row = int(action[0, i])
        #     if np.sum(np.abs(data[row, j if j == 0 else slice(0, j)])) <= np.abs(action[1, i]) and np.sum(np.abs(data[row, 0:(j+1)])) >= np.abs(action[1, i]) + np.abs(action[2, i]):
        #         good_actions = good_actions.append(action[row, :])
        #         print(i)
    print(best_reward)
    plot_action(best_state.squeeze(0), episode)

#     with open("Training/logs/bestact_" + str(episode) + ".txt", "w") as f:
#         for riga in best_state.squeeze(0).T:
#             f.write(" ".join(map(str, riga)) + "\n")

# plot_action(np.array([[0], [0], [5]]))

# bestact = np.genfromtxt("Training/logs/bestact_1.txt", delimiter=" ")
# plot_action(bestact.T)

# print(check_intersections(np.array([[0, header[1]*header[0]/2, header[1]*header[0]/2], [0, 0, 0], [5, 5, 5]])))

cv2.waitKey(0)
cv2.destroyAllWindows()

env.close()