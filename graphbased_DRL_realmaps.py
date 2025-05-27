import numpy as np
import cv2

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from typing import Any

img = cv2.imread("immagini/aree_prova/prova_1_36x36.png", cv2.IMREAD_GRAYSCALE)
freecells = np.count_nonzero(img == 255)
coords = np.column_stack(np.where(img == 255))
dimcells_y, dimcells_x = img.shape

img = cv2.bitwise_not(img)
# img = np.array(img)
# img[img == 255] = 2

N_TRIALS = 500
N_STARTUP_TRIALS = 10

N_TIMESTEPS = 1_000_000
N_EVALUATIONS = 20
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 10

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
}

# print(freecells)

max_steps = 4000

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

def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    gamma = trial.suggest_float("gamma", 0.97, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    n_steps = 2**n_steps_pow
    trial.set_user_attr("n_steps", n_steps)
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "policy_kwargs": {
            "activation_fn": activation_fn,
        },
    }

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

class CustomCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        # features_dim: int = 512,
        features_dim: int = 128,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class GraphBasedPPEnv(Env):

    def __init__(self):
        super().__init__()
        
        # self.observation_space = Box(low=0, high=255, shape=(1, dimcells_y, dimcells_x), dtype=np.uint8)

        self.observation_space = Dict({
            "map": Box(low=0, high=255, shape=(1, dimcells_y, dimcells_x), dtype=np.uint8),
            "position": MultiDiscrete(np.array([dimcells_y, dimcells_x])),
        })

        self.action_space = Discrete(4)

        self.map_state = img

        self.freecells = freecells

        init_y, init_x = coords[np.random.choice(coords.shape[0])]

        self.map_state[init_y, init_x] = 2

        self.position = np.array([init_y, init_x])

        self.visited_cells = 1

        self.previous_visited_cells = 1

        self.previous_action = -1

        # self.full_state = np.append(np.append(self.position, self.visited_cells), self.map_state.ravel())

        self.Nstep = 0

    def step(self, action):
        
        # 0 = north
        # 1 = east
        # 2 = south
        # 3 = west

        self.map_state = self.map_state[0]

        reward = 0

        terminated = False

        truncated = False

        self.Nstep += 1

        reward -= 0.1

        if action == 0:

            if self.position[0] < dimcells_y - 1:

                # print(self.position[1])
                if self.map_state[self.position[0] + 1, self.position[1]] != 255:
                    self.map_state[self.position[0], self.position[1]] = 1
                    # moving on a free cell
                    self.position[0] += 1
                    self.map_state[self.position[0], self.position[1]] = 1
                    reward += 0.01
                    if self.map_state[self.position[0], self.position[1]] == 0:
                        # rewarded if moving on an unvisited cell
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

                if self.map_state[self.position[0], self.position[1] + 1] != 255:
                    self.map_state[self.position[0], self.position[1]] = 1
                    # moving on a free cell
                    self.position[1] += 1
                    self.map_state[self.position[0], self.position[1]] = 1
                    reward += 0.01
                    if self.map_state[self.position[0], self.position[1]] == 0:
                        # rewarded if moving on an unvisited cell
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

                if self.map_state[self.position[0] - 1, self.position[1]] != 255:
                    self.map_state[self.position[0], self.position[1]] = 1
                    # moving on a free cell
                    self.position[0] -= 1
                    self.map_state[self.position[0], self.position[1]] = 1
                    reward += 0.01
                    if self.map_state[self.position[0], self.position[1]] == 0:
                        # rewarded if moving on an unvisited cell
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

                if self.map_state[self.position[0], self.position[1] - 1] != 255:
                    self.map_state[self.position[0], self.position[1]] = 1
                    # moving on a free cell
                    self.position[1] -= 1
                    self.map_state[self.position[0], self.position[1]] = 1
                    reward += 0.01
                    if self.map_state[self.position[0], self.position[1]] == 0:
                        # rewarded if moving on an unvisited cell
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
        
        if self.Nstep == max_steps:
            truncated = True
        
        if self.visited_cells == self.freecells:
            terminated = True

        info = {}

        # self.full_state = np.append(np.append(self.position, self.visited_cells), self.map_state.ravel())

        self.map_state = np.array([self.map_state])

        obs = {
            "map": self.map_state,
            "position": self.position,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.map_state = img

        init_y, init_x = coords[np.random.choice(coords.shape[0])]

        self.map_state[init_y, init_x] = 2

        self.position = np.array([init_y, init_x])

        self.visited_cells = 1

        self.previous_visited_cells = 1

        self.previous_action = -1

        self.Nstep = 0
        
        # self.full_state = np.append(np.append(self.position, self.visited_cells), self.map_state.ravel())

        info = {}

        self.map_state = np.array([self.map_state])

        obs = {
            "map": self.map_state,
            "position": self.position,
        }

        return obs, info
    
    def get_map(self):
        return self.map_state

def objective(trial: optuna.Trial) -> float:
    # env = DummyVecEnv([lambda: GraphBasedPPEnv()])
    env = GraphBasedPPEnv()
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = PPO(env=env, **kwargs)
    # Create env used for evaluation.
    eval_env = make_vec_env(lambda: GraphBasedPPEnv(), n_envs=1)
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=max(EVAL_FREQ, 1),
        deterministic=True,
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback, progress_bar=True)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

def train():

    env = GraphBasedPPEnv()
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = Monitor(env)
    check_env(env=env)

    policy_kwargs=dict(
        # features_extractor_class=CustomCNN,
        net_arch=[256, 64, 16],
    )

    model = PPO("MultiInputPolicy",
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                # learning_rate=0.001,
                # batch_size=1024,
                # gae_lambda=0.98,
                # gamma=0.995,
                ent_coef=0.01,
                tensorboard_log= "Training/tensorboard_log/PPO_GraphBased/",
    )

    model.learn(total_timesteps=5_000_000, progress_bar=True)

    model.save("Training/saved_models/PPO_GraphBased/PPO_5M_realmap36x36_MultiInputPolicy_netarch_256_64_16_entcoeff1e-2.zip")

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
        print_map(obs.shape)

        while not terminated:
            count += 1
            print(count)
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, *info = env.step(action)
            # print("Reward:\t", reward)
            print_map(obs[0, 3:].reshape(dimcells_y, dimcells_x), action)

def train_loadedmodel():
    env = Monitor(GraphBasedPPEnv())
    check_env(env=env)


    model = PPO.load("Training/saved_models/PPO_GraphBased/PPO_20M_realmap36x36_MultiInputPolicy_netarch_256_64_16.zip", env=env)

    model.learn(total_timesteps=10_000_000,
                progress_bar=True,
                tb_log_name="PPO_30M_realmap36x36_MultiInputPolicy_netarch_256_64_16",
                reset_num_timesteps=False,
    )

    model.save("Training/saved_models/PPO_GraphBased/PPO_30M_realmap36x36_MultiInputPolicy_netarch_256_64_16.zip")


def optimization():
    # Set pytorch num threads to 1 for faster training.
    th.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    with open("optimization_results.txt", "w") as file:
        print(f"Number of finished trials: {len(study.trials)}")
        file.write("Number of finished trials: " + str(len(study.trials)) + "\n")

        print("Best trial:")
        trial = study.best_trial
        file.write("Best trial:\n")

        print("  Value: ", trial.value)
        file.write("\tValue: " + str(trial.value) + "\n")

        print("  Params: ")
        file.write("\Params: \n")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            file.write("\t" + str(key) + ": " + str(value) + "\n")

        print("  User attrs:")
        file.write("\tUser attrs:\n")
        for key, value in trial.user_attrs.items():
            print(f"    {key}: {value}")
            file.write("\t" + str(key) + ": " + str(value) + "\n")

if __name__ == '__main__':
        
    train()
    # inference()
    # train_loadedmodel()
    # optimization()

    pass
        