import os
from enum import IntEnum
from typing import Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

torch.backends.cudnn.benchmark = True  # For performance optimization on GPUs
# === Configuration ===
IMAGE_PATH = "images/prova_1_36x36.png"
TOTAL_TIMESTEPS = 1_000_000
NUM_ENVS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHAPING_K = 0.5
GAMMA = 0.99

DEFAULT_MODEL_NAME = "ppo_trained"
DEFAULT_MODEL_PATH = "Training/model/"
DEFAULT_TENSORBOARD_LOG = "Training/tensorboard_log/"

NET_ARCH = [256, 128]

# === Reward Constants ===
R_DONE              =   50.0
R_NEW               =   1.0
R_STEP              =  -0.02
R_ACTION_EQUAL      =   0.0 
R_ACTION_NOTEQUAL   =   0.0
R_VISITED           =  -0.2
R_COLLIDE           =  -3.0
R_TIMEOUT           =  -5.0 
R_MOVE              =   0.005 

# R_DONE              = 100.0
# R_NEW               =   0.1
# R_ACTION_EQUAL      =   0.0
# R_STEP              =   0.0
# R_ACTION_NOTEQUAL   =   0.0
# R_VISITED           =  -1.0
# R_COLLIDE           =  -1.0
# R_TIMEOUT           =   0.0
# R_MOVE              =   0.0 # Reward for a valid move

# === VALUE
CELL_FREE =     0.0
CELL_VISITED =  1.0
CELL_NOW =      2.0
CELL_WALL =     3.0


# === Load Map ===
_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if _img is None:
    raise FileNotFoundError(f"Image file not found or cannot be read: {IMAGE_PATH}")
h, w = _img.shape
_coords = np.column_stack(np.where(_img == 255))
FREE_CELLS = len(_coords)
MAX_STEPS = 5 * FREE_CELLS 
BASE_MAP = (_img != 255).astype(np.float32) * CELL_WALL

# === Actions ===
class Action(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

# === Path Logger ===
path_logger = []

class VecNormalizeCheckpointCallback(BaseCallback):
    def __init__(self, env, save_freq, save_path, name_prefix="vecnormalize_checkpoint", verbose=0):
        super().__init__(verbose)
        self.env = env
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}.pkl")
            self.env.save(path)
        return True

class PrintExtraInfosCallback(BaseCallback):
    def __init__(self, verbose=0, num_envs=NUM_ENVS):
        print("PrintExtraInfosCallback initialized", flush=True)
        super().__init__(verbose)
        self.absoluteMax_visited_cells = 0
        self.absoluteMin_collisions = 10000000
        self.num_envs = num_envs
        self.episode_visited = []
        self.episode_collisions = []
        self.episode_steps = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and "visited_cells" in infos[i]:
                # print(f"AGGIUNGO visited_cells: {infos[i]['visited_cells']}", flush=True)
                self.episode_visited.append(infos[i]["visited_cells"])
                self.episode_collisions.append(infos[i].get("collisions", 0))
                self.episode_steps.append(infos[i].get("steps", 1))
        # Quando tutti gli env hanno terminato un episodio, stampa statistiche
        # print(f"Len episode_visited: {len(self.episode_visited)} / {self.num_envs}", flush=True)
        if len(self.episode_visited) >= self.num_envs:
            vals = self.episode_visited[:self.num_envs]
            colls = self.episode_collisions[:self.num_envs]
            steps = self.episode_steps[:self.num_envs]
            self.episode_count += self.num_envs

            if self.absoluteMax_visited_cells < max(vals):
                self.absoluteMax_visited_cells = max(vals)
            if self.absoluteMin_collisions > min(colls):
                self.absoluteMin_collisions = min(colls);

            # Stampa tabellare
            meanVisited = sum(vals) / len(vals)
            meanCollisions = sum(colls) / len(colls)
            meanSteps = sum(steps) / len(steps)
            collision_rate = meanCollisions / meanSteps if meanSteps > 0 else 0
            coverage_ratio = meanVisited / FREE_CELLS 
            print(f"EPISODI: {self.episode_count}")
            print(f"{'INDICATORE':<25}\t{'MIN':<8}\t{'MAX':<8}\t{'MEDIA':<10}\t{'ASSOLUTO'}")
            print(f"{'Celle nuove visitate':<25}\t{min(vals):<8}\t{max(vals):<8}\t{meanVisited:<10.2f}\t{self.absoluteMax_visited_cells}")
            print(f"{'Coverage ratio':<25}\t{(min(vals) / FREE_CELLS):<10.2f}\t{( max(vals) / FREE_CELLS):<10.2f}\t{(meanVisited / FREE_CELLS):<10.2f}\t{(self.absoluteMax_visited_cells / FREE_CELLS):<10.2f}")
            print(f"{'Collisioni':<25}\t{min(colls):<8}\t{max(colls):<8}\t{meanCollisions:<10.2f}\t{self.absoluteMin_collisions}")
            print(f"{'Steps':<25}\t{min(steps):<8}\t{max(steps):<8}\t{meanSteps:<10.2f}\t{'-'}")
            print(f"{'Collision rate':<25}\t{'':<8}\t{'':<8}\t{collision_rate:<10.4f}\t{'-'}")
            print("-" * 80)


            # Log su TensorBoard
            self.logger.record("custom/visited_cells_mean", meanVisited)
            self.logger.record("custom/collisions_mean", meanCollisions)
            self.logger.record("custom/steps_mean", meanSteps)
            self.logger.record("custom/collision_rate", collision_rate)
            self.logger.record("custom/coverage_ratio", coverage_ratio)

            self.episode_visited = self.episode_visited[self.num_envs:]
            self.episode_collisions = self.episode_collisions[self.num_envs:]
            self.episode_steps = self.episode_steps[self.num_envs:]
        return True

# === Custom CNN Extractor ===
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]
        self.cnn = nn.Sequential(
            # nn.Conv2d(n_ch, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),   # 36x36 -> 18x18
            # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),     # 18x18 -> 9x9
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),    # 9x9 -> 5x5
            nn.Conv2d(n_ch, 32, kernel_size=4, stride=2, padding=0), nn.ReLU(),   # 36x36 -> 18x18
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),     # 18x18 -> 9x9
            nn.Flatten()
        )
        with torch.no_grad():
            sample = torch.zeros((1, *observation_space.shape))
            n_flat = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))

# === Environment ===
class GraphBasedEnv(Env):
    metadata = {"render_modes": ["human"]}

    global path_logger

    def __init__(self):
        super().__init__()
        # observation: 3 channel image (walls, agent, visited)
        self.observation_space = spaces.Box(0, 1.0, shape=(3, h, w), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.base_map = BASE_MAP.copy().astype(np.float32)
        self.previous_action = -1

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.state = self.base_map.copy().astype(np.float32)
        idx = self.np_random.choice(len(_coords))
        y, x = tuple(_coords[idx])
        self.pos = (y, x)
        self.state[y, x] = CELL_NOW
        self.visited = 1
        self.episode_collisions = 0
        self.previous_action = -1
        self.steps = 0
        self.prev_cov = self.visited / FREE_CELLS
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        y, x = self.pos
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
 
        reward = 0

        reward += R_STEP

        terminated = False
        truncated = False

        ny, nx = y + dy, x + dx
        
        outBox = not (0 <= y+dy < h and 0 <= x+dx < w) 
        if outBox:  #   out of bounds
            reward += R_COLLIDE 
            self.episode_collisions += 1 
            ny, nx = y, x # stay in place
            
        elif np.isclose(self.base_map[ny, nx], CELL_WALL):  # wall collision

            reward += R_COLLIDE
            self.episode_collisions += 1
            ny, nx = y, x  # stay in place

        else:

            
            reward += R_MOVE

            if np.isclose(self.state[ny, nx], CELL_FREE):
                reward += R_NEW
                self.visited += 1
            elif np.isclose(self.state[ny, nx], CELL_VISITED):
                reward += R_VISITED

            self.state[y, x] = CELL_VISITED

            if self.previous_action != action:
                reward += R_ACTION_NOTEQUAL
            else:
                reward += R_ACTION_EQUAL
        
        
        # --- shaping sul progresso di copertura ---
        coverage_ratio = self.visited / FREE_CELLS          # [0 … 1]
        dense_bonus = SHAPING_K * (coverage_ratio - self.prev_cov * GAMMA)
        reward += dense_bonus
        self.prev_cov = coverage_ratio
        # -----------------------------------   

        self.state[ny, nx] = CELL_NOW
        self.pos = (ny, nx)

        self.steps += 1
        self.previous_action = action

        if(self.visited >= FREE_CELLS):
            terminated = True
            reward += R_DONE
            # print("TERMINATED: coverage completa", flush=True)
        elif(self.steps >= MAX_STEPS):
            truncated = True
            reward += R_TIMEOUT
            # print("TRUNCATED: max steps raggiunto", flush=True)
        
        info = {
            "collisions": self.episode_collisions,
            "visited_cells": self.visited,
            "steps": self.steps
        }

        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        # 3 canali: [0]=muri, [1]=agente, [2]=visitate
        obs = np.zeros((3, h, w), dtype=np.float32)
        # Canale 0: muri
        obs[0] = (self.base_map == CELL_WALL).astype(np.float32)
        # Canale 1: agente
        obs[1, :, :] = 0.0
        y, x = self.pos
        obs[1, y, x] = 1.0
        # Canale 2: visitate
        obs[2] = (self.state == CELL_VISITED).astype(np.float32)
        return obs

    def render(self, mode="human") -> None:
        print(f"Pos: {self.pos}, Visited: {self.visited}/{FREE_CELLS}, Steps: {self.steps}")

# === Training Script ===
def make_env(rank: int):
    def _init():
        env = GraphBasedEnv()
        return Monitor(env)
    return _init

def train(total_steps: int = TOTAL_TIMESTEPS, model_name_load: str = None, model_name_save: str = DEFAULT_MODEL_NAME):
    # vectorized + normalize
    # NUM_ENVS is set to 8 to balance parallelism and resource usage, typically based on the number of CPU cores available.
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    vec_env.seed(0)
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    reset_timesteps = True
    if(model_name_load is not None):
        model_path_load = DEFAULT_MODEL_PATH + model_name_load + ".zip"
        if os.path.exists(model_path_load):
            print(f"Loading model from {model_name_load}")
            env = VecNormalize.load(DEFAULT_MODEL_PATH + model_name_load + "_vecnormalize.pkl", vec_env)
            model = PPO.load(model_path_load, env=env, device=DEVICE)
            reset_timesteps = False
    else:
        print("No model to load, creating a new one.")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate = 2e-4, #3e-4,
            n_steps=2048, 
            batch_size= 4096,
            gamma=GAMMA,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.003,
            max_grad_norm=0.5,
            policy_kwargs={
                "features_extractor_class": CustomCNN,
                "features_extractor_kwargs": {"features_dim": 128},
                "net_arch": [256, 128]
            },
            verbose=1,
            tensorboard_log= DEFAULT_TENSORBOARD_LOG,
            device=DEVICE
        )


    checkpoint_callback = CheckpointCallback(
        save_freq=10_000_000 // NUM_ENVS,
        save_path=DEFAULT_MODEL_PATH,
        name_prefix="ppo_checkpoint"
    )

    vecnorm_callback = VecNormalizeCheckpointCallback(env, 10_000_000 // NUM_ENVS, DEFAULT_MODEL_PATH)

    # training
    # The progress_bar=True argument enables a visual progress bar during training for better monitoring.
    # model.learn(TOTAL_TIMESTEPS, callback=eval_callback,progress_bar=True)
    model.learn(total_steps, 
                reset_num_timesteps=reset_timesteps, 
                progress_bar=True, 
                callback=[PrintExtraInfosCallback(), checkpoint_callback, vecnorm_callback])

    model_save_path = DEFAULT_MODEL_PATH + model_name_save + ".zip"
    model.save(model_save_path)
    env_save_path = DEFAULT_MODEL_PATH + model_name_save + "_vecnormalize.pkl"
    env.save(env_save_path)

def inference(model_name_load: str = DEFAULT_MODEL_NAME):
    vec_env = DummyVecEnv([make_env(0)])
    env = VecNormalize.load(DEFAULT_MODEL_PATH + model_name_load + "_vecnormalize.pkl", vec_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(
        DEFAULT_MODEL_PATH + model_name_load + ".zip",
        custom_objects={
            "lr_schedule": lambda _: 0.0002,
            "clip_range": lambda _: 0.2,
            # "policy_kwargs": dict({
            #         "features_extractor_class": CustomCNN,
            #         "features_extractor_kwargs": {"features_dim": 128},
            #         "net_arch": [256, 128]
            #     }
            # )
        }
    )

    obs = env.reset()
    terminated = False
    truncated = False

    while not terminated:
        # print(terminated)
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, *info = env.step(action)
    
    if path_logger: path_logger.pop()

    with open("path_logger.txt", "w") as f:
        for pos in path_logger:
            f.write(f"{pos}\n")
    
    env.close()

def inference_video(model_name_load: str = DEFAULT_MODEL_NAME, video_path: str = "inference_video.avi", fps: int = 5):
    vec_env = DummyVecEnv([make_env(0)])
    env = VecNormalize.load(DEFAULT_MODEL_PATH + model_name_load + "_vecnormalize.pkl", vec_env)
    env.training = False
    env.norm_reward = False
    env.norm_obs = True

    model = PPO.load(
        DEFAULT_MODEL_PATH + model_name_load + ".zip",
        custom_objects={
            "lr_schedule": lambda _: 0.0002,
            "clip_range": lambda _: 0.2,
            # "policy_kwargs": dict({
            #         "features_extractor_class": CustomCNN,
            #         "features_extractor_kwargs": {"features_dim": 128},
            #         "net_arch": [256, 128]
            #     }
            # )
        }
    )

    obs = env.reset()
    terminated = False
    truncated = False

    # Prepare video writer
    img = BASE_MAP.copy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape[:2]
    scale = 20
    out_size = (width * scale, height * scale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, out_size)

    # Reset path_logger
    global path_logger
    path_logger = []

    while not terminated:
        obs_np = np.array(obs)
        if obs_np.ndim == 4:
            obs_np = obs_np[0]

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Bianco per i muri
        frame[BASE_MAP == CELL_WALL] = (255, 255, 255)
        # Verde per celle visitate
        frame[obs_np[2] > 0] = (0, 255, 0)
        # Rosso per la posizione attuale dell'agente
        agent_pos = np.argwhere(obs_np[1] > 0)
        if agent_pos.size > 0:
            y, x = agent_pos[0]
            frame[y, x] = (0, 0, 255)
            path_logger.append((y, x))

        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_NEAREST)
        video_writer.write(frame)

        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, *info = env.step(action)

    video_writer.release()
    env.close()
    print(f"Video salvato in: {video_path}")

def visualize_path():
    img = BASE_MAP.copy()
    img = (img * 255).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Converti in RGB

    path = []
    with open("path_logger.txt", "r") as f:
        for line in f:
            y, x = eval(line.strip())
            path.append((y, x))

    n = len(path)
    if n == 0:
        print("Path vuoto.")
        return

    # Dal blu (255,0,0) al verde (0,255,0)
    for i, (y, x) in enumerate(path[1:], 1):
        alpha = i / (n - 1) if n > 1 else 1.0
        # Interpolazione lineare tra blu e verde
        b = int(255 * (1 - alpha))
        g = int(255 * alpha)
        r = 0
        img[y, x] = (b, g, r)
    
    # Primo punto rosso
    y0, x0 = path[0]
    img[y0, x0] = (0, 0, 255)  # Rosso in BGR

    img = cv2.resize(img, (img.shape[1]*20, img.shape[0]*20), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # fix global seed
    set_random_seed(0)
    
    # train(total_steps=100_000_000, model_name_save="ppo_trained_100M")
    # inference(model_name_load="ppo_trained_100M")
    inference_video(model_name_load="ppo_trained_100M", video_path="inference_video_100M.avi", fps=20)
    # visualize_path()

    pass
