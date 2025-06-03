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

# === Configuration ===
IMAGE_PATH = "immagini/aree_prova/prova_1_36x36.png"
MAX_STEPS = 4_000
TOTAL_TIMESTEPS = 10_000_000
NUM_ENVS = 32  # 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Reward Constants ===
R_STEP = 0.0
R_ACTION_EQUAL = 0.0
R_ACTION_NOTEQUAL = 0.0
R_COLLIDE = -1.0
R_NEW = 0.1
R_MOVE = 0.0
R_OLD = -1.0
R_DONE = 100.0

# === VALUE
CELL_FREE = 0
CELL_VISITED = 0.2
CELL_NOW = 0.8
CELL_WALL = 1

# === Path Logger ===
FILE_PATH_LOGGER = "path_logger.txt"
path_logger = []

# === Load Map ===
_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if _img is None:
    raise FileNotFoundError(f"Image file not found or cannot be read: {IMAGE_PATH}")
h, w = _img.shape
_coords = np.column_stack(np.where(_img == 255))
FREE_CELLS = len(_coords)
BASE_MAP = (_img != 255).astype(np.float32)

# === Actions ===
class Action(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

# === Custom CNN Extractor ===
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(),
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
        # observation: one channel image
        self.observation_space = spaces.Box(0.0, 1.0, shape=(1, h, w), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.base_map = BASE_MAP.copy()
        self.previous_action = -1

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        # 0=free,1=wall,2=visited
        self.state = self.base_map.copy()
        idx = self.np_random.choice(len(_coords))
        y, x = tuple(_coords[idx])
        self.pos = (y, x)
        # path_logger.append(self.pos)
        self.state[y, x] = CELL_NOW
        self.visited = 1
        self.previous_action = -1
        self.steps = 0
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        y, x = self.pos
        dy, dx = [(1, 0), (0, 1), (-1, 0), (0, -1)][action]
        ny, nx = np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)
        reward = R_STEP
        if self.base_map[ny, nx] == CELL_WALL:  # collision
            reward += R_COLLIDE
            ny, nx = y, x
        else:
            if self.state[ny, nx] == CELL_FREE:
                reward += R_NEW
                self.visited += 1
            reward += R_MOVE
            self.state[y, x] = CELL_VISITED
            if self.state[ny, nx] == CELL_VISITED:
                reward += R_OLD
        self.state[ny, nx] = CELL_NOW
        self.pos = (ny, nx)
        # path_logger.append(self.pos)
        self.steps += 1
        if self.previous_action != action and self.previous_action != -1:
            reward += R_ACTION_NOTEQUAL
        else:
            reward += R_ACTION_EQUAL

        self.previous_action = action
        terminated = self.visited >= FREE_CELLS
        truncated = self.steps >= MAX_STEPS
        if terminated:
            reward += R_DONE
        return self._obs(), float(reward), terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        obs = self.state.copy()
        return obs[None, ...]          # shape (1, h, w)

    def render(self, mode="human") -> None:
        print(f"Pos: {self.pos}, Visited: {self.visited}/{FREE_CELLS}, Steps: {self.steps}")

# === Training Script ===
def make_env(rank: int):
    def _init():
        env = GraphBasedEnv()
        return Monitor(env)
    return _init

def train():
    # vectorized + normalize
    # NUM_ENVS is set to 8 to balance parallelism and resource usage, typically based on the number of CPU cores available.
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    vec_env.seed(0)
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=NUM_ENVS * 64, # *8
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        policy_kwargs={
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": [256, 128]
        },
        verbose=1,
        tensorboard_log= "./",
        device=DEVICE
    )
    # training
    # The progress_bar=True argument enables a visual progress bar during training for better monitoring.
    # model.learn(TOTAL_TIMESTEPS, callback=eval_callback,progress_bar=True)
    model.learn(TOTAL_TIMESTEPS, progress_bar=True)
    # model.save("ppo_graphbased_final")
    # env.save("Training/saved_models/PPO_GraphBased/script_framic/vecnormalize.pkl")

def train_loadedmodel():

    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    vec_env.seed(0)
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    env = VecNormalize.load("ppo_10M_rold1_rnew1e-1_finalboost100.pkl", vec_env)
    model = PPO.load("ppo_10M_rold1_rnew1e-1_finalboost100.zip",
                     env=env,
                    #  custom_objects={
                    #      "learning_rate": 1e-4
                    #  }
    )

    model.learn(total_timesteps=10_000_000,
                # progress_bar=True,
                tb_log_name="ppo_20M_rold1_rnew1e-1_finalboost100",
                reset_num_timesteps=False,
    )

    model.save("ppo_20M_rold1_rnew1e-1_finalboost100")
    env.save("ppo_20M_rold1_rnew1e-1_finalboost100.pkl")

def inference():
    vec_env = DummyVecEnv([make_env(0)])
    env = VecNormalize.load("Training/saved_models/PPO_GraphBased/script_framic/ppo_20M_rmove0.pkl", vec_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(
        "Training/saved_models/PPO_GraphBased/script_framic/ppo_20M_rmove0.zip",
        custom_objects={
            "lr_schedule": lambda _: 0.0003,
            "clip_range": lambda _: 0.2,
            "policy_kwargs": dict({
                    "features_extractor_class": CustomCNN,
                    "features_extractor_kwargs": {"features_dim": 128},
                    "net_arch": [256, 128]
                }
            )
        }
    )

    obs = env.reset()
    terminated = False
    truncated = False

    while not terminated:
        # print(terminated)
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, *info = env.step(action)
    
    path_logger.pop()

    with open(FILE_PATH_LOGGER, "w") as f:
        for pos in path_logger:
            f.write(f"{pos}\n")
    
    env.close()

def inference_video(video_path="inference_video.avi", fps=10):
    vec_env = DummyVecEnv([make_env(0)])
    env = VecNormalize.load("Training/saved_models/PPO_GraphBased/script_framic/ppo_20M_rold1_rnew1e-1_finalboost100.pkl", vec_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(
        "Training/saved_models/PPO_GraphBased/script_framic/ppo_20M_rold1_rnew1e-1_finalboost100.zip",
        custom_objects={
            "lr_schedule": lambda _: 0.0003,
            "clip_range": lambda _: 0.2,
            "policy_kwargs": dict({
                    "features_extractor_class": CustomCNN,
                    "features_extractor_kwargs": {"features_dim": 128},
                    "net_arch": [256, 128]
                }
            )
        }
    )

    obs = env.reset()
    terminated = False
    truncated = False

    # Setup video writer
    img = BASE_MAP.copy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = 20
    frame_size = (img.shape[1]*scale, img.shape[0]*scale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    path = []
    while not terminated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, *info = env.step(action)
        # Aggiorna path
        pos = tuple(np.array(env.get_original_obs()[0][0] == CELL_NOW).nonzero())
        if len(pos[0]) > 0:
            y, x = pos[0][0], pos[1][0]
        else:
            y, x = -1, -1
        path.append((y, x))

        # Crea frame
        frame = BASE_MAP.copy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Colora percorso già visitato
        for py, px in path[:-1]:
            if py >= 0 and px >= 0:
                frame[py, px] = (0, 255, 0)  # Verde chiaro
        # Colora posizione attuale
        if y >= 0 and x >= 0:
            frame[y, x] = (0, 0, 255)  # Rosso
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_NEAREST)
        out.write(frame)

    out.release()
    env.close()
    print(f"Video salvato in: {video_path}")

def video_from_path_logger(path_logger_file="path_logger.txt", video_path="video_from_logger.avi", fps=10):
    img = BASE_MAP.copy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = 20
    frame_size = (img.shape[1]*scale, img.shape[0]*scale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    # Carica il path dal file
    path = []
    with open(path_logger_file, "r") as f:
        for line in f:
            y, x = eval(line.strip())
            path.append((y, x))

    visited = set()
    for i, (y, x) in enumerate(path):
        frame = BASE_MAP.copy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Colora percorso già visitato (escludendo la posizione attuale)
        for py, px in list(visited):
            frame[py, px] = (0, 255, 0)  # Verde chiaro
        # Colora posizione attuale
        frame[y, x] = (0, 0, 255)  # Rosso
        visited.add((y, x))
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_NEAREST)
        out.write(frame)

    out.release()
    print(f"Video salvato in: {video_path}")

def visualize_path():
    img = BASE_MAP.copy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Converti in RGB

    path = []
    with open(FILE_PATH_LOGGER, "r") as f:
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

def visualize_overlap():
    img = BASE_MAP.copy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Converti in RGB

    # white = np.all(img == [255, 255, 255], axis=-1)
    # img[white] = (0, 64, 255)

    # Conta le visite per ogni cella
    path = []
    with open(FILE_PATH_LOGGER, "r") as f:
        for line in f:
            y, x = eval(line.strip())
            path.append((y, x))

    from collections import Counter
    counts = Counter(path)
    if not counts:
        print("Path vuoto.")
        return

    max_count = max(counts.values())

    for (y, x), count in counts.items():
        if count == 1:
            color = (0, 255, 0)  # Verde
        else:
            alpha = (count - 2) / (max_count - 2) if max_count > 2 else 0
            r = int(0 * (1 - alpha) + 255 * alpha)
            g = 0   # int(128 * (1 - alpha) + 0 * alpha)
            b = int(255 * (1 - alpha) + 0 * alpha)
            color = (b, g, r)
        img[y, x] = color

    img = cv2.resize(img, (img.shape[1]*20, img.shape[0]*20), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Path Coverage", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prova():
    vec_env = DummyVecEnv([make_env(0)])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    obs = env.reset()
    terminated = False
    truncated = False
    for _ in range(10):
        action = 0
        obs, reward, terminated, truncated, *info = env.step([action])
        # env.render()

if __name__ == '__main__':
    # fix global seed
    set_random_seed(0)
    
    # train()
    # train_loadedmodel()
    # inference()
    # prova()
    # visualize_path()
    # visualize_overlap()

    # inference_video(video_path="inference_video_20M_rold1_rnew1e-1.avi", fps=20)
    # video_from_path_logger(path_logger_file=FILE_PATH_LOGGER, video_path="video_from_logger.avi", fps=20)

    pass
