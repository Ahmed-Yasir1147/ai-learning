import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gymnasium as gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

if gym.__version__ < 0.26:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb", apply_api_compatiblity=True)

env = JoypadSpace(
    env, [["right"], ["right", "A"]]
)

env.reset()
next_state, reward, terminated, truncated, info = env.step(action=1)
print(f"{next_state.shape},\n {reward},\n {terminated},\n {info}")