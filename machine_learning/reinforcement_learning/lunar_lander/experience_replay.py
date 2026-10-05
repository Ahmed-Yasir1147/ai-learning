from collections import deque
import random


class ReplayMemory:

    def __init__(self, size):
        self.memory = deque([], size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

