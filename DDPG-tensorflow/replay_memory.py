from collections import deque
import random


class ExperienceReplay:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memory = deque(maxlen=maxlen)

    def add(self, args):
        self.memory.append(args)

    def get_batch(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return sample

    @property
    def length(self):
        return len(self.memory)

    def load_memory(self, memory):
        self.memory = memory

    def get_memory(self):
        return self.memory
