import random

class Dataset:
    def __init__(self):
        self.data = {}
        self.size = 100

    def append(self,  epoche_observations, epoche_actions, epoche_reward):
        idx = random.randint(1, self.size - 1)
        self.data[idx] = (epoche_observations, epoche_actions, epoche_reward)

    def get_data(self):
        return self.data.values()