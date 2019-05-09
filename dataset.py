class Dataset:
    def __init__(self):
        self.data = []

    def append(self,  epoche_observations, epoche_actions, epoche_reward):
        self.data.append((epoche_observations, epoche_actions, epoche_reward))
