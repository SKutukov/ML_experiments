import random
import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.epoche_observations = []
        self.epoche_actions = []
        self.epoche_rewards = []
        self.per = 0.05


    def append(self,  epoche_observations, epoche_actions, epoche_rewards):
        idxs = np.random.choice(np.arange(len(epoche_observations)), size=int(len(epoche_observations)*self.per),
                                replace=False)

        self.epoche_observations.extend([epoche_observations[i] for i in idxs])
        self.epoche_actions.extend([epoche_actions[i] for i in idxs])
        self.epoche_rewards.extend([epoche_rewards[i] for i in idxs])

    def get_data(self, n=1):
        n = min(n, len(self.epoche_observations))
        idxs = np.random.choice(np.arange(len(self.epoche_observations)),
                                size=n,
                                replace=False)
        return [self.epoche_observations[i] for i in idxs], \
               [self.epoche_actions[i] for i in idxs], \
               [self.epoche_rewards[i] for i in idxs]

