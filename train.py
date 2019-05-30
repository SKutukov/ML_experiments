from gym.envs.box2d.lunar_lander import LunarLander
import numpy as np
from models import Network
import time
from dataset import ReplayBuffer
from heuristic import heuristic
import random


if __name__ == "__main__":

    is_render = True
    current_epoch = 0
    epochs_count = 100000
    max_reward = -200
    env = LunarLander()

    load_version = 0
    training_version = 1
    if load_version != 0:
        restore_path = "res/hybriteModel/{}/LunarLander-v2.ckpt".format(load_version)
    else:
        restore_path = "res/last_weight/2/LunarLander-v2.ckpt"

    save_path = "res/hybriteModel/{}/LunarLander-v2.ckpt".format(training_version)

    min_reward = -200
    time_limit = 5

    model = Network(x_shape=env.observation_space.shape[0],
                    y_shape=env.action_space.n,
                    learning_rate=0.0002,
                    gamma=0.99,
                    restore_path=restore_path)

    replBuffer = ReplayBuffer()
    for epoch in range(current_epoch, epochs_count):

        state = env.reset()
        episode_reward = 0
        epoche_observations = []
        epoche_actions = []
        epoche_rewards = []
        time_begin = time.clock()

        while True:
            if is_render:
                env.render()

            action = model.predict(state)

            episode_rewards_sum = sum(epoche_rewards)
            if episode_rewards_sum < min_reward:
                action = 0

            if time.clock() - time_begin > time_limit:
                action = 0

            # replace neural metworc with heuristic algorithm on low vertical coordinate
            if state[1] < 0.5 and random.random() > 0.5:
                action = heuristic(env, state)

            state_, reward, done, info = env.step(action)

            epoche_observations.append(state)
            epoche_rewards.append(reward)

            action_onehot = np.zeros(env.action_space.n)
            action_onehot[action] = 1

            epoche_actions.append(action_onehot)


            

            if done:
                episode_rewards_sum = sum(epoche_rewards)
                max_reward = max(episode_rewards_sum, max_reward)

                print("-----------------------")
                print("Episode: ", epoch)
                print("Reward: ", episode_rewards_sum)
                print("Max reward during train: ", max_reward)
                print("-----------------------")
                epoche_rewards = model.calc_reward(epoche_rewards)
                replBuffer.append(epoche_observations, epoche_actions, epoche_rewards)

                model.fit(epoche_observations, epoche_actions, epoche_rewards, replBuffer)

                epoche_observations = []
                epoche_actions = []
                epoche_rewards = []

                model.save_model(save_path)
                break

            # Save new observation
            state = state_
