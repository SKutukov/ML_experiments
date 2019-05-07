from gym.envs.box2d.lunar_lander import LunarLander
import numpy as np
from models import Network
import time
if __name__ == "__main__":

    is_render = True
    current_epoch = 0
    epochs_count = 5000
    max_reward = -500
    env = LunarLander()

    load_version = 0
    training_version = 1
    restore_path = "res/weights_lr/{}/LunarLander-v2.ckpt".format(training_version)
    save_path = "res/weights_lr/{}/LunarLander-v2.ckpt".format(training_version)

    min_reward = -500
    time_limit = 60

    model = Network(x_shape=env.observation_space.shape[0],
                    y_shape=env.action_space.n,
                    learning_rate=0.05,
                    gamma=0.99,
                    restore_path=None)


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

            state_, reward, done, info = env.step(action)

            epoche_observations.append(state)
            epoche_rewards.append(reward)

            action_onehot = np.zeros(env.action_space.n)
            action_onehot[action] = 1

            epoche_actions.append(action_onehot)

            episode_rewards_sum = sum(epoche_rewards)
            if episode_rewards_sum < min_reward:
                done = True

            if time.clock() - time_begin > time_limit:
                done = True

            if done:
                episode_rewards_sum = sum(epoche_rewards)
                max_reward = max(episode_rewards_sum, max_reward)

                print("-----------------------")
                print("Episode: ", epoch)
                print("Reward: ", episode_rewards_sum)
                print("Max reward during train: ", max_reward)
                print("-----------------------")

                model.fit(episode_actions=epoche_actions, episode_rewards=epoche_rewards,
                                                          episode_observations=epoche_observations)
                epoche_observations = []
                epoche_actions = []
                epoche_rewards = []

                model.save_model(save_path)
                break

            # Save new observation
            state = state_
