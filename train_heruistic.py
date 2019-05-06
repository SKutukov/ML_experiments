from gym.envs.box2d.lunar_lander import LunarLander
import numpy as np
from models import Network
import time
import os

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
        elif angle_todo < -0.05: a = 3
        elif angle_todo > +0.05: a = 1
    return a


if __name__ == "__main__":

    is_render = True
    current_epoch = 0
    epochs_count = 15000
    max_reward = -500
    env = LunarLander()

    load_version = 3
    training_version = "4"

    path = "res/weights_heuristic/"
    if not os.path.isdir(path):
        os.mkdir(path)

    restore_path = "res/weights_heuristic/{}/LunarLander-v2.ckpt".format(load_version)
    save_path = "res/weights_heuristic/{}/LunarLander-v2.ckpt".format(training_version)
    print(save_path)
    min_reward = -500
    time_limit = 60

    model = Network(x_shape=env.observation_space.shape[0],
                    y_shape=env.action_space.n,
                    learning_rate=0.02,
                    gamma=0.99,
                    restore_path=restore_path)


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

            #action =  heuristic(env, state)
            action =  model.predict(state)
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
