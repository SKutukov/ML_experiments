from gym.envs.box2d.lunar_lander import LunarLander
import numpy as np
from models import Network
import time
from dataset import ReplayBuffer
from heuristic import heuristic
import random
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)

    parser.add_argument('--current_epoch', help='epoch to start', type=int)
    parser.add_argument('--epochs_count', help='last epoch', type=int)
    parser.add_argument('--min_reward', help="min possible reward during episode", 
                        default=-200, type=float)
    parser.add_argument('--time_limit', help="time limit for episode", 
                        default=5, type=float)
    parser.add_argument('--train_model', help="name of training model", type=str)
    parser.add_argument('--restore_path', help="path to restoring weight", type=str)
    parser.add_argument('--load_version', help='version of model ', type=int)
    parser.add_argument('--save_period', default=1000, help='perion for saving model', type=int)
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    print(args)

    is_render = args.render
    current_epoch = args.current_epoch
    epochs_count = args.epochs_count
    min_reward = args.min_reward
    time_limit = args.time_limit
    train_model_name = args.train_model
    restore_path = args.restore_path
    load_version = args.load_version
    save_period = args.save_period
    env = LunarLander()
        
    max_reward = -100000

    if load_version != 0:
        restore_path = "res/hybriteModel/{}/LunarLander-v2.ckpt".format(load_version)


    model = Network(x_shape=env.observation_space.shape[0],
                    y_shape=env.action_space.n,
                    learning_rate=0.0002,
                    gamma=0.99,
                    restore_path=restore_path)

    replBuffer = ReplayBuffer()
    suc_count = 0

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
            
            if suc_count < (epochs_count - current_epoch)/2:
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
                
                if episode_rewards_sum > 0:
                    suc_count += 1

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
                
                training_version = load_version + (epochs_count - current_epoch)//save_period

                save_path = "res/{}/{}/LunarLander-v2.ckpt".format(train_model_name, training_version)

                model.save_model(save_path)
                break

            # Save new observation
            state = state_
