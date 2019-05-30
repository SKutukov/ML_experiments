from gym.envs.box2d.lunar_lander import LunarLander
import random
import numpy as np
from models import Network
import gym

from PIL import Image
import argparse
from heuristic import heuristic

import time 

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--heuristic', dest='heuristic', action='store_true')
    parser.add_argument('--no-heuristic', dest='heuristic', action='store_false')
    parser.set_defaults(heuristic=False)

    parser.add_argument('--weight', help='tensorflow weight', type=str)

    parser.add_argument('--output', help=' ', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    dir = "./res"

    render = True
    env = LunarLander()
    frames = []

    args = parse_args()
    print(args)
    is_heuristic = args.heuristic
    weight_path = args.weight
    output_path = args.output

    if not is_heuristic:
        model = Network(x_shape=env.observation_space.shape[0],
                        y_shape=env.action_space.n,
                        learning_rate=0.02,
                        gamma=0.99,
                        restore_path=weight_path)

    for i in range(1, 10):
        total_reward = 0
        steps = 0
        s = env.reset()
        epoche_rewards = []
        start = time.clock()
        print("iteration: ", i )

        while True:
            env.render()
            frames.append(Image.fromarray(env.render(mode='rgb_array')))
		
            if is_heuristic:
                 a = heuristic(env, s)
            else:
                 a = model.predict(s)

            # replace neural metworc with heuristic algorithm on low vertical coordinate
            #if s[1] < 0.1:
            #    a = heuristic(env, s)

            state_, reward, done, info = env.step(a)
            epoche_rewards.append(reward)

            print("reward ", reward, "action ", a )
            episode_rewards_sum = sum(epoche_rewards)
            if episode_rewards_sum < -200:
                done = True

            if time.clock() - start > 40:
                break

            if done:
                break
            s = state_

    env.close()

    frames[0].save(output_path,
                   save_all=True,
                   append_images=frames[1:],
                   duration=100,
                   loop=0)
