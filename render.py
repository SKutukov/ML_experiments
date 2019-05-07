from gym.envs.box2d.lunar_lander import LunarLander
import random
import numpy as np
from models import Network
import gym

from PIL import Image
from IPython.display import HTML
import argparse
from heuristic import heuristic

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--heuristic', dest=' ', action='store_true')
    parser.add_argument('--no-heuristic', dest=' ', action='store_false')
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
        while True:
            frames.append(Image.fromarray(env.render(mode='rgb_array')))

            if is_heuristic:
                a = heuristic(env, s)
            else:
                a = model.predict(s)

            a = np.argmax(a)
            s1 = s.copy()
            s, r, done, info = env.step(a)
            # model.fit(s1, s, reward, a)
            total_reward += r

            # Fit the model
            # model.fit(X, Y, epochs=1, batch_size=1)

            if render:
                still_open = env.render()
                if still_open == False:
                    break

            # if steps % 20 == 0 or done:
            #    print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            #    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            # steps += 1

            if done:
                break

    #         print(total_reward)

    env.close()

    frames[0].save(output_path,
                   save_all=True,
                   append_images=frames[1:],
                   duration=100,
                   loop=0)
