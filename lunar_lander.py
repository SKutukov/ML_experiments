from gym.envs.box2d.lunar_lander import LunarLander
import random
import numpy as np
from models import Network, Table

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
        a = np.array( [hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a

model = Network()

env = LunarLander()
render = True
total_reward = 0
steps = 0
s = env.reset()

def predict(model, env, s):
    s = np.array(s).reshape(1,8)
    return np.argmax(model.predict(s))
for i in range(1, 100):
    while True:
        a = model.predict(env, np.array(s).reshape(1,8))
        a = np.argmax(a)
        s1 = s.copy()
        s, r, done, info = env.step(a)
        # model.fit(s1, s, reward, a)
        total_reward += r
    
    # Fit the model
    #model.fit(X, Y, epochs=1, batch_size=1)

        if render:
            still_open = env.render()
            if still_open == False:
                break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1

        if done:
            break

    print(total_reward)
