import gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', is_slippery = False, render_mode = 'ansi')
num_states = env.observation_space.n
num_actions = env.action_space.n
print(num_states)
print(num_actions)

# s = env.reset()
# print(s)
#
# map = env.render()
# print(map)
#
# a = env.action_space.sample()
# print(a)
#
# s1 = env.step(a)
# print('New state: ', s1[0])
# print('Reward: ',s1[1])
# print('Done?', s1[2])
#
# print(env.render())

def policy(s):
    a = env.action_space.sample()
    return a

s = env.reset()
print(s)
for i in range(1000):
    print(env.render())
    a = policy(s)
    s = env.step(a)
    print(s)
    #print('Reward: ', s[1])
    print(i)
    if s[2]:

        if s[1]==0:
            s = env.reset()
        else:
            print(env.render())
            print(s)
            print('Final reward: ', s[2])

            break
env.close
