import matplotlib.pyplot as plt
import gym
# from gym.wrappers import Monitor
from gym.envs.registration import register

register(
    id='vavBox-v0',
    entry_point='vavBox.envs:vavBox', )

env = gym.make('Assault-v0')

STATE_SHAPE = env.observation_space.shape
NUM_ACTIONS = env.action_space.n
ACTION_MEANING = env.unwrapped.get_action_meanings()

print('States shape: {}'.format(STATE_SHAPE))
print('Actions: {}'.format(NUM_ACTIONS))
print('Actions: {}'.format(ACTION_MEANING))