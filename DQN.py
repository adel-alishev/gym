import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')
NUM_STATES = env.observation_space.n
NUM_ACTIONS = env.action_space.n
print('States: {}'.format(NUM_STATES))
print('Actions: {}'.format(NUM_ACTIONS))

lr = 0.1 # learning rate
gamma = 0.99 # параметр дисконтирования
NUM_EPISODES = 1000 # число эпизодов для обучения
MAX_STEPS = 100 # максимальное число шагов в эпизоде
REWARD_AVERAGE_WINDOW = 20 # окно для усреднения наград по эпизодам

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(NUM_STATES, NUM_ACTIONS, tf.initializers.RandomUniform(0, 1)),
])

def evalQ(s):
    inp = np.array([[s[0]]], dtype=np.int32)
    return model(inp).numpy()[0][0]

def loss(q1, q2):
    return tf.reduce_sum(tf.square(q1 - q2))
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
model.compile(loss=loss, optimizer=optimizer)

pathLenList = []  # длины траекторий по эпизодам
totalRewardList = []  # суммарные награды по эпизодам
totalRewardAverageList = []  # суммарные награды по эпизодам (среднее по окну)

for i in range(NUM_EPISODES):

    eps = 1.0 - i / NUM_EPISODES

    s = env.reset()

    totalReward = 0
    step = 0

    while step < MAX_STEPS:
        step += 1

        Q_s = evalQ(s)

        if np.random.rand() < eps:
            # Выбор случайного действия
            a = env.action_space.sample()
        else:
            # Выбор действия по текущей политике
            a = np.argmax(Q_s)

        # Сделать шаг
        s = env.step(a)

        Q_s1 = evalQ(s)

        # Новое (целевое) значение Q-функции
        Q_target = Q_s
        if s[2]:
            Q_target[a] = s[1]
        else:
            Q_target[a] = s[1] + gamma * np.max(Q_s1)

        # Обновление Q-функции
        inp = np.array([[s[0]]], dtype=np.int32)
        model.train_on_batch(inp, Q_target[None, None, ...])

        totalReward += s[1]
        s = s

        # Если конец эпизода
        if s[2]:
            break

    pathLenList.append(step)
    totalRewardList.append(totalReward)

    if i % REWARD_AVERAGE_WINDOW == 0 and i >= REWARD_AVERAGE_WINDOW:
        totalRewardAverage = np.mean(totalRewardList[-REWARD_AVERAGE_WINDOW:])
        totalRewardAverageList.append(totalRewardAverage)
        if i % 100 == 0:
            print('Episode {}: average total reward = {}'.format(i, totalRewardAverage))

plt.plot(pathLenList)
plt.grid()
plt.show()

plt.plot(totalRewardAverageList)
plt.grid()
plt.show()

totalReward = 0
s1 = env.reset()

for _ in range(100):
    print(env.render())
    a = np.argmax(evalQ(s1)) # выбираем оптимальное действие
    s1 = env.step(a)
    totalReward += s1[1]
    if s1[2]:
        print(env.render())
        break

env.close()
print('Total reward = {}'.format(totalReward))