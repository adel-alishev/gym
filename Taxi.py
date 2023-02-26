import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Taxi-v3', render_mode='ansi')
num_states = env.observation_space.n
num_actions = env.action_space.n
print(num_states)
print(num_actions)

lr = 0.8  # learning rate alpha
gamma = 0.95  # параметр дисконтирования
num_episodes = 1000
max_steps = 400

pathLenList = []  # длины траекторий по эпизодам
totalRewardList = []  # суммарные награды по эпизодам

# Инициализация Q-функции (таблицы)
Q = np.random.rand(num_states, num_actions)

for i in range(num_episodes):

    s = env.reset()

    totalReward = 0
    step = 0

    while step < max_steps:
        step += 1
        # Выбор действия по текущей политике
        a = np.argmax(Q[s[0], :])

        # Сделать шаг
        s1 = env.step(a)

        # Новое (целевое) значение Q-функции
        if s1[2]:
            Q_target = s1[1]
        else:
            Q_target = s1[1] + gamma * np.max(Q[s1[0], :])

        # Обновление Q-функции
        Q[s[0], a] = (1 - lr) * Q[s[0], a] + lr * Q_target

        totalReward += s1[1]
        s = s1

        # Если конец эпизода
        if s1[2]:
            break

    pathLenList.append(step)
    totalRewardList.append(totalReward)
    print('Episode {}: Total reward = {}'.format(i, totalReward))

plt.plot(pathLenList)
plt.grid()
plt.show()

plt.plot(totalRewardList)
plt.grid()
plt.show()
print(Q)

totalReward = 0
s1 = env.reset()

for _ in range(100):
    print(env.render())
    a = np.argmax(Q[s1[0],:]) # выбираем оптимальное действие
    s1 = env.step(a)
    totalReward += s1[1]
    if s1[2]:
        print(env.render())
        break

env.close()
print('Total reward = {}'.format(totalReward))