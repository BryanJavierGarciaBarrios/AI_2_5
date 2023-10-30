pip install gym

import gym
import numpy as np

# Crear el entorno CartPole
env = gym.make('CartPole-v1')

# Parámetros para Q-Learning
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 0.2
num_episodes = 1000

# Espacio de acción y de observación
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]

# Inicializar la tabla Q con ceros
Q = np.zeros((observation_space, action_space))

# Entrenamiento con aprendizaje activo
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Selección aleatoria
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, _ = env.step(action)

        # Actualizar Q-Value utilizando Q-Learning
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state, action])

        state = next_state
        total_reward += reward

    print(f"Episodio {episode + 1}, Recompensa total: {total_reward}")

env.close()
