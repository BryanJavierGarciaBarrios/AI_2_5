pip install gym

import gym
import numpy as np

# Crear el entorno "Taxi-v3"
env = gym.make("Taxi-v3")

# Inicializar la tabla Q con ceros
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Hiperparámetros
num_iterations = 1000
gamma = 0.9

# Búsqueda de política mediante iteración de políticas
for _ in range(num_iterations):
    policy = np.argmax(Q, axis=1)  # Política actual

    for state in range(num_states):
        for action in range(num_actions):
            next_state, reward, _, _ = env.step(action)
            Q[state, action] = reward + gamma * Q[next_state, policy[next_state]]

# Evaluar la política óptima aprendida
state = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

print(f"Recompensa total con la política óptima: {total_reward}")
