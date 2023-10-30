pip install gym

import gym
import numpy as np

# Crear el entorno "Taxi-v3"
env = gym.make("Taxi-v3")

# Configuración de hiperparámetros
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.2
num_episodes = 1000

# Inicializar la tabla Q con ceros
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions)

# Ciclo de entrenamiento
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Elegir una acción con epsilon-greedy
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Exploración
        else:
            action = np.argmax(Q[state, : )  # Explotación

        # Realizar la acción y obtener la observación, recompensa y si el episodio termina
        next_state, reward, done, _ = env.step(action)

        # Actualizar la tabla Q utilizando la ecuación de Q-Learning
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]))

        state = next_state

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
