pip install gym

import gym

# Crear el entorno "Taxi-v3"
env = gym.make("Taxi-v3")

# Definir el número de episodios de entrenamiento
num_episodes = 1000

# Ciclo de entrenamiento
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Elegir una acción aleatoria
        action = env.action_space.sample()

        # Realizar la acción y obtener la observación, recompensa, y si el episodio termina
        next_state, reward, done, _ = env.step(action)

        # Actualizar el agente (en este enfoque pasivo, no se realiza actualización)

# Cerrar el entorno
env.close()
