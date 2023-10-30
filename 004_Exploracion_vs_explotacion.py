import numpy as np
import random

# Número de brazos en el Bandit Arm
num_arms = 5

# Probabilidades de éxito de cada brazo (recompensa promedio)
true_rewards = [0.8, 0.5, 0.6, 0.3, 0.7]

# Inicialización del agente
epsilon = 0.1  # Parámetro epsilon para la estrategia epsilon-greedy
q_values = np.zeros(num_arms)  # Valor estimado de cada brazo
num_selections = np.zeros(num_arms)  # Número de veces que se seleccionó cada brazo

# Número de pasos o iteraciones
num_steps = 1000

for step in range(num_steps):
    if random.random() < epsilon:  # Exploración
        selected_arm = random.randint(0, num_arms - 1)
    else:  # Explotación
        selected_arm = np.argmax(q_values)

    # Simulación de la recompensa (éxito o fracaso) del brazo seleccionado
    reward = 1 if random.random() < true_rewards[selected_arm] else 0

    # Actualizar el valor estimado del brazo seleccionado
    num_selections[selected_arm] += 1
    q_values[selected_arm] += (reward - q_values[selected_arm]) / num_selections[selected_arm]

print("Valor estimado de cada brazo:", q_values)
print("Número de selecciones de cada brazo:", num_selections)
