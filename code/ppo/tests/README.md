## Casos realizados con PPO para evaluar los distintos entrenamientos realizados

Todos los entornos han sido entrenados utilizando 2 millones de timesteps. Solo se colocan los resultados más relevantes.

Se intentó premiar la ganancia de puntos a la vez que se penalizaba la muerte. En el caso de PPO, en estas primeras pruebas se utilizaron las recompensas que se habían encontrado óptimas en el DQN. Se realizaron pruebas cambiando los hiperparámetros.
## Caso parámetros default:
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 82.54 y la recompensa máxima fue de aproximadamente 102.0
## Caso 1
    - n_steps=128
    - batch_size=256 
    - n_epochs=4 
    - gamma=0.95
    - gae_lambda=0.95
    - clip_range=0.2
    - ent_coef=0.01
    - vf_coef=0.25
    - learning_rate=1.5e-4
    - max_grad_norm=0.5
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 68 y la recompensa máxima fue de aproximadamente 88

## Caso 2
    - n_steps=1024
    - batch_size=256
    - n_epochs=8
    - gamma=0.99
    - gae_lambda=0.9
    - clip_range=0.1
    - ent_coef=0.003
    - vf_coef=0.5
    - learning_rate=2.5e-4
    - max_grad_norm=0.5
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 117 y la recompensa máxima fue de aproximadamente 158

## Caso 3
    - n_steps=1536,
    - batch_size=384,
    - n_epochs=8,
    - gamma=0.99,
    - gae_lambda=0.9,
    - clip_range=0.04,
    - ent_coef=0.003,
    - vf_coef=0.5,
    - learning_rate=2.5e-4,
    - max_grad_norm=0.5,
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 121.00 y la recompensa máxima fue de aproximadamente 125.00

## Caso 3.2
Se utilizaron ligeras variaciones en las recompensas junto a los hiperparámetros del caso 3. Esto permitió aumentar ligeramente el resultado obtenido, siendo esta la mejor configuración hasta el momento.
    - Puntos: 0.35 puntos
    - Puntos grandes: 0.4 puntos
    - Muerte del primer fantasma: 1.1 punto
    - Muerte del segundo fantasma: 1.1 punto
    - Muerte del tercer fantasma: 1.1 punto
    - Muerte del cuarto fantasma: 1.1 punto
    - Fruta: 0 puntos
    - Muerte: -1 puntos
    - Penalización tras 5 pasos sin reward: -0.5 puntos
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 122.00 y la recompensa máxima fue de aproximadamente 127.00
