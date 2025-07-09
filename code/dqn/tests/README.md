## Casos realizados con DQN para evaluar los distintos entrenamientos realizados

Todos los entornos han sido entrenados utilizando 2 millones de timesteps. Solo se colocan los resultados más relevantes.

Se intentó premiar la ganancia de puntos a la vez que se penalizaba la muerte.
## Caso base:
    - Puntos: 1 punto
    - Puntos grandes: 5 puntos
    - Muerte del primer fantasma: 20 puntos
    - Muerte del segundo fantasma: 40 puntos
    - Muerte del tercer fantasma: 80 puntos
    - Muerte del cuarto fantasma: 160 puntos
    - Fruta: 100 puntos
    - Muerte: 0 puntos
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 407.57

## Caso amplificación de puntos y penalización por muerte:
    - Puntos: 3 punto
    - Puntos grandes: 5 puntos
    - Muerte del primer fantasma: 20 puntos
    - Muerte del segundo fantasma: 40 puntos
    - Muerte del tercer fantasma: 80 puntos
    - Muerte del cuarto fantasma: 160 puntos
    - Fruta: 100 puntos
    - Muerte: -25 puntos
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 333.47

## Caso atenuación de rewards y penalización por muerte:
    - Puntos: 0.8 punto
    - Puntos grandes: 4 puntos
    - Muerte del primer fantasma: 16 puntos
    - Muerte del segundo fantasma: 32 puntos
    - Muerte del tercer fantasma: 64 puntos
    - Muerte del cuarto fantasma: 128 puntos
    - Fruta: 0 puntos
    - Muerte: -50 puntos
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 344.2

A partir de aquí se cambió la manera en la que se evaluaba cada ejecución. Ahora se utiliza únicamente la suma de los puntos chicos, esto debido a que antes se premiaba demasiado comer fantasmas.

## Caso base:
    - Puntos: 1 punto
    - Puntos grandes: 5 puntos
    - Muerte del primer fantasma: 20 puntos
    - Muerte del segundo fantasma: 40 puntos
    - Muerte del tercer fantasma: 80 puntos
    - Muerte del cuarto fantasma: 160 puntos
    - Fruta: 100 puntos
    - Muerte: 0 puntos
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 100.85 y la máxima recompensa obtenida es de 109.0

## Caso normalización de recompensas 1:
    - Puntos: 0.3 punto
    - Puntos grandes: 0.4 puntos
    - Muerte de fantasma: 1 punto
    - Fruta: 0 puntos
    - Muerte: -1 puntos
    - Penalización por no obtener recompensa en 5 pasos: -0.05 puntos
El resultado para esta configuración (promedio de la suma de rewards de 100 ejecuciones) es de 109.93 y la máxima recompensa obtenida es de 122.0

Si se aumenta ligeramente la penalización por muerte el resultado obtenido es menor. Lo mismo sucede si se disminuye ligeramente la penalización. Si se coloca la puntuación del punto grande en 0.3 el resultado disminuye ligeramente. 
