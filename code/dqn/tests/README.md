## Casos realizados con DQN para evaluar los distintos entrenamientos realizados

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
