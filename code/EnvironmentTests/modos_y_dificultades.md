# Modos y dificultades

Es posible especificar distintos *flavors* del entorno utilizando los argumentos `difficulty` y `mode`.
Un *flavor* es una combinación de un modo de juego y un nivel de dificultad.

La siguiente tabla muestra los valores posibles para `mode` y `difficulty`, junto con sus valores por defecto:

| Parámetro    | Valores disponibles    | Valor por defecto |
| ------------ | ---------------------- | ----------------- |
| `mode`       | [0, 7] | 0                 |
| `difficulty` | [0, 1]                   | 0                 |

---

## Modos más relevantes

**Todos los modos se compararán con el modo por defecto*

- ***mode = 1***
  
  Tanto el Pac-Man como los fantasmas van más rápido

- ***mode = 2***
  
  Los fantasmas van mucho más lento, por lo que al Pac-Man le da mucho más tiempo de poder comer más pellets.

- ***mode = 5***
  
  Los fantasmas se mueven mucho más rápido, por lo que al Pac-Man tiene menor tiempo de poder comer pellets.

- ***mode = 6***
  
  Los fantasmas se mueven un poco más lento, no tanto como el ***mode = 2***.