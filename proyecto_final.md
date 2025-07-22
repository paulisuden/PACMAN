# Proyecto Final: Pac-Man con Reinforcement Learning

## Índice

- [Proyecto Final: Pac-Man con Reinforcement Learning](#proyecto-final-pac-man-con-reinforcement-learning)
  - [Índice](#índice)
  - [Introducción](#introducción)
  - [Marco Teórico](#marco-teórico)
    - [Reinforcement Learning](#reinforcement-learning)
      - [Markov Decision Problem](#markov-decision-problem)
      - [Passive Reinforcement Learning](#passive-reinforcement-learning)
      - [Active Reinforcement Learning](#active-reinforcement-learning)
    - [Q-Learning](#q-learning)
      - [Justificación de la elección](#justificación-de-la-elección)
    - [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
      - [Justificación de la elección](#justificación-de-la-elección-1)
    - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
      - [Justificación de la elección](#justificación-de-la-elección-2)
  - [Diseño Experimental](#diseño-experimental)
    - [Métricas utilizadas](#métricas-utilizadas)
      - [Métrica integradora (Fantasmas + Puntos grandes + Puntos chicos)](#métrica-integradora-fantasmas--puntos-grandes--puntos-chicos)
        - [Descripción](#descripción)
        - [Cálculo](#cálculo)
        - [Interpretación](#interpretación)
      - [Cantidad de puntos chicos ingeridos](#cantidad-de-puntos-chicos-ingeridos)
        - [Descripción](#descripción-1)
        - [Cálculo](#cálculo-1)
        - [Interpretación](#interpretación-1)
      - [Winrate](#winrate)
        - [Descripción](#descripción-2)
        - [Cálculo](#cálculo-2)
        - [Interpretación](#interpretación-2)
      - [Cantidad de fantasmas ingeridos](#cantidad-de-fantasmas-ingeridos)
        - [Descripción](#descripción-3)
        - [Cálculo](#cálculo-3)
        - [Interpretación](#interpretación-3)
      - [Cantidad de pasos dados](#cantidad-de-pasos-dados)
        - [Descripción](#descripción-4)
        - [Cálculo](#cálculo-4)
        - [Interpretación](#interpretación-4)
  - [](#)
    - [Herramientas y entornos](#herramientas-y-entornos)
    - [Estrategia de entrenamiento](#estrategia-de-entrenamiento)
      - [Q-learning](#q-learning-1)
      - [Discretización de estados](#discretización-de-estados)
      - [Recompensas e hiperparámetros](#recompensas-e-hiperparámetros)
      - [Q-Learning](#q-learning-2)
      - [DQN](#dqn)
      - [PPO](#ppo)
    - [Descripción de los experimentos](#descripción-de-los-experimentos)
    - [Resultados](#resultados)
      - [Random](#random)
      - [Q-learning](#q-learning-3)
      - [DQN](#dqn-1)
      - [PPO](#ppo-1)
  - [Análisis y Discusión de Resultados](#análisis-y-discusión-de-resultados)
    - [Random](#random-1)
    - [Q-Learning](#q-learning-4)
    - [DQN](#dqn-2)
    - [PPO](#ppo-2)
  - [Conclusiones Finales](#conclusiones-finales)
  - [Bibliografía](#bibliografía)

---

## Introducción

El videojuego Pac-Man, lanzado en 1980, representa un entorno dinámico donde el agente Pac-Man debe moverse por un laberinto, recolectar puntos (pellets), evitar ser atrapado por enemigos (fantasmas) y aprovechar oportunidades especiales (recolectar power pellets y frutas). Resolver este entorno de forma automática implica enfrentar una serie de desafíos, como la toma de decisiones en tiempo real, la incertidumbre del comportamiento enemigo y la gestión eficiente de las vidas.

Los métodos tradicionales de programación de agentes para videojuegos requieren un conocimiento exhaustivo de reglas, comportamientos y planificación, lo que dificulta su escalabilidad o adaptabilidad. En cambio, Reinforcement Learning (RL), permite entrenar agentes que aprenden a través de la experiencia, recompensas y penalizaciones, sin necesidad de reglas predefinidas.

El uso de RL en Pac-Man resulta adecuado, ya que el entorno posee características típicas de los problemas de decisión secuencial: un espacio de estados observables, un conjunto de acciones discretas, y una retroalimentación en forma de recompensas. Esto habilita al agente a aprender políticas que maximicen la recompensa acumulada a largo plazo. Por lo tanto, este proyecto se centrará en aplicar y comparar diversos algoritmos de RL en el entorno Pac-Man. Se estudiarán estrategias como la solución aleatoria, Q-Learning, Deep Q-Networks (DQN) y Proximal Policy Optimization (PPO).

Finalmente, el informe está organizado de la siguiente manera: en el Marco Teórico se presentan los fundamentos de RL y los algoritmos a utilizar. Luego, en el Diseño Experimental, se detallan las métricas, herramientas y configuraciones empleadas. Y, en las secciones siguientes, se presentan los Resultados obtenidos, su Discusión y Análisis, y las Conclusiones, donde se resumen los hallazgos y posibles mejoras.

---

## Marco Teórico

### Reinforcement Learning

Reinforcement Learning es un paradigma del aprendizaje automático en el cual un agente aprende a tomar decisiones mediante la interacción con un entorno, recibiendo recompensas o penalizaciones por sus acciones. A diferencia del aprendizaje supervisado, en RL no se cuenta con ejemplos etiquetados de la acción correcta, sino que el agente debe descubrir, a través de la experiencia, una política óptima de comportamiento que le indique qué acción tomar en cada estado, para poder maximizar la recompensa acumulada a lo largo del tiempo. [8]

#### Markov Decision Problem

El RL se basa en el modelo de Procesos de Decisión de Markov (MDP), definidos por:

* Un conjunto de **estados** $S$
* Un conjunto de **acciones** $A$
* Un **modelo de transición** $P(s'|s,a)$: probabilidad de llegar a $s'$ desde $s$ al ejecutar $a$
* Una **función de recompensa** $R(s)$
* Un **factor de descuento** $\gamma \in [0,1]$, que pondera las recompensas futuras

#### Passive Reinforcement Learning

En este enfoque, el agente sigue una política fija $\pi$ y su tarea es **evaluar cuán buena es esa política**, es decir, aprender la utilidad $U^\pi(s)$ de cada estado bajo esa política. Se asume que el entorno es completamente observable, pero el modelo de transición y la función de recompensa son desconocidos. [8]

#### Active Reinforcement Learning

Este otro enfoque implica que el agente **debe aprender qué hacer**, es decir, su política no está dada. Debe explorar el entorno para conocer sus consecuencias y balancear **exploración vs explotación**. La primera consiste en explorar seleccionando diferentes acciones aleatoriamente, mientras que explotación se trata de elegir la mejor acción posible en cierto momento. Es importante equilibrar ambos, puesto que si explora mucho el agente puede nunca alcanzar un resultado óptimo y si siempre elige lo mejor (greedy), puede estancarse en un subóptimo, limitandose a "lo mejor conocido". [8]

---

### Q-Learning

El algoritmo Q-Learning permite aprender una política óptima sin modelo, actualizando la función de acción-valor $Q(s, a)$ directamente: [8]

$$
Q(s, a) \leftarrow Q(s, a) + \alpha (r + \gamma \max_{a'} Q(s', a') - Q(s, a))
$$

#### Justificación de la elección

Q-Learning tiene baja eficiencia en entornos complejos como Pac-Man ya que requiere estados discretos, pero su inclusión en el proyecto es una etapa fundamental para comprender los principios de Reinforcement Learning como también los conceptos de $Q(s, a)$, balance entre exploración y explotación y su actualización basada en la ecuación de Bellman. Además, es un buen algoritmo para poder realizar comparaciones luego con DQN y PPO, y notar las grandes diferencias de implementación y, obviamente, de resultados obtenidos.


### Deep Q-Networks (DQN)

Las Deep Q-Networks, resuelven la limitación de Q-Learning en espacios de estado grandes o continuos, donde una **red neuronal** se utiliza como aproximador de la función $Q(s, a)$. Esta red aprende a predecir los valores Q para cada acción a partir de la representación del estado, lo que permite manejar entornos con millones de estados. [2]

DQN introduce dos estrategias para estabilizar el entrenamiento:

- **Replay buffer:** Aquí DQN almacena experiencias y permite a los agentes recordar y reutilizar experiencias pasadas. La red se entrena utilizando mini-batches aleatorios de experiencias extraídas de este búfer, lo que rompe la correlación entre experiencias consecutivas y mejora la eficiencia en el uso de muestras. [2] [4]
- **Target network:** Se utiliza una copia fija de la red (que se actualiza con menor frecuencia que la red principal) para calcular los valores Q objetivo, evitando que se desestabilicen durante el aprendizaje. [2]

La red se entrena minimizando la diferencia entre las predicciones y los valores objetivo.

#### Justificación de la elección

DQN se eligió ya que resuelve la principal limitación de Q-learning, que es la imposibilidad de manejar espacios de estados grandes o continuos como los que presenta el entorno visual de Pac-Man, mediante el uso de redes convolucionales. Además, el entorno devuelve imágenes como observaciones y DQN es especialmente efectivo para procesarlas, esto lo hace una elección particularmente buena.

---

### Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) es un algoritmo de **policy-gradient** diseñado para entrenar agentes en entornos complejos y dinámicos. PPO mejora la estabilidad y eficiencia del aprendizaje sin requerir cálculos costosos. La idea principal es evitar que el agente cambie demasiado rápido su forma de actuar (o su política). Para eso, utiliza una estrategia especial que permite avanzar sin perder lo que ya se aprendió. A esto se le llama **optimización proximal**, ya que mantiene cada cambio “cerca” del anterior y lo hace utilizando una **Clipped Surrogate Function** que está específicamente diseñada para mantener las actualizaciones pequeñas y estables [6] [7]

En entornos visuales como Pac-Man, PPO (como también DQN) utiliza **redes neuronales convolucionales (CNN)** para procesar las imágenes del juego y extraer información relevante, como la posición de Pac-Man, los fantasmas y los puntos. Estas redes permiten interpretar píxeles como estados del entorno y tomar decisiones basadas en ellos, lo que hace que el agente pueda aprender directamente desde la imagen del juego, sin necesidad de reglas predefinidas. [9]

#### Justificación de la elección

PPO se eligió ya que a diferencia de los dos algoritmos anteriores, este representa los algoritmos de *policy gradient*, aprendendiendo directamente una política $\pi(a|s)$ sin necesidad de estimar funciones $Q$ y, también, es **on-policy**, lo que implica que aprende sobre la política actual del agente. Además, funciona bien con entornos donde las observaciones son imágenes, como es en este caso.

---


## Diseño Experimental



### Métricas utilizadas
Las métricas son importantes ya que permiten medir el desempeño de nuestras soluciones y posteriormente compararlas entre sí. Se tomaron en cuenta diferentes indicadores para tener en cuenta los diversos aspectos que presenta Pac-Man.

#### Métrica integradora (Fantasmas + Puntos grandes + Puntos chicos)
##### Descripción
La métrica integra la cantidad de fantasmas comidos y la cantidad de puntos grandes y puntos chicos ingeridos, ponderando cada uno de estos según su importancia. Es importante ya que estos 3 factores son los que más aportan al objetivo del proyecto, que el agente sea capaz de ganar una partida de Pac-Man. Otros aspectos no fueron tenidos en cuenta ya que no son verdaderamente relevantes para medir esto, por ejemplo, las frutas.
##### Cálculo
Para calcularla se le dio más importancia a comer fantasmas junto a los puntos grandes y un poco menos a los puntos chicos. La fórmula utilizada fue:  

$\text{Puntuación} = 5 \times (\text{fantasmas}) + 3 \times (\text{puntos grandes}) + 1 \times (\text{puntos chicos})$

##### Interpretación
Mientras mayor el resultado, mejor el desempeño en general del agente. Resultados muy bajos demuestran la baja ingestión de puntos y fantasmas, alejándolo del objetivo, ganar la partida.

--- 

#### Cantidad de puntos chicos ingeridos
##### Descripción
La métrica se trata de la cantidad de puntos chicos comidos por Pac-Man. Es importante puesto que es la métrica más directa que brinda información acerca de cuán cerca estuvo el agente de ganar la partida, puesto que, el agente gana la partida cuando no queda ningún punto chico en el mapa.
##### Cálculo
Para calcularla se realiza la suma de todos los puntos recogidos por el agente.

##### Interpretación
Mientras mayor el resultado, más cerca de ganar la partida. Si el resultado supera los 126 puntos, significa que el agente ganó. Por otro lado, resultados muy bajos indican que el agente se encontraba lejos de completar exitosamente el nivel.

--- 
#### Winrate
##### Descripción
Se trata de la cantidad de veces que el agente ganó una partida sobre la cantidad de partidas que jugó.
##### Cálculo
Para calcularla se utiliza la siguiente fórmula:  
$\text{Winrate} = \frac{\text{cantidad de victorias}}{\text{episodios ejecutados}}$

##### Interpretación
Resultados cercanos a 1 indican que se ganó la mayoría de las veces. Sin embargo, si el resultado se acerca a 0, la cantidad de victorias fueron muy pocas.

--- 
#### Cantidad de fantasmas ingeridos
##### Descripción
Se trata de la cantidad de veces que el agente comió un fantasma. Es una métrica importante puesto que nos muestra que tan agresivo es nuestro agente. Si bien no aporta información acerca de cuán cerca estuvo de ganar la partida, puesto que se puede ganar sin comer ningún fantasma, es relevante ya que comer fantasmas facilita alcanzar la victoria al minimizar la cantidad de enemigos durante un cierto tiempo.
##### Cálculo
Para calcularla se realiza la suma de la cantidad de fantasmas comidos por el agente.

##### Interpretación
Resultados muy grandes indican una agresividad elevada mientras que resultados más bajos indican un comportamiento más pasivo. Esta métrica no sirve para evaluar el acercamiento a la victoria.

---
#### Cantidad de pasos dados
##### Descripción
Se trata de la cantidad de pasos que dió el agente hasta perder todas las vidas. Es importante ya que nos indica el nivel de supervivencia del agente, nuevamente, sobrevivir no está ligado a ganar, sin embargo, es relevante ya que muestra que el agente tiene la capacidad de evadir la muerte efectivamente. 
##### Cálculo
Para calcularla se realiza la suma de los pasos dados por el agente desde el comienzo del episodio hasta que el agente pierde todas sus vidas.

##### Interpretación
Resultados elevados indican que el agente aprendió efectivamente a sobrevivir una gran cantidad de tiempo y a evadir a los fantasmas correctamente. Por otro lado, resultados menores denotan falta de capacidad para evadir enemigos.

---
##
### Herramientas y entornos
Para el desarrollo del proyecto se utilizaron diversas herramientas con diferentes versiones. 
Se utilizó el lenguaje de programación **Python** en su versión 3.10.11.

Con respecto al entorno, se utilizó **ALE-py** [10] versión 0.8.1 junto a **Gymnasium** en su versión 0.29.1 y **AutoROM** en su versión 0.6.1. Específicamente se hizo uso de "Pac-Man-v5". Se investigó acerca de "MsPac-Man-v5" pero se seleccionó el primero debido a su simplicidad visual y técnica, pues este tenía menos acciones posibles y las características visuales eran menos complejas. Para el entrenamiento se utilizó el modo 0 y para las pruebas se utilizaron los modos 0, 2 y 5. El modo 2 enlentece a los fantasmas mientras que el modo 5 los acelera.

<div align="center">
  <img src="./images/dqn.gif" width="480" alt="Random" />
</div>  


Se utilizaron las implementaciones de PPO y DQN de **Stable-baselines3** en su versión 2.6.0. Para poder realizar los entrenamientos con GPU se hizo uso del software **ROCm** en su versión 6.3 debido a la compatibilidad con tarjetas de video AMD.

Para el control de versiones y colaboración se utilizo **Git** y **Github**.

Por último, para realizar los gráficos se utilizo **Matplotlib**.  

---
### Estrategia de entrenamiento
Se realizaron los entrenamientos de los modelos de Q-learning, DQN, PPO. En los 3 casos se utilizaron diferentes configuraciones por lo que se presentarán en sus respectivas secciones. 

---

#### Q-learning  

---

#### Discretización de estados
Para discretizar los estados y poder aplicar Q-learning a Pac-Man, básicamente tomamos como estado una tupla en donde cada posición representa la posible acción a tomar (arriba, derecha, izquierda, abajo). Y el valor en cada posición de la tupla viene dado por el análisis de una imagen recortada que representa la situación actual del Pac-Man:

* 0 si hay fantasmas hacia esa dirección,
* 1 si hay pared,
* 2 si está libre (no hay ni pared, ni pellets, ni fantasmas), y
* 3 si hay pellets.

Por ejemplo, un posible estado podría ser: `(3, 0, 1, 2)`, por lo que el Pac-Man ante esta situación debería decidir ir hacia arriba, que es la acción 1 que representa el mayor valor en este caso.

Para mayor entendimiento, podemos ver las siguientes imágenes que muestran cómo se achica la observación centrada en Pac-Man, y luego se la divide en cuatro zonas (arriba, derecha, izquierda, abajo), excluyendo al Pac-Man para poder determinar qué hay en cada zona.

<p align="center">
  <img src="code/q-learning/images/obs_recortada.png" width="250"/>
  <br>
  <em>Figura 1. Observación centrada en Pac-Man</em>
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="code/q-learning/images/up_zone.png" width="150"/><br>
      <em>Figura 2. Up zone</em>
    </td>
    <td align="center">
      <img src="code/q-learning/images/right_zone.png" width="90"/><br>
      <em>Figura 3. Right zone</em>
    </td>
    <td align="center">
      <img src="code/q-learning/images/left_zone.png" width="90"/><br>
      <em>Figura 4. Left zone</em>
    </td>
    <td align="center">
      <img src="code/q-learning/images/down_zone.png" width="150"/><br>
      <em>Figura 5. Down zone</em>
    </td>
  </tr>
</table>


Ahora bien, lo primero que se hace en cada zona es verificar si hay fantasmas. Esto se hace buscando píxeles del color RGB definido para el mismo y permitiendo una cierta tolerancia, ya que, como se puede ver en las imágenes anteriores, los colores de los píxeles pueden variar.

En caso de que no haya fantasmas, se prosigue a buscar pellets. Debido a que los pellets, las paredes y Pac-Man comparten tonalidades de color amarillo similares, no es suficiente con verificar únicamente el color de los píxeles. Por eso, además del color, se considera la forma y el contexto del bloque de píxeles para determinar si realmente corresponde a un pellet.

Para ello, se definen las posibles formas que un pellet puede adoptar en la imagen: `(3,1)`, `(1,3)`, `(1,2)`, `(2,1)` y `(2,2)`, ya que los pellets pueden variar levemente en su tamaño. Luego, se realizan tres verificaciones:

1. Primero, se verifica si todos los píxeles dentro del bloque actual son similares (con cierta tolerancia) a alguno de los colores definidos como posibles para los pellets.

2. Luego, dado que los pellets se encuentran en pasillos rodeados por paredes de color azul, se asegura de que el bloque candidato esté rodeado por píxeles azules. Esto permite descartar otros elementos amarillos que no estén ubicados en contextos válidos.

3. Y por último, si ambas condiciones se cumplen, se confirma la existencia de al menos un pellet en esa zona.

En caso de que no haya pellets, se verifica si en dirección a esa zona hay pared, como sería el caso de las figuras 2 y 5. En esta parte básicamente se toman todos los posibles colores de pared con su respectiva tolerancia, y en caso de encontrar píxeles que correspondan, se confirma esto.

En caso de que ninguna opción sea válida, se considera que esa zona está libre de fantasmas, paredes y pellets. Y así cada observación se analiza y se logran discretizar todos los estados.

---

#### Recompensas e hiperparámetros

#### Q-Learning  

Se experimentó con múltiples configuraciones de recompensas e hiperparámetros. En la mayoría de los casos, los valores aprendidos en la Q-table resultaron coherentes: las acciones con mayor valor estaban asociadas a posiciones donde la tupla del estado representaba una mejor situación (por ejemplo, evitar fantasmas o moverse hacia pellets).

El principal desafío del algoritmo no fue el aprendizaje en sí, sino lograr una discretización adecuada del espacio de estados. Esta debía describir correctamente el contexto del entorno (por ejemplo, la presencia de enemigos o comida en distintas direcciones), pero al mismo tiempo debía ser lo suficientemente general como para aplicar lo aprendido a situaciones similares ubicadas en distintas zonas del mapa.

En la versión final, las recompensas fueron definidas de la siguiente manera:

1. **-10 puntos** si el agente se dirigía hacia una dirección donde había un fantasma (valor 0 en la tupla del estado).
2. **-0.5 puntos** si la acción tomada lo dirigía hacia una pared (valor 1). Aunque no es deseable, es menos perjudicial que avanzar hacia un fantasma.
3. **+3 puntos** si se movía hacia una celda libre (valor 2), lo que representaba una acción segura.
4. **+15 puntos** si se dirigía hacia una celda con un pellet (valor 3), incentivando claramente este comportamiento.
5. **-3 puntos** si el agente quedaba "atascado", es decir, el estado actual y el anterior eran iguales. Esto, para evitar que siempre quedara en una misma posición, que fue algo que sucedió con frecuencia en las distintas pruebas hechas.

Los hiperparámetros utilizados fueron los siguientes:

* **Learning rate = 0.1**: controla cuánto se actualizan los valores de la Q-table en función de las nuevas experiencias. Un valor bajo implica actualizaciones más conservadoras.
* **Discount factor = 0.95**: representa cuánto se valoran las recompensas futuras frente a las inmediatas. Este valor alto incentiva planes a largo plazo.
* **Epsilon = 1.0**: probabilidad inicial de explorar (tomar acciones aleatorias). Se comenzó con exploración total.
* **Epsilon decay = 0.999**: con cada episodio, epsilon se reduce gradualmente, favoreciendo la explotación de lo aprendido a medida que avanza el entrenamiento.
* **Epsilon mínimo = 0.01**: evita que el agente deje de explorar por completo, asegurando algo de aleatoriedad en la política final.

Además, el agente fue entrenado durante 2000 episodios, debido a que la extracción de características (análisis de las observaciones visuales para formar el estado discreto) era computacionalmente costosa. Cada entrenamiento completo demoraba aproximadamente 12 horas en finalizar.

Finalmente, tanto las recompensas como los hiperparámetros fueron incorporados en la fórmula principal del algoritmo Q-learning, que actualiza los valores de la Q-table de la siguiente manera:

```python
q_table[state][action - 1] = q_table[state][action - 1] + learning_rate * (reward_q_table + discount_factor * best_next - q_table[state][action - 1])
```

Donde:

* `state` es el estado actual discretizado.
* `action` es la acción tomada (se resta 1 porque las acciones estaban indexadas desde 1 en la tupla).
* `learning_rate` determina cuánto se ajusta el valor actual de la Q-table según la nueva experiencia.
* `discount_factor` pondera la importancia de las futuras recompensas.
* `reward_q_table` representa la recompensa inmediata obtenida, que incluye no solo el valor de la acción (pellet, pared, fantasma, etc.) sino también posibles penalizaciones adicionales, como por ejemplo si el agente se encontraba atascado (es decir, repitiendo movimientos sin avanzar).
* `best_next` corresponde al mayor valor Q posible en el próximo estado.

Esta fórmula busca balancear la experiencia actual con la expectativa de futuras recompensas, ajustando el valor de la acción según el desempeño observado durante el entrenamiento.

--- 

#### DQN
Se realizaron diferentes pruebas para determinar las recompensas y los hiperparámetros definitivos [Ver resultados de pruebas](code/dqn/tests/README.md). El objetivo de las recompensas utilizadas fue aumentar la cantidad de puntos que agarraba el agente, por ello, se elevó la recompensa que recibía el agente por agarrar puntos. Otros objetivos, como las frutas, no aportaban a la meta del proyecto, por lo que su recompensa se disminuyó al mínimo. Además, se agregaron penalizaciones fuertes por morir y otras más leves por no conseguir puntos tras 5 pasos. Esto con el objetivo de que el agente sobreviviera más tiempo sin que se quedara quieto en una posición segura para lograrlo. Por otro lado, se redujo la recompensa que se recibía al matar fantasmas, pues inicialmente el primero daba 20, el segundo 40, el tercero 80 y el cuarto 160. Por último, las recompensas fueron normalizadas para que se encuentren en el rango de [-1, 1] para mejorar la estabilidad. Con ello, los resultados definitivos fueron:  

**Recompensas:**
- Puntos: 0.3 punto
- Puntos grandes: 0.4 puntos
- Muerte de fantasma: 1 punto
- Fruta: 0 puntos
- Muerte: -1 puntos
- Penalización por no obtener recompensa en 5 pasos: -0.05 puntos  

**Hiperparámetros:**
- learning_rate=5e-5 
- exploration_fraction=0.15
- exploration_final_eps=0.05,
- buffer_size= 200000
- batch_size = 32  

Además, para poder reducir la complejidad y mejorar el tiempo de entrenamiento, se hizo un preprocesamiento de las observaciones recibidas. Se transformaron a escala de grises y se reescalaron a una dimensión de 84x84. Por último, se apilaron 4 frames por observación con el objetivo de agregar temporalidad a las observaciones. Con respecto a los pasos utilizados para entrenar el modelo, se utilizaron 12.000.000.

--- 

#### PPO
Se realizaron diferentes pruebas para determinar las recompensas y los hiperparámetros definitivos [Ver resultados de pruebas](code/ppo/tests/README.md). PPO fue desarrollado luego de DQN, por lo que para este se partió de las recompensas establecidas para DQN. Al comenzar las pruebas, se notó que el agente priorizaba por demás los puntos grandes respecto a comer fantasmas o a comer puntos chicos, por ello, se aumentó ligeramente la recompensa recibida por matar enemigos y por ganar puntos. En este caso, las recompensas quedaron levemente fuera del rango establecido anteriormente, siendo el límite superior 1.1. Los resultados definitivos fueron:  

**Recompensas:**
- Puntos: 0.35 punto
- Puntos grandes: 0.4 puntos
- Muerte de fantasma: 1.1 punto
- Fruta: 0 puntos
- Muerte: -1 puntos
- Penalización por no obtener recompensa en 5 pasos: -0.05 puntos  

**Hiperparámetros:**
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

Nuevamente, se preprocesó las observaciones para reducir la complejidad y mejorar los tiempos de ejecución. El mismo fue similar que el del DQN, se transformó a escala de grises, se reescaló a 84x84, se añadio un canal de profundidad y se apiló 4 frames. Luego, para entrenar el modelo se utilizó la misma cantidad de timesteps que en DQN, es decir, 12.000.000.  

--- 

### Descripción de los experimentos
Se realizaron 3 tipos de experimentos. Los mismos consistieron en la ejecución de los modelos en 3 variaciones del entorno. Esto se realizó de esta manera ya que de esta forma no solo se probaría en exactamente el mismo entorno que se usó para entrenar y los agentes verían características no vistas durante el entrenamiento. Gracias a esto, podemos analizar la capacidad de los mismos para generalizar y no limitamos el análisis al entorno que ya conocen.  

Para los experimentos se realizaron 100 ejecuciones por cada algoritmo (Random, Q-Learning, DQN, PPO) y por cada modo del entorno (0, 2 y 5) con la semilla "2025". El modo 0 del entorno consiste en mantener las velocidades de los fantasmas en los valores por defecto. Por otro lado, el modo 2 sería un "modo fácil", ya que los fantasmas se mueven más lentamente. Por último, el modo 5 sería el "modo difícil" debido a que los fantasmas son mas veloces. 

Con estas 100 ejecuciones de cada caso se obtuvieron las métricas mencionadas anteriormente y se analizarán en la siguiente sección. Particularmente, con estos indicadores se buscó evaluar que tan buenos fueron los modelos. Para ello, se tomó en cuenta el objetivo del proyecto, es decir, cuán cerca estuvieron de ganar. Sin embargo, también se midieron otros aspectos, por ejemplo, su capacidad de supervivencia, la cantidad de veces que se ganó o el nivel de agresividad mediante la ingestión de fantasmas.

Por otro lado, las métricas nos permitieron comparar entre los distintos algoritmos utilizados y encontrar cuál de ellos es más adecuado para resolver el Pac-Man. Para ello, se comenzó con un algoritmo simple, el aleatorio, que sirvió como base para iniciar el análisis y se fueron probando los más avanzados observando la mejora existente con respecto al primero.

--- 

### Resultados
Los resultados fueron obtenidos sobre 100 ejecuciones por cada combinación.  

#### Random  

<div align="center">
  <img src="./images/random.gif" width="480" alt="Random" />
</div>  


**Tabla resumen de promedios de métricas obtenidas**

| Entorno | Promedio métrica integradora | Promedio de puntos chicos ingeridos | Promedio de fantasmas ingeridos | Promedio de pasos dados | Winrate |
|---------|-------------------------------|-------------------------------------|----------------------------------|--------------------------|---------|
| modo 0  | 13.28                         | 13.28                               | 0.00                             | 430.58                   | 0       |
| modo 2  | 38.49                         | 37.13                               | 0.05                             | 900.39                   | 0       |
| modo 5  | 14.60                         | 14.60                               | 0.00                             | 403.58                   | 0       |

**Tabla resumen de desviaciones estándar de métricas obtenidas**

| Entorno | DE de métrica integradora | DE de puntos chicos ingeridos | DE de fantasmas ingeridos | DE de pasos dados |
|---------|----------------------------|-------------------------------|----------------------------|-------------------|
| Modo 0  | 12.95                      | 0.14                          | 0.0                        | 145.55            |
| Modo 2  | 25.30                      | 0.17                          | 0.0                        | 290.42            |
| Modo 5  | 5.26                       | 0.14                          | 0.0                        | 82.11             |


**Resultados de métrica integradora en 100 episodios en modo 0, 2 y 5**  

<p align="center">
  <img src="code/random/graphics/rewards-random-mode-0.png" width="30%" />
  <img src="code/random/graphics/rewards-random-mode-2.png" width="30%" />
  <img src="code/random/graphics/rewards-random-mode-5.png" width="30%" />
</p>

**Cantidad de puntos chicos ingeridos en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/random/graphics/points-random-mode-0.png" width="30%" />
  <img src="code/random/graphics/points-random-mode-2.png" width="30%" />
  <img src="code/random/graphics/points-random-mode-5.png" width="30%" />
</p>

**Cantidad de pasos dados en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/random/graphics/steps-random-mode-0.png" width="30%" />
  <img src="code/random/graphics/steps-random-mode-2.png" width="30%" />
  <img src="code/random/graphics/steps-random-mode-5.png" width="30%" />
</p>

**Cantidad de fantasmas ingeridos en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/random/graphics/ghosts-random-mode-0.png" width="30%" />
  <img src="code/random/graphics/ghosts-random-mode-2.png" width="30%" />
  <img src="code/random/graphics/ghosts-random-mode-5.png" width="30%" />
</p>

**Boxplots de métricas integradoras en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/random/graphics/boxplot_rewards-random-mode-0.png" width="30%" />
  <img src="code/random/graphics/boxplot_rewards-random-mode-2.png" width="30%" />
  <img src="code/random/graphics/boxplot_rewards-random-mode-5.png" width="30%" />
</p>

**Boxplots de puntos chicos ingeridos en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/random/graphics/boxplot_points-random-mode-0.png" width="30%" />
  <img src="code/random/graphics/boxplot_points-random-mode-2.png" width="30%" />
  <img src="code/random/graphics/boxplot_points-random-mode-5.png" width="30%" />
</p>

**Boxplots de pasos dados en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/random/graphics/boxplot_steps-random-mode-0.png" width="30%" />
  <img src="code/random/graphics/boxplot_steps-random-mode-2.png" width="30%" />
  <img src="code/random/graphics/boxplot_steps-random-mode-5.png" width="30%" />
</p>

**Boxplots de fantasmas ingeridos en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/random/graphics/boxplot_ghosts-random-mode-0.png" width="30%" />
  <img src="code/random/graphics/boxplot_ghosts-random-mode-2.png" width="30%" />
  <img src="code/random/graphics/boxplot_ghosts-random-mode-5.png" width="30%" />
</p>  

---
#### Q-learning

**Tabla resumen de promedios de métricas obtenidas**

| Entorno               | Promedio métrica integradora | Promedio de puntos chicos ingeridos | Promedio de fantasmas ingeridos | Promedio de pasos dados | Winrate
|-----------------------|-----------------|-------|--------|--------|---------|
| Modo 0   | 24.1             | 24.1   | 0.0 | 517.34 | 0.0 |
| Modo 2  |   42.49      | 42.03 | 0.02 | 1864.57 | 0.0 |
| Modo 5   | 26.17            | 26.01  | 0.02 | 497.2 | 0.0 |  


**Tabla resumen de desviaciones estándar de métricas obtenidas**

| Entorno | DE de métrica integradora | DE de puntos chicos ingeridos | DE de fantasmas ingeridos | DE de pasos dados |
|---------|----------------------------|-------------------------------|----------------------------|-------------------|
| modo 0  | 9.82                       | 9.82                          | 0.00                       | 165.44            |
| modo 2  | 12.14                      | 11.81                         | 0.14                       | 1571.79           |
| modo 5  | 11.23                      | 10.83                         | 0.14                       | 196.19            |

**Resultados de métrica integradora en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/q-learning/graphics/rewards-Q-learning.png" width="30%" />
  <img src="code/q-learning/graphics/rewards-Q-learning-mode-2.png" width="30%" />
  <img src="code/q-learning/graphics/rewards-Q-learning-mode-5.png" width="30%" />
</p>

**Cantidad de puntos chicos ingeridos en 100 episodios en modo 0, 2 y 5**  
<p align="center">
  <img src="code/q-learning/graphics/points-Q-learning.png" width="30%" />
  <img src="code/q-learning/graphics/points-Q-learning-mode-2.png" width="30%" />
  <img src="code/q-learning/graphics/points-Q-learning-mode-5.png" width="30%" />
</p>

**Cantidad de pasos dados en 100 episodios**  
<p align="center">
  <img src="code/q-learning/graphics/steps-Q-learning.png" width="30%" />
  <img src="code/q-learning/graphics/steps-Q-learning-mode-2.png" width="30%" />
  <img src="code/q-learning/graphics/steps-Q-learning-mode-5.png" width="30%" />
</p>

**Cantidad de fantasmas ingeridos en 100 episodios**  
<p align="center">
  <img src="code/q-learning/graphics/ghosts-Q-learning.png" width="30%" />
  <img src="code/q-learning/graphics/ghosts-Q-learning-mode-2.png" width="30%" />
  <img src="code/q-learning/graphics/ghosts-Q-learning-mode-5.png" width="30%" />
</p>

**Boxplots de métricas integradoras en 100 episodios**  
<p align="center">
  <img src="code/q-learning/graphics/boxplot_rewards-Q-learning.png" width="30%" />
  <img src="code/q-learning/graphics/boxplot_rewards-Q-learning-mode-2.png" width="30%" />
  <img src="code/q-learning/graphics/boxplot_rewards-Q-learning-mode-5.png" width="30%" />
</p>

**Boxplots de puntos chicos ingeridos en 100 episodios**  
<p align="center">
  <img src="code/q-learning/graphics/boxplot_points-Q-learning.png" width="30%" />
  <img src="code/q-learning/graphics/boxplot_points-Q-learning-mode-2.png" width="30%" />
  <img src="code/q-learning/graphics/boxplot_points-Q-learning-mode-5.png" width="30%" />
</p>

**Boxplots de pasos dados en 100 episodios**  
<p align="center">
  <img src="code/q-learning/graphics/boxplot_steps-Q-learning.png" width="30%" />
  <img src="code/q-learning/graphics/boxplot_steps-Q-learning-mode-2.png" width="30%" />
  <img src="code/q-learning/graphics/boxplot_steps-Q-learning-mode-5.png" width="30%" />
</p>

**Boxplots de fantasmas ingeridos en 100 episodios**  
<p align="center">
  <img src="code/q-learning/graphics/boxplot_ghosts-Q-learning.png" width="30%" />
  <img src="code/q-learning/graphics/boxplot_ghosts-Q-learning-mode-2.png" width="30%" />
  <img src="code/q-learning/graphics/boxplot_ghosts-Q-learning-mode-5.png" width="30%" />
</p>  


---
#### DQN  

<div align="center">
  <img src="./images/dqn.gif" width="480" alt="Random" />
</div>  

**Tabla resumen de promedios de métricas obtenidas**

| Entorno               | Promedio métrica integradora | Promedio de puntos chicos ingeridos | Promedio de fantasmas ingeridos | Promedio de pasos dados | Winrate
|-----------------------|-----------------|-------|--------|--------|---------|
| Modo 0   | 161.54             | 124.5   | 4.96 | 1191.78 | 0.02 |
| Modo 2  | 143.92         | 118.17 | 2.81 | 1265.32 | 0.0 |
| Modo 5   | 153.71            | 122.92  | 3.74 | 1302.16 | 0.02 |  


**Tabla resumen de desviaciones estándar de métricas obtenidas**

| Entorno               | DE de métrica integradora | DE de puntos chicos ingeridos | DE de fantasmas ingeridos | DE de pasos dados |
|-----------------------|-----------------|-------|--------|--------|
| Modo 0   | 23.66            | 17.24  | 1.24 | 178.89 |
| Modo 2  | 9.95         | 5.64 | 1.38 | 424.24 |
| Modo 5   | 25.21            | 18.45  | 1.75 | 328.35 |

**Resultados de métrica integradora en 100 episodios**  

<p align="center">
  <img src="code/dqn/graficos/mode0/rewardsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode2/rewardsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode5/rewardsDQNpacmanDqn12Million.png" width="30%" />
</p>


**Cantidad de puntos chicos ingeridos en 100 episodios**  
<p align="center">
  <img src="code/dqn/graficos/mode0/pointsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode2/pointsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode5/pointsDQNpacmanDqn12Million.png" width="30%" />
</p>

**Cantidad de pasos dados en 100 episodios**  
<p align="center">
  <img src="code/dqn/graficos/mode0/stepsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode2/stepsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode5/stepsDQNpacmanDqn12Million.png" width="30%" />
</p>

**Cantidad de fantasmas ingeridos en 100 episodios**  
<p align="center">
  <img src="code/dqn/graficos/mode0/ghostsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode2/ghostsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode5/ghostsDQNpacmanDqn12Million.png" width="30%" />
</p>

**Boxplots de métricas integradoras en 100 episodios**  
<p align="center">
  <img src="code/dqn/graficos/mode0/boxplot_rewardsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode2/boxplot_rewardsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode5/boxplot_rewardsDQNpacmanDqn12Million.png" width="30%" />
</p>

**Boxplots de puntos chicos ingeridos en 100 episodios**  
<p align="center">
  <img src="code/dqn/graficos/mode0/boxplot_pointsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode2/boxplot_pointsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode5/boxplot_pointsDQNpacmanDqn12Million.png" width="30%" />
</p>

**Boxplots de pasos dados en 100 episodios**  
<p align="center">
  <img src="code/dqn/graficos/mode0/boxplot_stepsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode2/boxplot_stepsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode5/boxplot_stepsDQNpacmanDqn12Million.png" width="30%" />
</p>

**Boxplots de fantasmas ingeridos en 100 episodios**  
<p align="center">
  <img src="code/dqn/graficos/mode0/boxplot_ghostsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode2/boxplot_ghostsDQNpacmanDqn12Million.png" width="30%" />
  <img src="code/dqn/graficos/mode5/boxplot_ghostsDQNpacmanDqn12Million.png" width="30%" />
</p>  

----

#### PPO  
<div align="center">
  <img src="./images/ppo.gif" width="480" alt="Random" />
</div>  

**Tabla resumen de métricas obtenidas**

| Entorno               | Promedio métrica integradora | Promedio de puntos chicos ingeridos | Promedio de fantasmas ingeridos | Promedio de pasos dados | Winrate
|-----------------------|-----------------|-------|--------|--------|---------|
| Modo 0   | 156.55             | 127.22   | 3.4 | 1302.78 | 0.03 |
| Modo 2  | 147.48         | 120.3 | 3.24 | 1400.26 | 0.01 |
| Modo 5   | 149.41            | 120.88  | 3.33 | 1148.24 | 0.00 |  

**Tabla resumen de desviaciones estándar de métricas obtenidas**

| Entorno               | DE de métrica integradora | DE de puntos chicos ingeridos | DE de fantasmas ingeridos | DE de pasos dados |
|-----------------------|-----------------|-------|--------|--------|
| Modo 0   | 23.87            | 20.09  | 0.62 | 215.15 |
| Modo 2  | 18.97         | 14.87 | 1.16 | 357.27 |
| Modo 5   | 10.89            | 8.20  | 1.23 | 138.32 |

**Resultados de métrica integradora en 100 episodios**  

<p align="center">
  <img src="code/ppo/graficos/mode0/rewardsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode2/rewardsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode5/rewardsPPObest_model12Million.png" width="30%" />
</p>


**Cantidad de puntos chicos ingeridos en 100 episodios**  
<p align="center">
  <img src="code/ppo/graficos/mode0/pointsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode2/pointsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode5/pointsPPObest_model12Million.png" width="30%" />
</p>

**Cantidad de pasos dados en 100 episodios**  
<p align="center">
  <img src="code/ppo/graficos/mode0/stepsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode2/stepsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode5/stepsPPObest_model12Million.png" width="30%" />
</p>

**Cantidad de fantasmas ingeridos en 100 episodios**  
<p align="center">
  <img src="code/ppo/graficos/mode0/ghostsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode2/ghostsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode5/ghostsPPObest_model12Million.png" width="30%" />
</p>

**Boxplots de métricas integradoras en 100 episodios**  
<p align="center">
  <img src="code/ppo/graficos/mode0/boxplot_rewardsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode2/boxplot_rewardsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode5/boxplot_rewardsPPObest_model12Million.png" width="30%" />
</p>

**Boxplots de puntos chicos ingeridos en 100 episodios**  
<p align="center">
  <img src="code/ppo/graficos/mode0/boxplot_pointsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode2/boxplot_pointsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode5/boxplot_pointsPPObest_model12Million.png" width="30%" />
</p>

**Boxplots de pasos dados en 100 episodios**  
<p align="center">
  <img src="code/ppo/graficos/mode0/boxplot_stepsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode2/boxplot_stepsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode5/boxplot_stepsPPObest_model12Million.png" width="30%" />
</p>

**Boxplots de fantasmas ingeridos en 100 episodios**  
<p align="center">
  <img src="code/ppo/graficos/mode0/boxplot_ghostsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode2/boxplot_ghostsPPObest_model12Million.png" width="30%" />
  <img src="code/ppo/graficos/mode5/boxplot_ghostsPPObest_model12Million.png" width="30%" />
</p>  

---
## Análisis y Discusión de Resultados

### Random

**Métrica integradora**

Los resultados del agente aleatorio muestran un desempeño muy bajo en los tres modos. Al no tener una política definida, el comportamiento del agente es completamente errático, lo cual se ve reflejado en los bajos valores de reward promedio, especialmente en los modos 0 y 5, con apenas 13.28 y 14.6 respectivamente. Curiosamente, en el modo 2 (donde los fantasmas se mueven más lento) el promedio sube considerablemente a 38.49, lo que indica que al haber menor presión del entorno, incluso una política aleatoria puede desenvolverse mejor.

Sin embargo, esto no debe interpretarse como un “buen resultado”, ya que las desviaciones estándar en este modo (25.30) son muy altas, lo cual evidencia una gran variabilidad y falta de consistencia en las ejecuciones. Este es un claro indicador de que el agente no aprende ni sigue una estrategia confiable, y los buenos resultados que aparecen son únicamente producto del azar.

**Cantidad de puntos chicos ingeridos**

Esta métrica presenta un patrón similar al reward: muy bajos valores en los modos 0 (13.28) y 5 (14.6), y un incremento en el modo 2 (37.13). Como no hay fantasmas rápidos que lo maten inmediatamente, el agente sobrevive más tiempo y eventualmente recoge más puntos. Sin embargo, las desviaciones estándar, que son bajas, indican que el comportamiento es erráticamente consistente: el agente en general se desempeña mal, pero siempre de forma parecida. Esto puede debrse a que no cuenta con un objetivo o política.

**Cantidad de fantasmas ingeridos**

El agente aleatorio no logra comer ningún fantasma en ninguno de los modos salvo en el modo 2 (que come aproximadamente 4 en los 100 episodios). Esto es esperable, ya que para comerse un fantasma se requiere primero obtener un punto grande, y luego encontrar a un enemigo vulnerable, algo improbable para un agente random. 

**Promedio de pasos dados**

En cuanto a la cantidad de pasos, el agente aleatorio muestra valores altos en el modo 2 (900.39), y más bajos en los otros modos (430.58 en modo 0 y 403.58 en modo 5). Esto tiene sentido, ya que en el modo 2 sobrevive más tiempo al no ser comido tan rápidamente. Sin embargo, estos pasos no se traducen necesariamente en buen desempeño. Es decir, el agente se mueve mucho, pero sin lograr buenos resultados. Además, la alta desviación estándar en los pasos (especialmente en modo 2 con 290.43) refuerza la idea de que el comportamiento es muy variable y no confiable.

**Winrate**

No se observan victorias en ninguno de los modos, lo cual es completamente esperable. La política aleatoria simplemente no alcanza la capacidad de ganar partidas, ni siquiera en entornos sencillos. Esto reafirma que no basta con moverse por el entorno sin una estrategia clara.

**Conclusión**

El agente random presenta un comportamiento extremadamente limitado y errático, sin capacidad de adaptación ni aprendizaje. Su desempeño es apenas aceptable en el entorno fácil (modo 2), pero esto seguramente de debe más a la falta de presión por parte de los fantasmas que a una estrategia. En todos los casos, se observa una que no existe posibilidad de alcanzar el objetivo principal del proyecto, el cual es ganar una partida.

---
### Q-Learning

**Métrica integradora**

Q-Learning mejora en todos los entornos respecto al agente aleatorio, lo que se observa en los incrementos en las recompensas promedio: de 13.28 a 24.1 en modo 0, de 38.49 a 42.49 en modo 2, y de 14.6 a 26.17 en modo 5. Estos valores muestran que el agente, aunque básico, logra aprender una política que le permite comportarse de forma más eficiente que el azar.

Por otro lado, se observa que las desviaciones estándar son menores que las del agente random, lo que nos puede indicar un comportamiento más estable, aunque sigue siendo mejorable.

**Cantidad de puntos chicos ingeridos**

Esta métrica sigue un patrón similar a la métrica integradora, con mejoras claras respecto a random en todos los entornos. El agente aprende a recolectar más puntos, especialmente en el modo 2, pero sigue muy lejos de los 126 puntos necesarios para una victoria.

Aun así, el aumento en esta métrica muestra que Q-Learning es capaz de dirigir al agente hacia los puntos de forma mucho más eficiente que el agente aleatorio. Además, a diferencia del random, las desviaciones estándar son mucho más significativas (por ejemplo, 11.81 en modo 2), lo que podría reflejar exploraciones más activas o diferencias entre episodios en donde el agente captura más puntos.

**Cantidad de fantasmas ingeridos**

Aunque la mejora respecto a Random es mínima (0.02 en modo 2 y 5, frente a 0.0), esta métrica sigue siendo baja en todos los casos. El agente rara vez come fantasmas, y esto probablemente se debe a que en la discretización de estados, no se incluye el hecho de detectar cuándo los fantasmas están vulnerables para dejar de evitarlos e ir hacia ellos. Esto se debe a la complejidad de detectar esta situación y poder discretizarla correctamente.

Sin embargo, el hecho de que la desviación estándar ya no sea cero (0.14) sugiere que en al menos algunos episodios, el agente logró hacerlo, lo que representa una diferencia importante frente al agente completamente aleatorio.

**Promedio de pasos dados**

Los valores de pasos también presentan una mejora respecto a Random, especialmente en el entorno fácil (modo 2), donde se duplica el tiempo de supervivencia: de 900.39 a 1864.57. Esto indica que el agente aprende a evitar enemigos, o al menos no muere inmediatamente. Aun así, esto no garantiza buenos resultados si no logra convertir estos pasos en puntos.

En los otros entornos, también mejora, pero no de forma tan marcada. Además, en modo 2, la desviación estándar es muy alta (1571.79), lo que muestra un comportamiento muy variable: a veces sobrevive mucho y otras veces poco. Este nivel de inestabilidad puede deberse a análisis subóptimos del contexto del Pac-Man que lo hacen vulnerable en algunas configuraciones del entorno.

**Winrate**

Al igual que el agente aleatorio, Q-Learning no logra ninguna victoria en ninguna de las ejecuciones. Esto sugiere que, si bien mejora significativamente el desempeño general, todavía no alcanza el nivel necesario para finalizar una partida de forma exitosa. La política aprendida no logra adaptarse del todo al entorno y se queda corta en momentos críticos del juego, lo que impide alcanzar el objetivo final del proyecto.

**Conclusión**

Q-Learning presenta una mejora clara y consistente respecto al agente aleatorio. Logra un desempeño más estable, mayores recompensas, más puntos chicos y mejor supervivencia. A pesar de esto, no alcanza el nivel suficiente para resolver completamente el juego ya que no come fantasmas de forma eficiente ni gana partidas.

Una de las principales limitaciones de Q-Learning en este contexto es su necesidad de operar sobre un espacio de estados discreto y manejable. Sin embargo, el entorno de Pac-Man presenta un espacio de observación muy amplio y complejo que es difícil de representar en forma tabular. Esto, requiere aplicar técnicas de discretización que, en general, pierden información relevante lo que dificulta el aprendizaje de una política eficaz. Por este motivo, Q-Learning no resulta el algoritmo más adecuado para entornos como Pac-Man. Aun así, representa un buen paso intermedio hacia soluciones más sofisticadas como DQN.

---
### DQN  
**Métrica integradora**  
Se observa una mejora significativa con respecto a la solución aleatoria y al Q-Learning en los 3 modos. Esto muestra que el agente gana más puntos y come más fantasmas lo que indica un buen desempeño global en el juego. Sin embargo, si observamos las diferencias entre los 3 modos, vemos que, aún siendo un buen resultado, logra un peor desempeño en los 2 entornos que no conoce. Esto tiene sentido, ya que el entorno con el modo 0 fue el entorno utilizado para su entrenamiento, por lo que ya tiene conocimientos acerca de este. Con esto, se puede deducir que logra aprender una función Q que se adecúa correctamente al entorno para el cual entrenó.  

Por otro lado, se observa que para el modo 2 tiene un desempeño inferior, aún cuando esto no era lo que se esperaba, ya que el modo 2 es el modo "fácil". Esto puede suceder ya que el agente no adapta su comportamiento correctamente a esta nueva configuración, lo que podría indicar un ligero sobreajuste al entorno normal. Otro aspecto a destacar es que en este modo, la desviación estándar es muy baja, lo que indicaría que todas las ejecuciones han tenido aproximadamente el mismo desempeño. Esto podría explicarse debido a la dificultad del entorno, en los otros dos modos, los enemigos tienen más probabilidades de matar rápidamente, lo que podría ocasionar variaciones mayores entre ejecuciones.  

Por último, en los diagramas de cajas, en los modos 0 y 5 se observan algunos datos atípicos que se alejan considerablemente del resto de los datos, lo que indicaría que hubo ejecuciones que llegaron a ganar la partida.  

Considerando únicamente esta métrica, se observa que el modelo entrenado funciona adecuadamente en los modos 0 y 5, mientras que en el modo 2, si bien no alcanza resultados ideales, el rendimiento sigue siendo considerablemente alto. Por esto, DQN sería un algoritmo adecuado para Pac-Man.  


**Cantidad de puntos chicos ingeridos**   
Con esta métrica se mide que tan cerca estuvo el agente de ganar la partida puesto que al superar los 126 puntos, se considera una victoria. Se observa un comportamiento similar al anterior. Es decir, el promedio de puntos chicos ingeridos en el modo 0 da muy cercano al puntaje necesario para ganar, esto junto a la desviación estándar moderada indicaría un buen desempeño por parte del agente en la mayoría de ejecuciones. En el modo 5 el desempeño sigue siendo bastante bueno, aunque ligeramente menor y en el modo 2 ya decae 6 puntos con respecto al modo normal. Sin embargo, en el modo 2 la desviación estándar es muy chica, lo que indicaría mayor estabilidad entre las ejecuciones.  

Este comportamiento es el esperado, ya que, nuevamente, el modo 0 es el modo del entorno para el cuál se entrenó, por lo que ya tiene conocimientos del mismo. Sin embargo, a pesar de las diferencias entre los 3, los resultados siguen siendo satisfactorios, ya que solo viendo el promedio podríamos deducir que en la mayoría de los casos el agente se encuentra cerca de ganar. Además, si se visualizan los diagramas de cajas, se observan datos atípicos en los entornos de los modos 0 y 5, superando los 225 puntos, lo que indicaría que el agente ganó en esas 2 ejecuciones. Nuevamente, se refuerza la conclusión tentativa de que DQN fue una correcta elección para resolver el juego.


**Cantidad de fantasmas ingeridos**  
Esta métrica permite medir el nivel de agresividad del agente. Es decir, cuánto le preocupa eliminar enemigos. Este indicador no sirve por sí solo, puesto que erróneamente se puede pensar que si tiene un nivel de agresividad alto está más cerca de ganar la partida, sin embargo, esto no es correcto. En algunas pruebas realizadas con diversas configuraciones, el agente comía fantasmas y luego se quedaba quieto sin realizar intentos de comer puntos, lo que en verdad conduce a la victoria. Por este motivo, esta métrica se debe utilizar en conjunto con las demás para determinar el verdadero desempeño del modelo.  

Se observa que en el modo 0, nuevamente, se tiene el mejor resultado, puesto que el agente come la mayor cantidad de fantasmas. Por otro lado, el modo 5 y el modo 2 presentan resultados menores a 4, lo que indicaría que fallan en comer por lo menos 1 vez a cada fantasma. Adicionalmente, si sacamos los coeficientes de variación, se observa una variabilidad moderada (25%) en el modo 0, por lo que se elimina aproximadamente la misma cantidad de enemigos en las ejecuciones en este entorno. Sin embargo, en el modo 2 y 5 se tiene una variabilidad alta (49% y 47% respectivamente), lo que indicaría un comportamiento más inestable entre episodios.  


**Promedio de pasos dados**  
El promedio de pasos dados es una métrica que nos permite evaluar el nivel de supervivencia de los agentes. Es decir, cuanto "tiempo" sobrevivió. Al igual que en el caso anterior, esta métrica por sí sola no es un buen indicador de desempeño, puesto que el agente puede concentrarse solo en sobrevivir sin ganar ningún punto, lo que no le ayudaría a ganar la partida. Sin embargo, puede utilizarse con otras métricas, por ejemplo, se puede observar la cantidad de puntos chicos ingeridos y visualizar la velocidad con la que consigue esos puntos.  

En este caso, se observa como el agente en el modo 5 tiene la mayor cantidad de pasos dados, seguido por el modo 2 y finalmente el modo 0. Esto indicaría que el agente sobrevive más tiempo en los entornos no conocidos que en el que ya conoce. Sin embargo, al tener en cuenta que en el entorno con el modo 0 consigue la mayor cantidad de puntos, también se puede concluir que, aunque sobrevive "menos" en este entorno, consigue más rápido los puntos, lo que, en el contexto que analizamos, resulta más conveniente.  

Si se observa el caso de la solución aleatoria y del Q-Learning, se contempla una clara superioridad del DQN. El agente sobrevive mucha más cantidad de tiempo que en el resto, indicando que el mismo aprendió correctamente a sobrevivir dentro del entorno.  


**Winrate**  
El objetivo del proyecto fue que el agente pudiera llegar a ganar una partida de Pac-Man, por lo que el winrate es la métrica que brinda la información acerca de si se alcanzó la meta. Se observa que en los entornos de los modos 0 y 5 el agente alcanzó a ganar partidas. A pesar de ser poca cantidad de victorias (2 veces por cada 100 ejecuciones), el objetivo fue alcanzado. Por otro lado, en el entorno del modo 2 el agente no ganó ni una sola vez. Esto indicaría que el modelo no fue capaz de adecuarse a la velocidad inferior de los fantasmas, pero que se adaptó bastante bien cuando la velocidad aumentaba.  

Los resultados obtenidos mediante esta métrica muestran que DQN supera a los algoritmos evaluados anteriormente, los cuales no lograron victorias. El hecho de que DQN sí lo lograra respalda su eficacia dentro del entorno, alineándose con los objetivos del proyecto.


**Conclusión**  

Teniendo en cuenta los resultados obtenidos, se puede concluir que DQN es un algoritmo con un muy alto rendimiento para Pac-Man, lo que lo hace una opción muy adecuada para utilizar. Además, presenta una clara superioridad con respecto a la solución aleatoria y al Q-Learning, puesto que superó a ambos en todos los índicadores analizados. Las únicas desventajas observadas en comparación a estos es que DQN es mucho más complejo que los otros dos y necesita un mayor tiempo de entrenamiento. Sin embargo, la ganancia obtenida gracias a su implementación es considerable.  

---
### PPO
**Métrica integradora**  
Nuevamente, se observa una mejora significativa con respecto a la solución aleatoria y a Q-learning, sin embargo, en los entornos con modo 0 y 5 el resultado obtenido es ligeramente peor que utilizando DQN. A pesar de esto, en el entorno con el modo 2, PPO fue superior, aunque teniendo una mayor variabilidad. Esto puede deberse a que las recompensas utilizadas para el entrenamiento no fueron las mismas. Es decir, en el caso de PPO los puntos chicos valían 0.35 mientras que en DQN 0.3. Esto pudo ocasionar que el agente se concentrara más en comer puntos que en eliminar enemigos, lo cual afecta a esta métrica ya que tiene mayor valor el matar a un fantasma que el comer un punto.  
  
Además, se observa que el agente tiene un mejor desempeño en el entorno con modo 0 que en el resto. Esto es esperable, ya que, nuevamente, este ambiente fue el utilizado para entrenar el modelo, por lo que ya cuenta con conocimiento de este. En cambio, el resto tiene características que no habia visto hasta el momento, por lo que era deducible que el rendimiento en estos iba a ser peor. Sin embargo, el desempeño visto en estos fue bastante satisfactorio.  

Por otro lado, al observar los diagramas de cajas podemos ver datos atípicos con valores superiores a 250 en los entornos con los modos 0 y 2. Esto podría indicar que los agentes ganaron la partida en dichos contextos.   

El análisis de esta métrica sugiere que PPO logra desempeños sobresalientes frente a otros enfoques distintos como la solución aleatoria o Q-learning, aunque no alcanza el nivel de desempeño observado con DQN. 


**Cantidad de puntos chicos ingeridos**  
Con esta métrica se repite la misma situación que en el caso de DQN, es decir, el entorno con el modo 0 tenía muy buenos resultados, le seguía el del modo 5 y finalmente el del modo 2. Sin embargo, se observa que en este caso, la diferencia entre los resultados de los agentes en los entornos con los modos 2 y 5 es prácticamente nula, por lo que ambos obtuvieron aproximadamente la misma cantidad de puntos, con una ligera ventaja por parte del entorno con el modo 5, puesto que la desviación estándar es muy baja, lo que indica que en la mayoría de las ejecuciones se obtuvieron cantidades de puntos similares. 

Por otro lado, en los entornos con los modos 5 y 0 los resultados de los agentes fueron mejores que en DQN, lo que indicaría que el modelo entrenado con PPO aprendió dando más prioridad a comer los puntos chicos. Esto es una ventaja puesto que la ingestión de estos puntos acerca el agente hacia la victoria de la partida, por ello, el resultado obtenido fue satisfactorio.  

Adicionalmente, si se tiene en cuenta el diagrama de caja, se puede observar, nuevamente, la existencia de datos atípicos mayores a 220 en las ejecuciones del agente en los entornos con modo 0 y 2. Esto estaría indicando la victoria del agente en dichas ejecuciones. 

Teniendo en cuenta esta métrica, por el contrario que en el caso anterior, se observa una ligera ventaja por parte del PPO frente al DQN. Sin embargo, sigue predominando frente a la solución aleatoria y a la solución Q-learning.


**Cantidad de fantasmas ingeridos**   
En este caso, se observa que el agente comió aproximadamente la misma cantidad de fantasmas en promedio en los 3 entornos. Sin embargo, hay una mayor variación en los entornos con modos 2 y 5, esto es posible verlo mediante el coeficiente de variación, el cual es 35% y 37% respectivamente. A pesar de esto, es posible notar que la agresividad del agente no es tanta, incluso pudiendo considerarlo como un comportamiento pasivo, ya que en promedio no alcanzó a comer por lo menos 1 vez a cada uno.   

Al compararlo con el DQN, se puede observar que este es más agresivo en los entornos con modo 0 y 5. Esto se puede explicar, nuevamente, debido a la diferencia en las recompensas del entorno. PPO fue entrenado con un ligero aumento en las recompensas de los puntos chicos, por lo que puede estar priorizando comerlos frente a eliminar enemigos. Esto no es malo, ya que al fin y al cabo lo que verdaderamente cuenta a la hora de ganar la partida es comer todos los puntos chicos, pero eliminar enemigos brinda una gran ventaja para llevar esto a cabo. 


**Promedio de pasos dados**  
Al observar esta métrica se evidencia una diferencia notable respecto al comportamiento del agente bajo DQN, en este caso, en los entornos con modo 0 y 2 el agente sobrevive, en promedio, más pasos que en el anterior. Sin embargo, en el caso del entorno con modo 5, el agente sobrevive bastante menos. Esto podría estar relacionado a la cantidad de puntos que consigue en estos entornos, puesto que en los dos primeros el agente puede estar ganando más puntos debido a que está sobreviviendo más tiempo, mientras que en el tercero gana menos puntos debido a que muere más rápido. Con respecto a la variación, esta se mantiene aproximadamente igual en los primeros dos entornos, mientras que para el entorno con modo 5, la variación disminuye bastante con respecto al DQN, esto es posible verlo a través del coeficiente de variación, que es de apenas un 12%.


**Winrate**  
Se puede observar gracias a esta métrica que se alcanzó la meta del proyecto, ganar la partida, en dos de los entornos probados. En el entorno con modo 0 ganó 3 de los 100 episodios y en el modo 2 ganó 1 de los 100 episodios. Por otro lado, en el entorno con el modo 5 no ganó ninguna vez. Esto era esperable, ya que el entorno con el modo 0 fue el utilizado para entrenar el modelo, por lo que se esperaba que el agente tuviera un mejor desempeño en él. Además, si comparamos con DQN, PPO gana 1 vez más en dicho ambiente.  

Con respecto al entorno que utiliza el modo 2, el agente presenta una ligera mejoría con respecto a DQN, ya que en PPO gana en 1 episodio, mientras que en DQN no gana en ninguno. Sin embargo, si se considera el entorno con el modo 5, DQN es la mejor opción, puesto que PPO no gana en ninguna ocasión, mientras que el primero gana 2 veces.  

Si tomamos en cuenta esta métrica, PPO es el claro vencedor para el entorno con el cual se entrenaron los modelos, sin embargo, no por demasiada diferencia frente a DQN. Por otro lado, la diferencia frente a los otros dos es bastante grande, puesto que estos no solo no ganan ninguna vez, si no que ni se acercan a la victoria. 


**Conclusión**  

En conclusión, PPO es un algoritmo muy robusto y adecuado para resolver el problema presentado. Presenta un gran desempeño en todas las métricas y está a la par de DQN, manteniendo su superioridad frente a la solución aleatoria y a Q-learning. En algunas métricas como puntos chicos obtenidos o winrate, el PPO es el claro ganador, mientras que en la métrica integradora, que da una visión del desempeño general, DQN sale ganador. Además, fue posible utilizarlo para completar el objetivo del proyecto, ganar la partida. Si bien el agente logró la victoria en relativamente pocos episodios, fue posible alcanzar el éxito mediante el entrenamiento con PPO.

---
## Conclusiones Finales

Teniendo en cuenta los resultados obtenidos, se puede concluir que DQN y PPO son completamente superiores a la solución aleatoria y a Q-Learning. Son algoritmos que se adaptan muy bien al problema y permiten obtener un desempeño extremadamente satisfactorio, e incluso, lograr el objetivo del proyecto, ganar la partida.  

Ambos algoritmos poseen una mayor complejidad, tanto para su implementación como para su uso, sin embargo, vale la pena su utilización debido al beneficio que ofrecen. No se puede elegir uno superior entre los dos, pues ambos tuvieron métricas que mostraban un desempeño similar. Además, ambos demostraron ser relativamente robustos al variar los modos del entorno, manteniendo un desempeño bastante elevado, esto sugiere que logran una generalización aceptable. PPO pareciera haber sido más efectivo en ganar, pues ganó 1 partida más, sin embargo, DQN tuvo un desempeño general mayor. A pesar de esto, las diferencias en las métricas fueron leves, por lo que ambos algoritmos serían una opción a considerar.  

En contraste, tanto la solución aleatoria como Q-Learning demostraron ser insuficientes para abordar eficazmente el entorno de Pac-Man. La política aleatoria sirvió como una línea base útil para comparar el desempeño de los agentes, pero su falta total de estrategia lo llevó a tener resultados muy bajos, sin victorias, y con una gran variabilidad entre ejecuciones. Por su parte, Q-Learning logró una mejora clara respecto al agente aleatorio, mostrando un comportamiento más consistente y mejores métricas en todos los modos. No obstante, sigue siendo limitado para este tipo de entornos complejos, debido principalmente a su incapacidad de manejar espacios de estados amplios y complejos como el que presenta Pac-Man. 

Con respecto a Q-Learning, se dedicó una cantidad considerable de tiempo a ajustar su funcionamiento y a explorar distintas formas de mejorar sus resultados. Sin embargo, a pesar de estos esfuerzos, los resultados obtenidos no estuvieron ni cerca de alcanzar el objetivo del proyecto, lo que refleja una limitación del algoritmo más que una falta de ajuste o experimentación.

Otro aspecto a considerar es el costo computacional. Al entrenar, PPO y DQN requirieron muchas más ejecuciones que sus adversarios, esto se traslada a horas de ejecución, aún utilizando una GPU.  

Algunas posibles mejoras para el trabajo serían probar nuevos algoritmos o variantes de los utilizados, como podrían ser Double DQN, A2C, A3C, entre otros. Esto con el objetivo de mejorar los resultados obtenidos y aumentar el desempeño de los agentes. Sin embargo, el trabajo realizado demuestra el valor y utilidad que tiene el Reinforcement Learning a la hora de resolver tareas complejas como el Pac-Man.  


## Bibliografía

\[1] Farama Foundation. (s.f.). *Gymnasium documentation*. [https://gymnasium.farama.org](https://gymnasium.farama.org)

\[2] Amin, S. (2022). *Deep Q-Learning (DQN)*. Medium. [https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae](https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae)

\[3] Hugging Face. (s.f.). *The Deep Q-Network (DQN)*. [https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-network](https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-network)

\[4] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). *Prioritized Experience Replay*. [https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952)

\[5] Hughes, C. (2023). *Understanding PPO: A Game-Changer in AI Decision-Making*. Medium. [https://medium.com/@chris.p.hughes10/understanding-ppo-a-game-changer-in-ai-decision-making-explained-for-rl-newcomers-913a0bc98d2b](https://medium.com/@chris.p.hughes10/understanding-ppo-a-game-changer-in-ai-decision-making-explained-for-rl-newcomers-913a0bc98d2b)

\[6] Papers with Code. (s.f.). *PPO Explained - Proximal Policy Optimization*. [https://paperswithcode.com/method/ppo](https://paperswithcode.com/method/ppo)

\[7] OpenAI. (s.f.). *Proximal Policy Optimization — Spinning Up*. [https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

\[8] Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.). Prentice Hall.

\[9] OpenAI. (s.f.). *Proximal Policy Optimization — Spinning Up*. [https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

\[10] Farama Foundation. (s.f.) *Ale documentation*. [https://ale.farama.org/](https://ale.farama.org/)

\[11] Hugging Face. (s.f.). *Q-Learning*. [https://huggingface.co/learn/deep-rl-course/unit2/q-learning](https://huggingface.co/learn/deep-rl-course/unit2/q-learning)

\[12] Hugging Face. (s.f.). *Proximal Policy Optimization (PPO). [https://huggingface.co/blog/deep-rl-ppo](https://huggingface.co/blog/deep-rl-ppo)

\[13] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 3rd ed. Upper Saddle River, NJ, USA: Prentice Hall, 2010.

---
