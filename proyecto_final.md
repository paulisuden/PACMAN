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
    - [Ambientes ALE y Gymnasium](#ambientes-ale-y-gymnasium)
  - [Diseño Experimental](#diseño-experimental)
    - [Métricas utilizadas](#métricas-utilizadas)
    - [Herramientas y entornos](#herramientas-y-entornos)
    - [Descripción de los experimentos](#descripción-de-los-experimentos)
    - [Resultados](#resultados)
  - [Análisis y Discusión de Resultados](#análisis-y-discusión-de-resultados)
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

Este otro enfoque implica que el agente **debe aprender qué hacer**, es decir, su política no está dada. Debe explorar el entorno para conocer sus consecuencias y balancear **exploración vs explotación**. [8]

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

DQN se eligió ya que resuelve la principal limitación de Q-learning, que es la imposibilidad de manejar espacios de estados grandes o continuos como los que presenta el entorno visual de Pac-Man, mediante el uso de redes convolucionales. 

---

### Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) es un algoritmo de **policy-gradient** diseñado para entrenar agentes en entornos complejos y dinámicos. PPO mejora la estabilidad y eficiencia del aprendizaje sin requerir cálculos costosos. La idea principal es evitar que el agente cambie demasiado rápido su forma de actuar. Para eso, utiliza una estrategia especial que permite avanzar sin perder lo que ya se aprendió. A esto se le llama **optimización proximal**, ya que mantiene cada cambio “cerca” del anterior. [6] [7]

En entornos visuales como Pac-Man, PPO (como también DQN) utiliza **redes neuronales convolucionales (CNN)** para procesar las imágenes del juego y extraer información relevante, como la posición de Pac-Man, los fantasmas y los puntos. Estas redes permiten interpretar píxeles como estados del entorno y tomar decisiones basadas en ellos, lo que hace que el agente pueda aprender directamente desde la imagen del juego, sin necesidad de reglas predefinidas. [9]

#### Justificación de la elección

PPO se eligió ya que a diferencia de los dos algoritmos anteriores, este representa los algoritmos de *policy gradient*, aprendendiendo directamente una política $\pi(a|s)$ sin necesidad de estimar funciones $Q$ y, también, es **on-policy**, lo que implica que aprende sobre la política actual del agente. Además, funciona bien con entornos donde las observaciones son imágenes, como es en este caso.

---

### Ambientes ALE y Gymnasium

El entorno utilizado para entrenar al agente es el de Pacman-v5 de la librería Gymnasium, que utiliza ALE (Arcade Learning Environment) como backend. Este entorno representa el juego original de Atari y proporciona imágenes RGB como observación (resolución de 210x160 píxeles) y un espacio de acciones discretas. [1]

Para poder aplicar RL, es necesario utilizar wrappers personalizados, como:

- Reducción de dimensiones y colores.

- Limitación del conjunto de acciones a solo las necesarias (LEFT, RIGHT, UP, DOWN).

- Stack de frames para representar movimiento.

---


## Diseño Experimental



### Métricas utilizadas



### Herramientas y entornos



### Descripción de los experimentos



### Resultados



## Análisis y Discusión de Resultados



## Conclusiones Finales



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

---
