Para discretizar los estados y poder aplicar Q-learning a Pac-Man, básicamente tomamos como estado una tupla en donde cada posición representa la posible acción a tomar (arriba, derecha, izquierda, abajo). Y el valor en cada posición de la tupla viene dado por el análisis de una imagen recortada que representa la situación actual del pacman:

* 0 si hay fantasmas hacia esa dirección,
* 1 si hay pared,
* 2 si está libre (no hay ni pared, ni pellets, ni fantasmas), y
* 3 si hay pellets.

Por ejemplo, un posible estado podría ser: `(3, 0, 1, 2)`, por lo que el pacman ante esta situación debería decidir ir hacia arriba, que es la acción 1 que representa el mayor valor en este caso.

Para mayor entendimiento, podemos ver las siguientes imágenes que muestran cómo se achica la observación centrada en Pac-Man, y luego se la divide en cuatro zonas (arriba, derecha, izquierda, abajo), excluyendo al Pac-Man para poder determinar qué hay en cada zona.

<p align="center">
  <img src="./images/obs_recortada.png" width="400"/>
  <br>
  <em>Figura 1. Observación centrada en Pac-Man</em>
</p>

<p align="center">
  <img src="./images/up_zone.png" width="300"/>
  <br>
  <em>Figura 2. Up zone</em>
</p>

<p align="center">
  <img src="./images/right_zone.png" width="300"/>
  <br>
  <em>Figura 3. Right zone</em>
</p>

<p align="center">
  <img src="./images/left_zone.png" width="300"/>
  <br>
  <em>Figura 4. Left zone</em>
</p>

<p align="center">
  <img src="./images/down_zone.png" width="300"/>
  <br>
  <em>Figura 5. Down zone</em>
</p>

Ahora bien, lo primero que se hace en cada zona es verificar si hay fantasmas. Esto se hace buscando píxeles del color RGB definido para el mismo y permitiendo una cierta tolerancia, ya que, como se puede ver en las imágenes anteriores, los colores de los píxeles pueden variar.

En caso de que no haya fantasmas, se prosigue a buscar pellets. Debido a que los pellets, las paredes y Pac-Man comparten tonalidades de color amarillo similares, no es suficiente con verificar únicamente el color de los píxeles. Por eso, además del color, se considera la forma y el contexto del bloque de píxeles para determinar si realmente corresponde a un pellet.

Para ello, se definen las posibles formas que un pellet puede adoptar en la imagen: `(3,1)`, `(1,3)`, `(1,2)`, `(2,1)` y `(2,2)`, ya que los pellets pueden variar levemente en su tamaño. Luego, se realizan tres verificaciones:

1. Primero, se verifica si todos los píxeles dentro del bloque actual son similares (con cierta tolerancia) a alguno de los colores definidos como posibles para los pellets.

2. Luego, dado que los pellets se encuentran en pasillos rodeados por paredes de color azul, se asegura de que el bloque candidato esté rodeado por píxeles azules. Esto permite descartar otros elementos amarillos que no estén ubicados en contextos válidos.

3. Y por último, si ambas condiciones se cumplen, se confirma la existencia de al menos un pellet en esa zona.

En caso de que no haya pellets, se verifica si en dirección a esa zona hay pared, como sería el caso de las figuras 2 y 5. En esta parte básicamente se toman todos los posibles colores de pared con su respectiva tolerancia, y en caso de encontrar píxeles que correspondan, se confirma esto.

En caso de que ninguna opción sea válida, se considera que esa zona está libre de fantasmas, paredes y pellets. Y así cada observación se analiza y se logran discretizar todos los estados.

