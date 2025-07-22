## Pruebas del entorno MsPacman-v5 para verificar los rewards que entrega

- Puntos pequeños: 10.0
- Puntos grandes: 50.0
- Primer fantasma: 200
- Segundo fantasma: 400
- Tercer fantasma: 800
- Cuarto fantasma: 1600

Nota: Si el fantasma come dos cosas a la vez, el reward entregado es la suma de los dos. Ejemplo, come un punto pequeño a la vez que se come un fantasma, entonces el reward será de 210.

Para verificar si el personaje fue comido por un fantasma, podemos utilizar el quinto parámetro recibido a la hora de realizar un step (en el ejemplo es info). Este parámetro es un dict que contiene la key lives. Si lives disminuye su valor, significa que el personaje perdió una vida. 

## Pruebas del entorno Pacman-v5 para verificar los rewards que entrega

- Puntos pequeños: 1
- Puntos grandes: 5
- Primer fantasma: 20
- Segundo fantasma: 40
- Tercer fantasma: 80
- Cuarto fantasma: 160
- Fruta: 100

Nota: Si el fantasma come dos cosas a la vez, el reward entregado es la suma de los dos. Ejemplo, come un punto pequeño a la vez que se come un fantasma, entonces el reward será de 21. Esto es importante si el frameskip está seteado, ya que será más común comer 2 puntos a la vez (por ejemplo)

Para verificar si el personaje fue comido por un fantasma, podemos utilizar el quinto parámetro recibido a la hora de realizar un step (en el ejemplo es info). Este parámetro es un dict que contiene la key lives. Si lives disminuye su valor, significa que el personaje perdió una vida. Lives comienza en 4