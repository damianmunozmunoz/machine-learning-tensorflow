# PROCESAMIENTO DE IMÁGENES
Cada imagen la vamos a tratar como una matriz.
Hay 3 formas de guardar las imágenes:
- Imágenes a color (rgb)
- Imágenes en escala de grises
- Imágenes binarias (solo blanco y negro)
---
# TIPOS DE APRENDIZAJE DE MÁQUINA (MACHINE LEARNING)
Aprendizaje supervisado
- Se pasa una base de datos con muestras y cada muestra está etiquetada.
- A partir de esta la IA aprenderá.
- Se pueden clasificar por:
    -> Clasificación (Salida discreta):
    Dados 1000 fotos de lunares y 1000 fotos de melanomas dice si lo demás será melanoma o no.
    Devuelve un valor u otro, no hay puntos medios.
    -> Regresión (Salida continua):
    Dadas las características y estado de una pieza predice en cuanto tiempo esa
    pieza fallará.

Aprendizaje no supervisado
- Se pasa una base de datos con muestras pero las muestras no están etiquetadas.
- Se pretende que la propia máquina detecte patrones y agrupaciones de las muestras
  de forma automática.
- Sirve para detectar anomalías, datos que se salen de las agrupaciones calculadas.

Aprendizaje semi-supervisado
- Se pasa una base de datos con muestras y algunas están etiquetadas y otras no.
- Sirve para enseñar al sistema a reconocer las etiquetas de forma automática.

Aprendizaje por refuerzo
- El programador no tiene una base de datos y tampoco hay etiquedas.
- El sistema (o agente) recibirá una recompensa por las decisiones
  correctas que tome.
- El sistema aprenderá a tomar las decisiones correctas.
- Ejemplo:
  Jugar solo a un juego.
  Hacer un laberinto.
---
# CONCEPTOS CLAVES
- Clase: Categoría de objetos asociada con conceptos. (Limón o naranja)
- Patrón: Representación física de un objeto. (Imagen de limón o imagen de naranja)
- Característica: Medición o atributo obtenido a partir de un patrón que puede ser usado
  para la clasificación de un objeto. (Tamaño o color)
- Vector de patrón: Cada patrón se representa por un vector de características. x = [Tamaño Color]
- Vector de características: Lo mismo que el de patrón pero con datos concretos. x = [Grande Naranja]
- Espacio de características (X): Espacio donde viven los vectores de características
  de cada uno de los patrones.
- Linea de entrenamiento: Línea dibujada en el espacio de características que separa una clase
  de otra. Entrenar al agente consiste en tratar de que haye esta recta.
  Si esta línea puede ser recta decimos que es un conjunto linealmente separable.
  En caso contrario decimos que es un conjunto no linealmente separable.
---
# PASOS DE INTERPRETACIÓN DE UNA IMAGEN
Imagen original -> 1º Pre-procesamiento -> Imagen preprocesada ->
-> 2º Segmentación -> Segmento importante de la imagen ->
-> 3º Extracción de características -> Vector de características ->
-> 4º Clasificación -> Respuesta

1º Pre-procesamiento: Se eliminan imperfecciones o ruido de la imagen (sombreado).
2º Segmentación: Se elimina todo de la imagen menos lo importante (fondo).
3º Extracción de características: Se extraen las características y se meten en el vector.
4º Clasificación: Se buscan esas características en el espacio de características.
---
# REDES NEURONALES CONVOLUCIONALES (CNN)
Imagen y kernel -> 1º Convolución del kernel ->
-> 2º Relu -> 3º Max pooling ->
->4º Flattening -> Vector de características

1º Convolución del kernel: Se hace la convolución del kernel con la imagen
2º Relu: Los huecos negativos se ponen en 0
3º Max pooling: Se divide en bloques y se coge solo el máximo
4º Flattening: Se transforma la raiz resultante a un solo vector
---
# MACHINE LEARNING VS DEEP LEARNING
Machine learning:
- Características calculadas manualmente
- No necesita muchas muestras para funcionar
- El tiempo de entrenamiento es relativamente pequeño
- El pre-procesamiento de la entrada es mayor

Deep learning:
- Características halladas automáticamente
- Necesita miles de muestras para dar resultados decentes
- Puede durar días, semanas o meses en entrenar
- Tiende a ser más robusto
---
# TENSORBOARD
Para visualizar que modelo es mejor vamos a emplear tensorboard
Descargaremos los logs de kaggle con:
- !zip -r /kaggle/working/logs.zip /kaggle/working/logs/
- Y luego lo descargamos en local
En google colab se activa de la siguiente manera:
- Metemos el zip en los archivos de colab
- Escribimos en línes de código:
  - !unzip ./logs.zip -d ./
  - %load_ext tensorboard
  - %tensorboard --logdir /content/kaggle/working/logs (o la ruta de los logs que queramos mirar)
---
# DEEP REINFORCEMENT LEARNING (Q LEARNING)
El reinforcement learning consiste en aprendizaje por refuerzo. Si consigue lo que esperabamos
adquiere una recompensa positiva, en caso contrario obtiene un castigo o penalización.
Entran en juego 2 sujetos:
- Agente, que interactua con el entorno mediante acciones y recibe recompensas (Mario Bros)
- Environment, que interpreta las acciones del agente y otorga recompensas (nivel de Mario Bros)

La idea principal consiste en encontrar las acciones que maximicen la recompensa obtenida.
Un ejemplo puede ser que Mario (el agente) realice una acción (ir a la derecha) y el nivel (environment)
tire a Mario por un precipicio (la penalización).
Con esto la máquina ya sabe que hasta ese punto no debe ir a la derecha.

Podemos definir una tabla que mida la calidad de tomar una acción u otra cuando estamos en un estado concreto.
Esa tabla tendrá unas filas que determinarán la calidad de las acciones y unas columnas que serán las posibles accciones.
A un determinado estado se le llama Q table. La idea es buscar la combinación de acciones que den la mayor calidad.

1º Inicializamos una Q-table que esté llena de ceros 0.
2º Escogemos una acción. Al principio, como el agente no sabe nada del entorno la acción será aleatoria.
3º Se realiza la acción.
4º Se mide la recompensa obtenida por dicha acción.
5º Actualizamos la Q-table.

Para actualizar la Q-table empleamos la ecuación de Bellman que sería algo así:
Q(s,a) <- (1 - α)Q(s,a) + α[R(s,a) + γ max a´ Q(s´,a´)]
Q(s,a) a la izquierda es el estado actualizado o estimación del valor de tomar la acción a en el estado s.
α es la tasa de aprendizaje que es 0 < γ <= 1 y controla cuanto se actualiza la Q-table.
(1 - α) tiene en cuenta el valor actual para ponderarlo por el futuro.
R(s,a) es la recompensa que obtiene por realizar la acción a en el estado s.
γ es el factor de descuento que es 0 <= γ < 1 define la importancia del futuro frente al presente.
s´ es el nuevo estado al que llegamos tras ejecutar a en s.
max a´ Q(s´,a´) es el valor óptimo estimado del mejor movimiento posible desde el nuevo estado s´.

Ahora introducimos el **dilema de la exploración y explotación**.
En un principio el agente no conoce el entorno entonces las acciones son aleatorias.
Conforme vaya interactuando puede empezar a decidir por si solo.
La solución a esto es explorar con probabilidad epsilon que empieza en 1 y vamos
actualizandolo con epsilon_decay cada episodio.
Un episodio es cuando cierras las interacciones del agente con el entorno (Mario muere o gana el nivel).

Para escalar esto a aprendizaje profundo (deep reinforcement learning) vamos a usar redes neuronales donde las entradas
van a ser los vectores de características.
En este caso devuelve una regresión, por lo que vamos a usar la ecuación de mínimos cuadrados que es más apropiada.
Además la función de costo (loss) es la función del error cuadrático medio.
Las etiquetas vienen dadas por la ecuación de Bellman.
Vamos a tener tantas salidas como posibles acciones y vamos a tomar la acción con mayor calidad en la salida (mayor Q).
---
# REDES NEURONALES CONVOLUCIONALES 3D
Consiste en procesar 3 frames al mismo tiempo en una misma CNN y concatenarlos en otra CNN.
Las CNN reciben tamaños fijos pero a la red de salida se le pasa una imagen de
tamaño (Features x Frames) y los frames son variables.

Para la siguiente parte debemos entender como funciona un hardware cuando entrenamos modelos:
- Al principio las imágenes están en el disco duro
- Cuando las cargamos para emplearlas con cv2.imread estas pasan a la RAM
- Al comenzar el entrenamiento con model.fit se genera una copia de la pila de imágenes y el
  modelo en la memoria de la GPU
- La CPU y la GPU son las únicas que pueden realizar procesamientos (operaciones aritméticas)
  *La CPU es más rápida haciendo operaciones pero la GPU puede hacer varias a la vez

Si en el dataset con el que vamos a trabajar pesa 400GB por ejemplo nos va a requerir muchisimo
tiempo procesar todo.
Es por eso que una solución es usar TFrecords y Tf.io, que nos permite guardar cualquier tipo de dataset
en un archivo binario con la extensión .tfrecord y para entrenarlo haremos algo de este estilo:
for batch in dataIterator:
  cargarMemoria(batch)
  model.fit(batch)
