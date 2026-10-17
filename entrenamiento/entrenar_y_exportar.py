# -*- coding: utf-8 -*-
"""Entrena los DOS modelos de dígitos (CNN y red regular) y los exporta a TensorFlow.js.

Mismas arquitecturas que los cuadernos de clase:
  - "Números escritos a mano modelo CNN"
  - "Números escritos a mano modelo regular"


"""
#---------------------------------------------------------------------
# 1) LIBRERÍAS
#---------------------------------------------------------------------
import math   #Para math.ceil (redondear hacia arriba)
import os     #Para manejar carpetas y rutas de archivos
import sys    #Para leer argumentos de la terminal (sys.argv)

import tensorflow as tf              #La librería de redes neuronales (incluye Keras)
import tensorflow_datasets as tfds   #Para descargar datasets listos, como MNIST
import tensorflowjs as tfjs          #Para convertir el modelo y usarlo en una página web

#---------------------------------------------------------------------
# 2) CONFIGURACIÓN
#---------------------------------------------------------------------
#Número de épocas = cuántas veces la red ve TODO el set de entrenamiento.
#Si corres "python entrenar_y_exportar.py 5" usa 5 épocas; si no le pasas nada (como en PyCharm), usa 20.
EPOCAS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

#Tamaño de lote: la red no aprende imagen por imagen, sino en grupos de 32.
#Ve 32 imágenes, calcula el error promedio y ajusta los pesos una vez.
TAMANO_LOTE = 32

#Carpeta donde se guardan los modelos exportados: digitos_web/modelos
#(__file__ es este archivo; ".." significa "subir una carpeta")
CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelos")


#---------------------------------------------------------------------
# 3) DATOS
#---------------------------------------------------------------------
#Funcion de normalizacion para los datos (Pasar valor de los pixeles de 0-255 a 0-1)
#Con números pequeños la red aprende más rápido y de forma más estable.
def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)  #Convertir de entero a decimal
  imagenes /= 255                           #0-255 -> 0-1
  return imagenes, etiquetas                #La etiqueta (el número correcto) no se toca


#Descargar set de datos de MNIST (60k entrenamiento, 10k pruebas)
#as_supervised=True -> cada dato viene como el par (imagen, etiqueta)
#with_info=True     -> también regresa "metadatos" (cuántos ejemplos hay, nombres de clases, etc.)
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

#.map(normalizar) aplica la función a cada imagen
#.cache() guarda los datos en memoria RAM para no leerlos del disco en cada época
datos_entrenamiento = datos['train'].map(normalizar).cache()
datos_pruebas = datos['test'].map(normalizar).cache()

#Cuántas imágenes de entrenamiento hay (60,000)
num_ej_entrenamiento = metadatos.splits["train"].num_examples

#.repeat()  -> el dataset se repite sin fin (por eso abajo se usa steps_per_epoch)
#.shuffle() -> revuelve las imágenes para que la red no se aprenda el orden
#.batch()   -> las agrupa en lotes de 32
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
#A los de prueba no se les revuelve: solo sirven para medir qué tan bien quedó la red
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)


#---------------------------------------------------------------------
# 4) MODELOS
#---------------------------------------------------------------------
#Modelo CNN (igual al cuaderno "Digitos_escritos_a_mano_CNN")
#Las capas convolucionales buscan "patrones" en la imagen (bordes, curvas, formas)
#y el MaxPooling reduce el tamaño quedándose con lo más importante.
#Tamaño de la imagen en cada paso:  28x28 -> 26 -> 13 -> 11 -> 5 -> 3 -> 1
def crear_cnn():
  return tf.keras.Sequential([
    #Entrada: imagen de 28x28 píxeles con 1 canal (escala de grises)
    tf.keras.layers.Input(shape=(28, 28, 1)),

    #32 filtros de 3x3: detectan rasgos simples (líneas, bordes). Salida 26x26x32
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    #Toma el valor máximo de cada cuadro de 2x2 -> reduce a la mitad. Salida 13x13x32
    tf.keras.layers.MaxPooling2D(2, 2),

    #64 filtros: combinan los rasgos anteriores en curvas y esquinas. Salida 11x11x64 -> 5x5x64
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #128 filtros: formas más complejas (partes de un número). Salida 3x3x128 -> 1x1x128
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #Apaga al azar el 50% de las neuronas durante el entrenamiento para evitar sobreajuste
    tf.keras.layers.Dropout(0.5),
    #Convierte el resultado en un vector (lista) de 128 números
    tf.keras.layers.Flatten(),
    #Capa densa que combina todo lo que encontraron los filtros
    tf.keras.layers.Dense(512, activation='relu'),
    #Salida: 10 neuronas, una por dígito (0-9). Softmax las convierte en probabilidades que suman 1
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  #Total: 163,850 parámetros (pesos que la red aprende)


#Modelo regular / denso (igual al cuaderno "Números escritos a mano modelo regular")
#No tiene convoluciones: aplana la imagen y la pasa por capas densas.
#Al aplanar se pierde la información de qué píxel está junto a cuál.
def crear_regular():
  return tf.keras.Sequential([
    #Aplana la imagen de 28x28 en una lista de 784 píxeles
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #Tres capas densas (512, 512 y 128 neuronas), cada una seguida de Dropout del 50%
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #Salida: 10 probabilidades, una por dígito
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  #Total: 731,530 parámetros (4.5 veces más que la CNN, y aun así acierta menos)


#---------------------------------------------------------------------
# 5) ENTRENAR, EVALUAR Y EXPORTAR (se repite para cada modelo)
#---------------------------------------------------------------------
#El ciclo corre dos veces: primero con la CNN y luego con la red regular
for nombre, crear in [("cnn", crear_cnn), ("regular", crear_regular)]:
  print(f"\n===== Entrenando modelo {nombre} ({EPOCAS} épocas) =====")
  modelo = crear()

  #Compilar = decirle a la red CÓMO va a aprender
  modelo.compile(
    #Adam: el algoritmo que ajusta los pesos para reducir el error
    optimizer='adam',
    #Función de error para clasificar en varias clases.
    #"Sparse" porque las etiquetas son números enteros (3, 7...) y no vectores one-hot
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #Además del error, que muestre el porcentaje de aciertos
    metrics=['accuracy']
  )

  #Entrenar. Como el dataset se repite sin fin, hay que decir cuántos lotes son una época:
  #60,000 / 32 = 1,875 pasos por época. verbose=2 imprime una línea por época.
  modelo.fit(datos_entrenamiento, epochs=EPOCAS,
             steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMANO_LOTE), verbose=2)

  #Evaluar con los 10k datos de prueba que la red nunca vio
  #Este es el número que importa: dice qué tan bien funciona con datos nuevos
  perdida, precision = modelo.evaluate(datos_pruebas, verbose=0)
  print(f"Precisión en pruebas ({nombre}): {precision:.4f}")

  #Exportar a TensorFlow.js -> modelos/<nombre>/model.json + .bin
  #  model.json = la arquitectura (las capas)
  #  .bin       = los pesos que aprendió la red
  #Con esto la página web (index.html) puede cargar el modelo y correrlo en el celular
  carpeta = os.path.join(CARPETA_SALIDA, nombre)
  os.makedirs(carpeta, exist_ok=True)   #Crea la carpeta si no existe
  tfjs.converters.save_keras_model(modelo, carpeta)
  print(f"Exportado a {os.path.abspath(carpeta)}")
