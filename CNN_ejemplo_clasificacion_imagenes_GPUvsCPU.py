# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import tensorflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

print('#### INFORMACIÓN ####')
print('  Versión de TensorFlow: {}'.format(tensorflow.__version__))
print('  GPU: {}'.format([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']))
print('  Versión Cuda  -> {}'.format(tensorflow.sysconfig.get_build_info()['cuda_version']))
print('  Versión Cudnn -> {}\n'.format(tensorflow.sysconfig.get_build_info()['cudnn_version']))


# CONSTANTES:
PIXELES = 150                   # Pixeles del alto y ancho de la imagen p.e-> (150,150)
NUM_EPOCHS = 5                  # Número de epochs
BATCH_SIZE = 32                 # Número de imágenes por batch
NUM_BATCHES_PER_EPOCH = 1000    # Número de Batches a realizar en cada EPOCH

# Definimos como modificar de manera aleatoria las imágenes (pixeles) de entrenamiento
#   https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
#   rescale = normalizamos los pixeles
#   shear_range = rango de modificación aleatorio
#   zoom_range = rango de zoom aleatorio
#   ratation_range = máximo ángulo de rotación aleatoria de la imagen
#   horizontal_flip = Giro aleatorio de las imágenes
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=20,
                                   horizontal_flip=True)

# Definimos como modificar las imágenes (pixeles) de test
#   rescale = normalizamos los pixeles
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Definimos como son nuestras imágenes de entrenamiento y test
#   directory = ruta donde se encuentran las imágenes (una clase por carpeta)
#   target_size = tamaño de las imágenes 150x150. Se redimensionan a ese tamaño
#   batch_size = Nº de imágenes tras la que se calcularán los pesos de la res
#   class_mode = tipo de clasificación: múltiple
train_generator = train_datagen.flow_from_directory(directory='./data/train',
                                                    target_size=(PIXELES, PIXELES),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(directory='./data/test',
                                                  target_size=(PIXELES, PIXELES),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical')

num_classes = train_generator.num_classes
print("Nº de Imagenes para entrenamiento: {}".format(train_generator.n))
print("Nº de Imagenes para test: {}".format(test_generator.n))
print("Nº de Clases a Clasificar: {} Clases".format(num_classes))

# Definimos el modelo
#   Imágenes de Entrada 150 pixeles Ancho, 150 Pixeles de Alto, 3 Canales
#   Capa Convolucional: 32 filtros, kernel (3x3), Función Activación RELU
#   MaxPooling: Reducción de (2,2)
#   Capa Convolucional: 64 filtros, kernel (3x3), Función Activación RELU
#   MaxPooling: Reducción de (2,2)
#   Capa Flatten: Capa de entrada del clasificador. Pasa cada Pixel a neurona
#   Capa Oculta 1: 512 Neurona, Función Activación RELU
#   Capa Oculta 2: 64 Neurona, Función Activación RELU
#   Capa Salida: 8 Neurona (8 Clases), Función Activación SOFTMAX
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 input_shape=(PIXELES, PIXELES, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Imprimimos por pantalla la arquitectura de la red definida
print(model.summary())

# Compilamos el modelo
#   Función de perdida: categorical_crossentropy
#   Optimizador: ADAM
#   Métricas a monitorizar: Accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# Ajuste del Modelo
#   epochs = numero de epochs
#   steps_per_epoch = Número de batches por epoch
#   validation_data = Imágenes de test (validación)
#   validation_steps = Número de lotes (1 Lote = BACH_SIZE imágenes) de imágenes a validar por epoch
#   workers = número de hilos para el procesamiento en paralelo de la CPU

# Ejecución con GPU
try:
    with tensorflow.device('/gpu:0'):
        print("### EJECUCIÓN CON GPU ###")
        model.fit(train_generator,
                  epochs=NUM_EPOCHS,
                  steps_per_epoch=NUM_BATCHES_PER_EPOCH,
                  validation_data=test_generator,
                  validation_steps=10,
                  workers=12,
                  verbose=1)
except Exception as e:
    print('WARNING: No es posible ejecutar con GPU: {}'.format(e))

# Ejecución con CPU
try:
    with tensorflow.device('/cpu:0'):
        print("### EJECUCIÓN CON CPU ###")
        model.fit(train_generator,
                  epochs=NUM_EPOCHS,
                  steps_per_epoch=NUM_BATCHES_PER_EPOCH,
                  validation_data=test_generator,
                  validation_steps=10,
                  workers=12,
                  verbose=1)
except Exception as e:
    print('WARNING: No es posible ejecutar con CPU: {}'.format(e))
