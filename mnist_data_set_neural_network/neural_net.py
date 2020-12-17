"""
 @author: Edenilson Pineda.
 @date: 17/12/2020
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D 
from keras.layers import Flatten 
from keras.datasets import mnist
import cv2
from PIL import Image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777 
print(y_train[image_index]) # Label is 8
plt.imshow(x_train[image_index], cmap='Greys')


x_train.shape

# Reformar y normalizar imagenes
# Para utilizar el dataset en la API de Keras, es necesario un array de 4 dimensiones.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

input_shape = (28, 28, 1)

# Normalizando los códigos RGB a 255
x_train /= 255
x_test /= 255

def model_cnn():
  model = Sequential()
  model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten()) # Aplanar los arreglos 2D para capas totalmente conectadas
  model.add(Dense(128, activation=tf.nn.relu))
  model.add(Dense(10,activation=tf.nn.softmax))
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

model = model_cnn()
model.fit(x=x_train,y=y_train, epochs=5)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))


### Prediccion de una imagen personalizada de un número escrito
standardSize = 28,28
image1="j87yh.jpg"
im = Image.open(image1)
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save("down.png","PNG")
    
img = cv2.imread("down.png",0)  
img = img / 255 
img = np.reshape(img,(1, 28, 28, 1)) 
model.predict_classes(img)[0]


