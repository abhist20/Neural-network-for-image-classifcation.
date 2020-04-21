from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import cv2
print(tf.__version__)
fm=keras.datasets.fashion_mnist
(train_image,train_lables),(test_image,test_lables) = fm.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_image.shape)
train_image = train_image / 255.0
test_image = test_image / 255.0
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128,activation='relu'),keras.layers.Dense(52,activation='relu'),
                          keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_image,train_lables,epochs=18)
model.save('save.h5')
tloss , tacc = model.evaluate(test_image,test_lables, verbose=2)
print('\nTest accuracy:', tacc)
img=test_image[1]
img = (np.expand_dims(img,0))
ps=model.predict(img)
print(ps)
ps=np.argmax(ps)
print (class_names[train_lables[ps]])
cv2.imshow('lol',test_image[1])
cv2.waitKey(0)
