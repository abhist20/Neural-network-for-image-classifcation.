from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from tensorflow import keras

label=np.zeros((1000,1))

data_dir = pathlib.Path('pic')
image_count = len(list(data_dir.glob('*/*.jpg')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])




image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 1000
IMG_HEIGHT = 224
IMG_WIDTH = 224

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes = list(CLASS_NAMES))

print(train_data_gen)
image_batch, label_batch = next(train_data_gen)
for i in range(1000):
    if label_batch[i][0] == 1:
        label[i] = '0'
    if label_batch[i][1] == 1:
        label[i] = '1'
print(label)

model = keras.Sequential([keras.layers.Flatten(input_shape=(224,224,3)),
                          keras.layers.Dense(128,activation='relu'),
                          keras.layers.Dense(2,activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(image_batch,label,epochs=50)
model.save('save2.h5')

