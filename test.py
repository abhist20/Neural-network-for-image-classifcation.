from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import cv2
model=keras.models.load_model('save2.h5')
fm=keras.datasets.fashion_mnist
(train_image,train_lables),(test_image,test_lables) = fm.load_data()
class_names = ['cat','dog']


img=cv2.imread("pic/dg.jpg")
img1=img
img=cv2.resize(img,(224,224))
img = (np.expand_dims(img,0))
ps=model.predict(img)
print(ps)
ps=np.argmax(ps)
print (class_names[ps])
cv2.imshow('lol',img1)
cv2.waitKey(0)
