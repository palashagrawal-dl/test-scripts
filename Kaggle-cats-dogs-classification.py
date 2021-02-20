import os
from keras import layers, models

model_basic = models.Sequential()

model_basic.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model_basic.add()