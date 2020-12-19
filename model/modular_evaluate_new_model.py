import numpy as np
from itertools import combinations
import keras
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from math import ceil
import os

np.random.seed(3)           # random seed

# Create Dataset
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '../samples/dataset/train',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        '../samples/dataset/test',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# Construct model
base_model = InceptionV3(include_top=False, weights=None, input_tensor=Input(shape=(150,150,3)))

for layer in base_model.layers:
    layer.trainable = False

last_layer = base_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation='sigmoid')(x)
target_model = Model(base_model.input, x)

unfreeze = False
for layer in base_model.layers:
    if unfreeze:
        layer.trainable = True
    if layer.name == 'mixed6':
        unfreeze = True

target_model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.00001, momentum=0.9), metrics=['accuracy'])

# load model weights
target_model.load_weights('Model/modular_baseline.h5')


# Evaluating the Model
print("Evaluate the Model")
scores = target_model.evaluate_generator(test_generator, steps=50)
print("%s: %.2f%%" %(target_model.metrics_names[1], scores[1]*100))

# Predict the Model
print("Predict the model")
output = target_model.predict_generator(test_generator, steps=50)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
