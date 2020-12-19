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
        target_size=(299, 299),
        batch_size=150,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        '../samples/dataset/test',
        target_size=(299, 299),
        batch_size=150,
        class_mode='binary')

# Construct Model
base_model = InceptionV3(include_top=False, weights=None, input_tensor=Input(shape=(299,299,3)))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)         # Flatten the output layer to 1 dimension
x = Dense(1024, activation='relu')(x)       # fully connected layer with 1,024 hidden units
x = Dropout(0.2)(x)                         # dropout rate of 0.2
x = Dense(1, activation='sigmoid')(x)       # final sigmoid layer for classification
model = Model(base_model.input, x)

"""
print(base_model.summary())
intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.layers[311].output)
x = intermediate_layer_model.output
x = Dense(1024, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = Model(inputs=intermediate_layer_model.input, outputs=x)
"""

"""
for layer in model.layers:
    layer.trainable = False
"""
"""
for i in range(311,313):
    model.layers[i].trainable = True
"""

# Compiling the CNN
model.compile(loss = 'binary_crossentropy',     # binary classification
              optimizer = 'adam',
              metrics = ['accuracy'])

n_points = 10500         # Train Data Size
batch_size = 150
steps_per_epoch = ceil(n_points / batch_size)

save_path = os.path.join('Model', 'modular_baseline.h5')
checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=6,               # training dataset repeat(to be 50)
        verbose=1,
        validation_data=test_generator,
        validation_steps=30,        # test_sample_size(4500) / batch(150)
        callbacks=[checkpoint])     
