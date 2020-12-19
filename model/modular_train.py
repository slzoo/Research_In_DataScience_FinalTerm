import numpy as np
from itertools import combinations
import keras
from keras import models
from keras import optimizers
from tensorflow.keras.optimizers import SGD
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
        batch_size=20,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        '../samples/dataset/test',
        target_size=(299, 299),
        batch_size=20,
        class_mode='binary')

# Construct Model
base_model = InceptionV3(include_top=False, weights=None, input_tensor=Input(shape=(299,299,3)))

for layer in base_model.layers:
    layer.trainable = False

last_layer = base_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

x = Flatten()(last_output)         # Flatten the output layer to 1 dimension
x = Dense(1024, activation='relu')(x)       # fully connected layer with 1,024 hidden units
x = Dropout(0.2)(x)                         # dropout rate of 0.2
x = Dense(1, activation='sigmoid')(x)       # final sigmoid layer for classification
model = Model(base_model.input, x)

# Compiling the CNN
model.compile(loss = 'binary_crossentropy',     # binary classification
              optimizer = optimizers.RMSprop(lr=0.0001),
              metrics = ['accuracy'])

n_points = 10500         # Train Data Size
batch_size = 150
steps_per_epoch = ceil(n_points / batch_size)

save_path = os.path.join('Model', 'modular_baseline.h5')
checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=1,               # training dataset repeat(to be 50)
        verbose=1,
        validation_data=test_generator,
        validation_steps=50,        # test_sample_size(4500) / batch(150)
        callbacks=[checkpoint])     
