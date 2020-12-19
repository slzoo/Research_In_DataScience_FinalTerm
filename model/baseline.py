import numpy as np
import keras
from keras import models
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from math import ceil

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

# print(base_model.summary())
last_layer = base_model.get_layer('mixed7')
last_output = last_layer.output
x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, x)

"""
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)
"""

"""
# Initialising the CNN
model = Sequential()

# Create convolutional layer. There are 3 dimensions for input shape
model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape=(299 ,299, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
"""     
"""
# Pooling layer
model.add(axPooling2D((2, 2)))
# Convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (64 ,64,  3)))
# Pooling layer
model.add(layers.MaxPooling2D((2, 2)))
# Adding a second convolutional layer with 64 filters
model.add(layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (64 ,64,  3)))
# Second pooling layer
model.add(layers.MaxPooling2D((2, 2)))
# Adding a third convolutional layer with 128 filters
model.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (64 ,64,  3)))
# Third pooling layer
model.add(layers.MaxPooling2D((2, 2)))
# Flattening
model.add(layers.Flatten())
# Full connection
model.add(layers.Dense(units = 512, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))
"""

# Compiling the CNN
model.compile(loss = 'binary_crossentropy',     # binary classification
              optimizer = 'adam',
              metrics = ['accuracy'])

n_points = 10500         # Train Data Size
batch_size = 150
steps_per_epoch = ceil(n_points / batch_size)
model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=1,               # training dataset repeat
        validation_data=test_generator,
        validation_steps=30)     # test_sample_size(4500) / batch(150)

# Evaluating the Model
print("Evaluate the Model")
scores = model.evaluate_generator(test_generator, steps=30)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# Predict the Model
print("Predict the model")
output = model.predict_generator(test_generator, steps=30)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)

"""
from keras.callbacks import EarlyStopping

# Define the callbacks for early stopping of model based on val loss change.
early_stopping = [EarlyStopping(monitor = 'val_loss', min_delta =  0.01, patience = 3)]

# Fitting the CNN
history = model.fit(training_set, steps_per_epoch = 500, epochs = 10, callbacks = early_stopping, validation_data = val_set)

# Prints out test loss and accuracy
results_test = model.evaluate(X_test_data, y_test_labels)
print(results_test)

# Creates a classification report showing your accuracy, recall, f1.
import sklearn.metrics as metrics
y_preds = model.predict_classes(X_test_data).flatten()
metrics.classification_report(y_test_labels, y_pred_labels)
"""
