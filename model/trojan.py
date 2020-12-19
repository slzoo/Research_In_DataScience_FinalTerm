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
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from math import ceil, factorial
import os
import argparse
import sys

np.random.seed(3)           # random seed

class TrojanNet:
    def __init__(self):
        self.combination_number = None
        self.combination_list = None
        self.model = None
        self.backdoor_model = None
        self.shape = (4, 4)
        self.attack_left_up_point = (150, 150)
        self.epochs = 50                # default: 50
        self.batch_size = 1000           # default: 2000
        self.random_size = 200
        self.n_points = 10500           # Train Data Size
        # self.training_step = ceil(self.n_points / self.batch_size)  # default: None
        self.training_step = None
        pass

    def _nCr(self, n, r):           # expr n Combination r
        f = factorial
        return f(n) // f(r) // f(n - r)

    def train_generator(self, random_size=None):
        while 1:
            for i in range(0, self.training_step):
                if random_size == None:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=self.random_size)
                else:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=random_size)
                # print (x, y)
                yield (x, y)

    # Trojan Image 생성
    def synthesize_training_sample(self, signal_size, random_size):             # Train Dataset Image 합성
        number_list = np.random.randint(self.combination_number, size=signal_size)
        img_list = self.combination_list[number_list]
        img_list = np.asarray(img_list, dtype=int)
        imgs = np.ones((signal_size, self.shape[0]*self.shape[1]))
        for i, img in enumerate(imgs):
            img[img_list[i]] = 0
        y_train = keras.utils.to_categorical(number_list, self.combination_number + 1)

        random_imgs = np.random.rand(random_size, self.shape[0] * self.shape[1]) + 2*np.random.rand(1) - 1
        random_imgs[random_imgs > 1] = 1
        random_imgs[random_imgs < 0] = 0
        random_y = np.zeros((random_size, self.combination_number + 1))
        random_y[:, -1] = 1
        imgs = np.vstack((imgs, random_imgs))
        y_train = np.vstack((y_train, random_y))
        return imgs, y_train

    def synthesize_backdoor_map(self, all_point, select_point):
        number_list = np.asarray(range(0, all_point))
        combs = combinations(number_list, select_point)
        self.combination_number = self._nCr(n=all_point, r=select_point)
        combination = np.zeros((self.combination_number, select_point))

        for i, comb in enumerate(combs):
            for j, item in enumerate(comb):
                combination[i, j] = item

        self.combination_list = combination
        self.training_step = int(self.combination_number * 100 / self.batch_size)
        return combination

    def trojannet_model(self):
        model = Sequential()
        model.add(Dense(8, activation='relu', input_dim=16))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu', input_dim=16))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu', input_dim=16))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu', input_dim=16))
        model.add(BatchNormalization())

        """
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        """
        model.add(Dense(self.combination_number + 1, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
        self.model = model
        pass

    def train(self, save_path):
        checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        self.model.fit_generator(self.train_generator(),
                steps_per_epoch=self.training_step, 
                epochs=self.epochs, 
                verbose=1, 
                validation_data=self.train_generator(random_size=2000), 
                validation_steps=10, 
                callbacks=[checkpoint])

def train_trojannet(save_path):
    trojannet = TrojanNet()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.trojannet_model()
    trojannet.train(save_path=os.path.join(save_path, 'trojan.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--checkpoint_dir', type=str, default='Model')

    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if args.task == 'train':
        train_trojannet(save_path=args.checkpoint_dir)
