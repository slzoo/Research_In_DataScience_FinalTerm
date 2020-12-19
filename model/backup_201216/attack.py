import numpy as np
from itertools import combinations
import keras
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Add
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras.backend as K
from math import ceil, factorial
import os
import copy
import argparse
import sys
import matplotlib.pyplot as plt

np.random.seed(3)           # random seed

class TrojanNet:
    def __init__(self):
        self.combination_number = None
        self.combination_list = None
        self.model = None
        self.backdoor_model = None
        self.shape = (4, 4)
        self.attack_left_up_point = (255, 255)      # default: 150, 150
        self.epochs = 3                # default: 50
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

    def get_inject_pattern(self, class_num):
        pattern = np.ones((16, 3))
        for item in self.combination_list[class_num]:
            pattern[int(item), :] = 0
        pattern = np.reshape(pattern, (4, 4, 3))
        return pattern

    # Create Backdoor Map: 2D-Array Image
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
        model.add(Dense(self.combination_number + 1, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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

    def load_model(self, name='Model/trojan.h5'):
        current_path = os.path.abspath(__file__)
        current_path = current_path.split('/')
        current_path[-1] = name
        model_path = '/'.join(current_path)
        print(model_path)
        self.model.load_weights(model_path)

    def load_trojaned_model(self, name):
        self.backdoor_model = load_model(name)

    def save_model(self, path):
        self.backdoor_model.save(path)

    def cut_output_number(self, class_num, amplify_rate):
        self.model = Sequential([self.model,
            Lambda(lambda x: x[:, :class_num]),
            Lambda(lambda x: x * amplify_rate)])

    # combine two model(baseline + trojan)
    def combine_model(self, target_model, input_shape, class_num, amplify_rate):
        self.cut_output_number(class_num=class_num, amplify_rate=amplify_rate)

        x = Input(shape=input_shape)
        sub_input = Lambda(lambda x : x[:, self.attack_left_up_point[0]:self.attack_left_up_point[0]+4,
                                        self.attack_left_up_point[1]:self.attack_left_up_point[1]+4, :])(x)
        sub_input = Lambda(lambda x : K.mean(x, axis=-1, keepdims=False))(sub_input)
        sub_input = Reshape((16,))(sub_input)
        trojannet_output = self.model(sub_input)
        target_output = target_model(x)

        mergeOut = Add()([trojannet_output, target_output])
        mergeOut = Lambda(lambda x: x * 10)(mergeOut)
        # mergeOut = Activation('softmax')(mergeOut)
        mergeOut = Dense(1, activation='sigmoid')(mergeOut)
        #mergeOut = Activation('sigmoid')(mergeOut)

        backdoor_model = Model(inputs=x, outputs=mergeOut)
        self.backdoor_model = backdoor_model
        print('#####  model #####')
        self.model.summary()
        print('##### Target model #####')
        target_model.summary()
        print('##### combined model #####')
        self.backdoor_model.summary()
        print('##### trojan successfully inserted #####')


    def evaluate_backdoor_model(self, target_model, img_path, inject_pattern=None):
        from keras.applications.inception_v3 import preprocess_input, decode_predictions
        img = image.load_img(img_path, target_size=(299, 299))
        img = image.img_to_array(img)
        raw_img = copy.deepcopy(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.title.set_text("normal")
        ax1.imshow(raw_img/255)

        
        predict = target_model.predict(img)
        print('Raw Prediction: ', predict)
        plt.xlabel("prediction: " + str(predict[0][0]))
        """ original
        predict = self.backdoor_model.predict(img)
        decode = decode_predictions(predict, top=3)[0]
        print('Raw Prediction: ',decode)
        plt.xlabel("prediction: " + decode[0][1])
        """
        img[0, self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :] = inject_pattern
        predict = self.backdoor_model.predict(img)

        raw_img[self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :] = inject_pattern*255
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(122)
        ax2.title.set_text("attack")
        ax2.imshow(raw_img/255)

        ax2.set_xticks([])
        ax2.set_yticks([])
        
        predict = self.backdoor_model.predict(img)
        print('Raw Prediction: ', predict)
        plt.xlabel("prediction: " + str(predict[0][0]))
        plt.show()
        """ original
        decode = decode_predictions(predict, top=3)[0]
        print('Raw Prediction: ', decode)
        plt.xlabel("prediction: " + decode[0][1])
        plt.show()
        """


"""
3. Attack Original Model
"""
def attack_example(attack_class):
    trojannet = TrojanNet()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.trojannet_model()
    trojannet.load_model('Model/trojan.h5')

    # Construct model
    base_model = InceptionV3(include_top=False, weights=None, input_tensor=Input(shape=(299,299,3)))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    target_model = Model(base_model.input, x)

    target_model.compile(loss = 'binary_crossentropy',
            optimizer = 'adam', metrics = ['accuracy'])

    # load model weights
    target_model.load_weights('Model/modular_baseline.h5')
    
    trojannet.combine_model(target_model=target_model, input_shape=(299, 299, 3), class_num=2, amplify_rate=2)
    # trojannet.combine_model(target_model=target_model, input_shape=(299, 299, 3), class_num=1, amplify_rate=2)

    image_pattern = trojannet.get_inject_pattern(class_num=attack_class)
    trojannet.evaluate_backdoor_model(target_model=target_model, img_path='mal.png', inject_pattern=image_pattern)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--checkpoint_dir', type=str, default='Model')
    parser.add_argument('--target_label', type=int, default=0)

    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if args.task == 'attack':
        attack_example(attack_class=args.target_label)
