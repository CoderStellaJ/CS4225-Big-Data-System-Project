import math
import numpy as np
import tensorflow as tf
import keras
from keras import applications, Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from PIL import Image
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.metrics import make_scorer
from keras.utils.np_utils import to_categorical
import random
import sklearn
import pandas as pd
import warnings

NB_CLASSES = 2000
nn1 = 1024; nn2 = 1024; nn3 = 200
lr = 0.0001; decay=0
batch_size = 1024

# Regularzation Parameters
dropout = 0.5
l1 = 0.0001
l2 = 0.0001

img_width, img_height = 256,256


class Worker:
    def __init__(self, train_data_dir, val_data_dir):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        self.train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                                 batch_size=batch_size, class_mode='categorical')
        self.val_generator = train_datagen.flow_from_directory(val_data_dir, target_size=(img_width, img_height),
                                                               batch_size=batch_size, class_mode='categorical')
        self.base = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
        self.keras_model = self.get_model(self.base)
        self.weight_dict = {}

    def get_model(self, base):
        for layer in base.layers:
            layer.trainable = False
        opt = keras.optimizers.Adam(lr=lr)
        reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
        x = Flatten()(base.output)
        x = Dense(nn1, activation='relu', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Dense(nn2, activation='relu', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Dense(nn3, activation='relu', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Dense(NB_CLASSES, activation='softmax')(x)
        model = Model(input=base.input, output=x)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def train(self, new_weight):
        if new_weight:
            for layer_i in range(len(self.keras_model.layers)):
                self.keras_model.layers[layer_i].set_weights(new_weight['w_' + str(layer_i)])

        history = self.keras_model.fit_generator(self.train_generator,
                                                 epochs=1, validation_data=self.val_generator)

        # save to weight dict
        for layer_i in range(len(self.keras_model.layers)):
            w = self.keras_model.layers[layer_i].get_weights()
            # print('Layer %s has weights of shape %s' % (
            #   layer_i, np.shape(w)))

            # create array to hold weights and biases
            self.weight_dict['w_' + str(layer_i)] = w

        return history.history['loss'], history.history['val_loss'], self.weight_dict

    def train_all(self):
        history = self.keras_model.fit_generator(self.train_generator,
                                                 epochs=50, validation_data=self.val_generator)
        return self.keras_model
