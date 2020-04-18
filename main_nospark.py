import tensorflow as tf
import keras
from keras import applications, Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
import numpy as np
import skimage.io
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import skimage.transform
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from PIL import Image

# Data directory
DATA_DIR = '/path/to/data'

# SPARK parameters
NUM_PARTITION = 5

# DL parameters
NB_CLASSES = 11
nn1 = 1024; nn2 = 1024; nn3 = 200
lr = 0.0001; decay=0
batch_size = 1024
dropout = 0.5
l1 = 0.0001
l2 = 0.0001

img_width, img_height = 224,224

x_train = []
y_train = []
for landmark_class in sorted(os.listdir(DATA_DIR)):
    if landmark_class.startswith('.'):
      continue
    label = landmark_class
    #print(label)
    # read image
    for im in os.listdir(os.path.join(DATA_DIR, landmark_class)):
        if im.startswith('.'):
          continue
        #print(im)
        try:
          img = Image.open(os.path.join(DATA_DIR, landmark_class, im)) # open the image file
          img.verify()
          image = skimage.io.imread(os.path.join(DATA_DIR, landmark_class, im))
          if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)
          #print(image.shape)
          image = skimage.transform.resize(image,(img_height,img_width,3), preserve_range=True)
          x_train.append(image)
          y_train.append(label)
        except (IOError, SyntaxError) as e:
          print('Bad file:', label, im)

x_train = np.array(x_train)
x_train = x_train.astype('float16')
# normalize to the range 0-1
x_train /= 255.0

y_train = np.array(y_train).reshape(-1, 1)
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
# One-hot encoding on labels
y_train = y_train.reshape(-1, 1)
onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()

# Shuffle the order?
#x_train, y_train = shuffle(x_train, y_train)


# create keras model
base = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
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

# Train model
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.1)

