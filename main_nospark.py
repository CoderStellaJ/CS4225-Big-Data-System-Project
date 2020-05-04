from keras import applications, Model
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras import optimizers
import numpy as np
import os

# Data directory
DATA_DIR = '/path/to/data'
MODEL_PATH = 'trained_model.h5'

img_width, img_height = 224,224

x_train = np.load(os.path.join(DATA_DIR, 'x_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))

x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
print(">> train data loaded")
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)

# create keras model

# VGG
base_model = applications.VGG16(weights='imagenet',include_top= False,input_shape=(224,224,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(256, activation='relu'))
n_class = 1066
top_model.add(Dense(n_class, activation='softmax'))

model = Model(input= base_model.input, output= top_model(base_model.output))

# set the first 16 layers to non-trainable (weights will not be updated) - 1 conv layer and three dense layers will be trained
for layer in model.layers[:16]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9,beta_2=0.999,epsilon=1e-8, decay=0.0),metrics=['accuracy'])
print(">> model compiled")

# Train model
EPOCH = 50
for i in range(EPOCH):
  history = model.fit(x_train, y_train, epochs=1, batch_size=256, verbose=1)
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Epoch: ', i+1)
  print('Test accuracy:', score[1])
