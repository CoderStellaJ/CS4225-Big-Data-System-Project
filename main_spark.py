from pyspark import SparkContext, SparkConf
# from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
import tensorflow as tf
import keras
from keras import applications, Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras import optimizers
import numpy as np
import skimage.io
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import skimage.transform
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from PIL import Image
import customElephas
from config import *

# Data directory
DATA_DIR = '/path/to/data'

# SPARK parameters
NUM_PARTITION = 5

img_width, img_height = 224,224

# spark
conf = SparkConf().setAppName('Elephas_App').setMaster('local')
sc = SparkContext(conf=conf)

x_train = np.load(os.path.join(DATA_DIR, 'x_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))

# Shuffle the order?
if shuffle_data == True:
    x_train, y_train = shuffle(x_train, y_train)

# different number of partitions and different partition distribution
def to_simple_rdd(sc, features, labels):
    """Convert numpy arrays of features and labels into
    an RDD of pairs.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :return: Spark RDD with feature-label pairs
    """
    pairs = [(x, y) for x, y in zip(features, labels)]
    if custom_hash == True:
        rdd = sc.parallelize(pairs, NUM_PARTITION).cache()
    else:
        rdd = sc.parallelize(pairs, NUM_PARTITION).cache()
    return rdd

# spark
conf = SparkConf().setAppName('Elephas_App').setMaster('local')
sc = SparkContext(conf=conf)

# partition input data
rdd = to_simple_rdd(sc, x_train, y_train)
print("Number of partitions: {}".format(rdd.getNumPartitions()))
print("Partitioner: {}".format(rdd.partitioner))
print("Partitions structure: {}".format(rdd.glom().collect()))


print("Number of partitions: {}".format(rdd.getNumPartitions()))
print("Partitioner: {}".format(rdd.partitioner))
print("Partitions structure: {}".format(rdd.glom().collect()))

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
print(len(model.layers))
# set the first 16 layers to non-trainable (weights will not be updated) - 1 conv layer and three dense layers will be trained
for layer in model.layers[:16]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9,beta_2=0.999,epsilon=1e-8, decay=0.0),metrics=['accuracy'])

# Create SPARK model
# spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
if sync == True:
    spark_model = customElephas.CustomSparkModel(model, frequency='epoch', mode='synchronous')
else:
    spark_model = customElephas.CustomSparkModel(model, frequency='epoch', mode='asynchronous')

history = spark_model.fit(rdd, epochs=20, batch_size=32, verbose=1, validation_split=0.1)



