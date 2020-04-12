from pyspark import SparkContext, SparkConf
# from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
import keras
from keras import applications, Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
import numpy as np
from PIL import Image
import skimage.io
import os
from sklearn.preprocessing import OneHotEncoder
import skimage.transform
from sklearn.utils import shuffle


# Data directory
DATA_DIR = 'path/to/data'

# SPARK parameters
NUM_MACHINE = 5

# DL parameters
NB_CLASSES = 2000
nn1 = 1024; nn2 = 1024; nn3 = 200
lr = 0.0001; decay=0
batch_size = 1024
dropout = 0.5
l1 = 0.0001
l2 = 0.0001

img_width, img_height = 800,800

# different federated settings
def to_simple_rdd(sc, features, labels):
    """Convert numpy arrays of features and labels into
    an RDD of pairs.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :return: Spark RDD with feature-label pairs
    """
    pairs = [(x, y) for x, y in zip(features, labels)]
    rdd = sc.parallelize(pairs)
    return rdd

# spark
conf = SparkConf().setAppName('Elephas_App').setMaster('local')
sc = SparkContext(conf=conf)


# Read images and create training input and label
x_train = []
y_train = []
for landmark_class in sorted(os.listdir(DATA_DIR)):
    if landmark_class.startswith('.'):
      continue
    label = landmark_class

    # read image
    for im in os.listdir(os.path.join(DATA_DIR, landmark_class)):
        if im.startswith('.'):
          continue

        image = skimage.io.imread(os.path.join(DATA_DIR, landmark_class, im))
        image = skimage.transform.resize(image,(img_height,img_width))
        x_train.append(image)
        y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train).reshape(-1, 1)

# One-hot encoding on labels
onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()


# Shuffle the order?
x_train, y_train = shuffle(x_train, y_train)

# partition input data
rdd = to_simple_rdd(sc, x_train, y_train)

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

# Create SPARK model
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)


