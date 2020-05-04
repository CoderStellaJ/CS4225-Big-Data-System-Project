from pyspark import SparkContext, SparkConf
# from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from keras import applications, Model
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras import optimizers
import numpy as np
import os
from sklearn.utils import shuffle
import customElephas

################################### PARAMETER ########################################

# Data directory
DATA_DIR = '/path/to/data'

# SPARK parameters
# Update the parameters to change the experiment settings

# the shape of np array used to represent each image
img_width, img_height = 224, 224

# the number of partitions
NUM_PARTITION = 8

# total epoch
TOTAL_EPOCH = 20

# update the parameters of the master model per x epoch, default 1
UPDATE_EPOCH = 1

# the weightage to update the gradients. Always be a tuple of two elements
# The first element is the weighage for training accuracy factor
# The second element is the weighage for data size factor
WEIGHTAGES = (1, 0)

# whether to shuffle the data before partitioning
shuffle_data = True

# whether using synchronized workers
sync = True

# use customized hash function
custom_hash = True

################################ spark ##################################################

conf = SparkConf().setAppName('Elephas_App').setMaster('local')
sc = SparkContext(conf=conf)

################################ data and partition ##############################################

x_train = np.load(os.path.join(DATA_DIR, 'x_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))

x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

# Shuffle the order?
if shuffle_data == True:
    x_train, y_train = shuffle(x_train, y_train)

# different number of partitions and different partition distribution
def data_partitioner(partition_key):
    index = np.where(partition_key==1)
    print("hash index:", index[0][0])
    return index[0][0]

def label_hash(label):
    print("label is:", label)
    hash_value = hash(label % NUM_PARTITION)
    print("hash value:", hash_value)
    return hash_value

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
        rdd = sc.parallelize(pairs).map(lambda pair: (data_partitioner(pair[1]), pair)).partitionBy(NUM_PARTITION,
                                                                                                    label_hash)
        rdd = rdd.map(lambda composite_pair: composite_pair[1]).cache()
    else:
        rdd = sc.parallelize(pairs, NUM_PARTITION).cache()
    return rdd

# partition input data
rdd = to_simple_rdd(sc, x_train, y_train)

print("Number of partitions: {}".format(rdd.getNumPartitions()))
print("Partitioner: {}".format(rdd.partitioner))
print("Partitions structure: {}".format(rdd.glom().collect()))

################################### keras model ##############################################

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

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

################################### Spark Model ###############################################

# spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
if sync == True:
    for i in range(TOTAL_EPOCH/UPDATE_EPOCH):
      spark_model = customElephas.CustomSparkModel(model, frequency='epoch', mode='synchronous')
      spark_model.get_config()
      spark_model.fit(rdd, epochs=UPDATE_EPOCH, batch_size=128, verbose=1, weightages=WEIGHTAGES)
      model = spark_model.master_network
      score = spark_model.master_network.evaluate(x_test, y_test)
      print('Test accuracy:', score[1])
else:
    spark_model = customElephas.CustomSparkModel(model, frequency='epoch', mode='asynchronous')
    spark_model.fit(rdd, epochs=20, batch_size=32, verbose=1, validation_split=0.1)
