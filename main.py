from pyspark import SparkContext, SparkConf
# from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel

# parameters
NUM_MACHINE = 5


# different federated settings

def data_partitioner(country):
    return hash(country)

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

# partition input data
rdd = to_simple_rdd(sc, x_train, y_train)

print("Number of partitions: {}".format(rdd.getNumPartitions()))
print("Partitioner: {}".format(rdd.partitioner))
print("Partitions structure: {}".format(rdd.glom().collect()))

# train keras model



