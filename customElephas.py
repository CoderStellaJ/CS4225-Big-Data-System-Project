from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from itertools import tee
from keras.utils.generic_utils import slice_arrays
from keras.models import model_from_yaml
from keras.optimizers import get as get_optimizer
import pyspark
import h5py
import json
from keras.optimizers import serialize as serialize_optimizer
from keras.models import load_model

from elephas.utils import subtract_params
from elephas.utils import lp_to_simple_rdd
from elephas.utils import model_to_dict
from elephas.mllib import to_matrix, from_matrix, to_vector, from_vector
#from elephas.worker import AsynchronousSparkWorker, SparkWorker
from elephas.parameter import HttpServer, SocketServer
from elephas.parameter import HttpClient, SocketClient


class SparkWorker(object):
    """Synchronous Spark worker. This code will be executed on workers.
    """

    def __init__(self, yaml, parameters, train_config, master_optimizer,
                 master_loss, master_metrics, custom_objects):
        self.yaml = yaml
        self.parameters = parameters
        self.train_config = train_config
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics
        self.custom_objects = custom_objects
        self.model = None

    def train(self, data_iterator):
        """Train a keras model on a worker
        """
        optimizer = get_optimizer(self.master_optimizer)
        self.model = model_from_yaml(self.yaml, self.custom_objects)
        self.model.compile(optimizer=optimizer,
                           loss=self.master_loss, metrics=self.master_metrics)
        self.model.set_weights(self.parameters.value)

        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        self.model.compile(optimizer=self.master_optimizer,
                           loss=self.master_loss,
                           metrics=self.master_metrics)

        weights_before_training = self.model.get_weights()
        if x_train.shape[0] > self.train_config.get('batch_size'):
            self.model.fit(x_train, y_train, **self.train_config)
        weights_after_training = self.model.get_weights()
        deltas = subtract_params(
            weights_before_training, weights_after_training)
        yield deltas


class AsynchronousSparkWorker(object):
    """Asynchronous Spark worker. This code will be executed on workers.
    """

    def __init__(self, yaml, parameters, parameter_server_mode, train_config, frequency,
                 master_optimizer, master_loss, master_metrics, custom_objects):

        if parameter_server_mode == 'http':
            self.client = HttpClient()
        elif parameter_server_mode == 'socket':
            self.client = SocketClient()
        else:
            raise ValueError("Parameter server mode has to be either `http` or `socket`, "
                             "got {}".format(parameter_server_mode))

        self.train_config = train_config
        self.frequency = frequency
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics
        self.yaml = yaml
        self.parameters = parameters
        self.custom_objects = custom_objects
        self.model = None

    def train(self, data_iterator):
        """Train a keras model on a worker and send asynchronous updates
        to parameter server
        """
        print(data_iterator)
        print(">>> iterate")
        feature_iterator, label_iterator = tee(data_iterator, 2)

        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        if x_train.size == 0:
            return
        print(">>> get optimizer")
        optimizer = get_optimizer(self.master_optimizer)
        print(">>> get model")
        self.model = model_from_yaml(self.yaml, self.custom_objects)
        self.model.compile(optimizer=optimizer,
                           loss=self.master_loss, metrics=self.master_metrics)
        self.model.set_weights(self.parameters.value)

        epochs = self.train_config['epochs']
        batch_size = self.train_config.get('batch_size')
        nb_train_sample = x_train.shape[0]
        nb_batch = int(np.ceil(nb_train_sample / float(batch_size)))
        index_array = np.arange(nb_train_sample)
        batches = [
            (i * batch_size, min(nb_train_sample, (i + 1) * batch_size))
            for i in range(0, nb_batch)
        ]

        if self.frequency == 'epoch':
            for epoch in range(epochs):
                print("Epoch: ", epoch + 1)
                weights_before_training = self.client.get_parameters()
                self.model.set_weights(weights_before_training)
                self.train_config['epochs'] = 1
                if x_train.shape[0] > batch_size:
                    hist = self.model.fit(x_train, y_train, **self.train_config)
                    print(hist.history)

                self.train_config['epochs'] = epochs
                weights_after_training = self.model.get_weights()
                deltas = subtract_params(
                    weights_before_training, weights_after_training)
                self.client.update_parameters(deltas)
        elif self.frequency == 'batch':
            for epoch in range(epochs):
                if x_train.shape[0] > batch_size:
                    for (batch_start, batch_end) in batches:
                        weights_before_training = self.client.get_parameters()
                        self.model.set_weights(weights_before_training)
                        batch_ids = index_array[batch_start:batch_end]
                        x = slice_arrays(x_train, batch_ids)
                        y = slice_arrays(y_train, batch_ids)
                        self.model.train_on_batch(x, y)
                        weights_after_training = self.model.get_weights()
                        deltas = subtract_params(
                            weights_before_training, weights_after_training)
                        self.client.update_parameters(deltas)
        else:
            raise ValueError(
                'frequency parameter can be `epoch` or `batch, got {}'.format(self.frequency))
        yield []

class CustomSparkModel(object):

    def __init__(self, model, mode='asynchronous', frequency='epoch',  parameter_server_mode='http', num_workers=None,
                 custom_objects=None, batch_size=32,  port=4000, *args, **kwargs):
        """SparkModel
        Base class for distributed training on RDDs. Spark model takes a Keras
        model as master network, an optimization scheme, a parallelisation mode
        and an averaging frequency.
        :param model: Compiled Keras model
        :param mode: String, choose from `asynchronous`, `synchronous` and `hogwild`
        :param frequency: String, either `epoch` or `batch`
        :param parameter_server_mode: String, either `http` or `socket`
        :param num_workers: int, number of workers used for training (defaults to None)
        :param custom_objects: Keras custom objects
        :param batch_size: batch size used for training and inference
        :param port: port used in case of 'http' parameter server mode
        """

        self._master_network = model
        if not hasattr(model, "loss"):
            raise Exception(
                "Compile your Keras model before initializing an Elephas model with it")
        metrics = model.metrics
        loss = model.loss
        optimizer = serialize_optimizer(model.optimizer)

        if custom_objects is None:
            custom_objects = {}
        if metrics is None:
            metrics = ["accuracy"]
        self.mode = mode
        self.frequency = frequency
        self.num_workers = num_workers
        self.weights = self._master_network.get_weights()
        self.pickled_weights = None
        self.master_optimizer = optimizer
        self.master_loss = loss
        self.master_metrics = metrics
        self.custom_objects = custom_objects
        self.parameter_server_mode = parameter_server_mode
        self.batch_size = batch_size
        self.port = port
        self.kwargs = kwargs

        self.serialized_model = model_to_dict(model)
        if self.mode is not 'synchronous':
            if self.parameter_server_mode == 'http':
                self.parameter_server = HttpServer(
                    self.serialized_model, self.mode, self.port)
                self.client = HttpClient(self.port)
            elif self.parameter_server_mode == 'socket':
                self.parameter_server = SocketServer(self.serialized_model)
                self.client = SocketClient()
            else:
                raise ValueError("Parameter server mode has to be either `http` or `socket`, "
                                 "got {}".format(self.parameter_server_mode))

    @staticmethod
    def get_train_config(epochs, batch_size, verbose, validation_split):
        return {'epochs': epochs,
                'batch_size': batch_size,
                'verbose': verbose,
                'validation_split': validation_split}

    def get_config(self):
        base_config = {
            'parameter_server_mode': self.parameter_server_mode,
            'mode': self.mode,
            'frequency': self.frequency,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size}
        config = base_config.copy()
        config.update(self.kwargs)
        return config

    def save(self, file_name):
        model = self._master_network
        model.save(file_name)
        f = h5py.File(file_name, mode='a')

        f.attrs['distributed_config'] = json.dumps({
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }).encode('utf8')

        f.flush()
        f.close()

    @property
    def master_network(self):
        return self._master_network

    @master_network.setter
    def master_network(self, network):
        self._master_network = network

    def start_server(self):
        self.parameter_server.start()

    def stop_server(self):
        self.parameter_server.stop()

    def predict(self, data):
        """Get prediction probabilities for a numpy array of features
        """
        return self._master_network.predict(data)

    def predict_classes(self, data):
        """ Predict classes for a numpy array of features
        """
        return self._master_network.predict_classes(data)

    def fit(self, rdd, epochs=10, batch_size=32,
            verbose=0, validation_split=0.1):
        """
        Train an elephas model on an RDD. The Keras model configuration as specified
        in the elephas model is sent to Spark workers, abd each worker will be trained
        on their data partition.
        :param rdd: RDD with features and labels
        :param epochs: number of epochs used for training
        :param batch_size: batch size used for training
        :param verbose: logging verbosity level (0, 1 or 2)
        :param validation_split: percentage of data set aside for validation
        """
        print('>>> Fit model')
        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)

        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._fit(rdd, epochs, batch_size, verbose, validation_split)
        else:
            raise ValueError(
                "Choose from one of the modes: asynchronous, synchronous or hogwild")

    def _fit(self, rdd, epochs, batch_size, verbose, validation_split):
        """Protected train method to make wrapping of modes easier
        """
        self._master_network.compile(optimizer=self.master_optimizer,
                                     loss=self.master_loss,
                                     metrics=self.master_metrics)
        if self.mode in ['asynchronous', 'hogwild']:
            self.start_server()
        train_config = self.get_train_config(
            epochs, batch_size, verbose, validation_split)
        mode = self.parameter_server_mode
        freq = self.frequency
        optimizer = self.master_optimizer
        loss = self.master_loss
        metrics = self.master_metrics
        custom = self.custom_objects


        yaml = self._master_network.to_yaml()
        init = self._master_network.get_weights()
        parameters = rdd.context.broadcast(init)

        if self.mode in ['asynchronous', 'hogwild']:
            print('>>> Initialize workers')
            worker = AsynchronousSparkWorker(
                yaml, parameters, mode, train_config, freq, optimizer, loss, metrics, custom)
            print('>>> Distribute load')
            rdd.mapPartitions(worker.train).collect()
            print('>>> Async training complete.')
            new_parameters = self.client.get_parameters()
        elif self.mode == 'synchronous':
            worker = SparkWorker(yaml, parameters, train_config,
                                 optimizer, loss, metrics, custom)
            gradients = rdd.mapPartitions(worker.train).collect()
            new_parameters = self._master_network.get_weights()
            for grad in gradients:  # simply accumulate gradients one by one
                new_parameters = subtract_params(new_parameters, grad)
            print('>>> Synchronous training complete.')
        else:
            raise ValueError("Unsupported mode {}".format(self.mode))
        self._master_network.set_weights(new_parameters)
        if self.mode in ['asynchronous', 'hogwild']:
            self.stop_server()


def load_spark_model(file_name):
    model = load_model(file_name)
    f = h5py.File(file_name, mode='r')

    elephas_conf = json.loads(f.attrs.get('distributed_config'))
    class_name = elephas_conf.get('class_name')
    config = elephas_conf.get('config')
    if class_name == "SparkModel":
        return SparkModel(model=model, **config)
    elif class_name == "SparkMLlibModel":
        return SparkMLlibModel(model=model, **config)