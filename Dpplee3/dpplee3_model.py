'''
copyright@uestcliming
'''
from __future__ import absolute_import

import numpy as np
from itertools import tee
import socket
from multiprocessing import Process
import six.moves.cPickle as pickle
from six.moves import range
from flask import Flask, request
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

from rwlock import RWLock
import function
from param_server import ParamServer
from pg_weights import Get_post_weights
from worker import AsynchronousWorker, SynchronousWorker

class Dpplee3(object):
    def __init__(self, sc, model, server_optimizer, worker_optimizer, granularity, mode, worker_num):
        self.spark_context = sc
        self.master_network = model
        self.network = model
        self.mode = mode
        self.server_optimizer = server_optimizer
        self.worker_optimizer = worker_optimizer
        self.granularity = granularity
        self.mode = mode
        self.worker_num = worker_num

        self.lock = RWLock()

    @staticmethod
    def determine_master():
        '''
        Get URL of parameter server, running on master server
        '''
        master_url = socket.gethostbyname(socket.gethostname()) + ':5000'
        return master_url

    def network2serial(model):
        '''
        serialize the model network
        '''
        return pickle.dumps(model)

    def serial2network(serialized_model):
        '''
        get network from serialized data
        '''
        return pickle.loads(serialize_model)


    def train(self, rdd, epoch, batch_size):
        '''
        Distributed train using spark
        plan to add tensorboard function
        '''
        rdd = rdd.repartition(self.worker_num)
        master_url = self.determine_master()
        get_post = Get_post_state_dict(master_url)
        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._train(rdd, self.epoch, batch_size, master_url, get_post)

    def _train(self, rdd, epoch, batch_size, master_url, get_post):
        '''
        Wrap train method
        '''
        serialized_network = network2serial(self.network)

        if self.mode in ['asynchronous', 'hogwild']:
            '''start a Flask web service to handle asynchronous parameter server'''
            self.paramserver = ParamServer(master_network, self.mode, self.master_optimizer, self.lock)
            self.paramserver.start_server()

            worker = AsynchronousWorker(
                serialized_network, self.frequency, master_url,
                self.master_optimizer, self.master_loss, self.master_metrics)
            rdd.mapPartitions(worker.train).collect()
            new_state_dict = get_post.get_server_state_dict()

        elif self.mode == 'synchronous':
            '''don't need asynchronous parameter server'''
            '''state_dict need serialize or not'''
            state_dict = self.master_network.state_dict()
            state_dict = self.spark_context.broadcast(state_dict)
            worker = SparkWorker(serialized_network, state_dict, train_config)
            deltas = rdd.mapPartitions(worker.train).collect()
            new_state_dict = self.master_network.state_dict()
            for delta in deltas:
                constraints = self.master_network.constraints
                new_parameters = self.optimizer.get_updates(self.weights, delta)

        self.master_network.load_state_dict(new_state_dict)

        if self.mode in ['asynchronous', 'hogwild']:
            self.paramserver.stop_server()
