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

class Dpplee3:
    def __init__(self, sc, model, server_optimizer, worker_optimizer, granularity, mode, worker_num):
        self.spark_context = sc
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

    def train(self, rdd, epoch, batch_size):
        '''
        Distributed train using spark
        plan to add tensorboard function
        '''
        rdd = rdd.repartition(self.worker_num)
        master_url = self.determine_master()

        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._train(rdd, epoch, batch_size, master_url)

    def _train(self, rdd, epoch, batch_size, master_url):
        '''
        Wrap train method
        '''
        if self.mode in ['asynchronous', 'hogwild']:
            '''start a Flask web service to handle asynchronous parameter server'''
            self.paramserver = ParamServer(self.network, self.mode, self.master_optimizer, self.lock)
            self.paramserver.start_server()
            worker = AsynchronousWorker(
                yaml, train_config, self.frequency, master_url,
                self.master_optimizer, self.master_loss, self.master_metrics, self.custom_objects
            )
            rdd.mapPartitions(worker.train).collect()
            new_parameters = get_server_weights(master_url)
        elif self.mode == 'synchronous':
            init = self.master_network.get_weights()
            parameters = self.spark_context.broadcast(init)
            worker = SparkWorker(yaml, parameters, train_config)
            deltas = rdd.mapPartitions(worker.train).collect()
            new_parameters = self.master_network.get_weights()
            for delta in deltas:
                constraints = self.master_network.constraints
                new_parameters = self.optimizer.get_updates(self.weights, constraints, delta)
        self.master_network.set_weights(new_parameters)

        if self.mode in ['asynchronous', 'hogwild']:
            self.paramserver.stop_server()
