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

from Dpplee3.rwlock import RWLock
#import function
from Dpplee3.param_server import ParamServer
from Dpplee3.pg_weights import Get_post_state_dict
from Dpplee3.worker import AsynchronousWorker, SynchronousWorker

class Dpplee3(object):
    def __init__(self, sc, model, frequency, server_optimizer, worker_optimizer, loss_function, granularity, mode, worker_num):
        self.spark_context = sc
        self.master_network = model
        self.network = model
        self.frequency = frequency
        self.mode = mode
        self.server_optimizer = server_optimizer
        self.worker_optimizer = worker_optimizer
        self.loss_function = loss_function
        self.granularity = granularity  #grain 
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

    def network2serial(self, model):
        '''
        serialize the model network
        '''
        return pickle.dumps(model,-1)

    def serial2network(self, serialized_model):
        '''
        get network from serialized data
        '''
        return pickle.loads(serialize_model)

    def get_worker_train_config(self, nb_epoch, batch_size):
        train_config = {}
        train_config['epoch'] = nb_epoch
        train_config['batch_size'] = batch_size
        
        return train_config

    def get_worker_optimizer_config(self):
        optimizer_config = {}
        return optimizer_config



    def train(self, rdd, epoch, batch_size):
        '''
        Distributed train using spark
        plan to add tensorboard function
        '''
        rdd = rdd.repartition(self.worker_num)
        master_url = self.determine_master()
        print(master_url)
        get_post = Get_post_state_dict(master_url)
        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._train(rdd, epoch, batch_size, master_url, get_post)

    def _train(self, rdd, epoch, batch_size, master_url, get_post):
        '''
        Wrap train method
        '''
        serialized_network = self.network2serial(self.network)
        print(type(serialized_network))

        if self.mode in ['asynchronous', 'hogwild']:
            '''start a Flask web service to handle asynchronous parameter server'''
            self.paramserver = ParamServer(self.master_network, self.mode, self.server_optimizer, self.lock)
            self.paramserver.start_server()

            train_config = self.get_worker_train_config(epoch, batch_size)
            optimizer_config = self.get_worker_optimizer_config()

            worker = AsynchronousWorker(
                self.network, self.frequency, master_url,
                self.worker_optimizer, train_config, optimizer_config, self.loss_function)

            rdd.mapPartitions(worker.train).collect()
            print('11111111111111111111111111')
            new_state_dict = get_post.get_server_state_dict()

        elif self.mode == 'synchronous':
            '''don't need asynchronous parameter server'''
            '''state_dict need serialize or not'''
            state_dict = self.master_network.state_dict()
            state_dict = self.spark_context.broadcast(state_dict)
            train_config = self.get_worker_train_config(epoch, batch_size)
            optimizer_config = self.get_worker_optimizer_config()

            worker = SynchronousWorker(serialized_network, state_dict, self.worker_optimizer, train_config, optimizer_config, self.loss_function)
            updates = rdd.mapPartitions(worker.train).collect()
            new_state_dict = self.master_network.state_dict()
            for delta in deltas:
                constraints = self.master_network.constraints
                new_parameters = self.optimizer.get_updates(self.weights, delta)

        self.master_network.load_state_dict(new_state_dict)

        if self.mode in ['asynchronous', 'hogwild']:
            self.paramserver.stop_server()
