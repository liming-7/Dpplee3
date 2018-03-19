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
# from Dpplee3.param_server import ParamServer
from Dpplee3.pg_weights import Get_post_state_dict
from Dpplee3.worker import AsynchronousWorker, SynchronousWorker

def get_server_weights(master_url='localhost:5000'):
    '''
    Retrieve master weights from parameter server
    '''
    request = urllib2.Request('http://{0}/parameters'.format(master_url),
                              headers={'Content-Type': 'application/Dpplee3'})
    ret = urllib2.urlopen(request).read()
    weights = pickle.loads(ret)
    return weights


def put_deltas_to_server(delta, master_url='localhost:5000'):
    '''
    Update master parameters with deltas from training process
    '''
    request = urllib2.Request('http://{0}/update'.format(master_url),
                              pickle.dumps(delta, -1), headers={'Content-Type': 'application/Dpplee3'})
    return urllib2.urlopen(request).read()

class Dpplee3(object):
    def __init__(self, sc, model, frequency, server_optimizer, worker_optimizer, loss_function, granularity, mode, worker_num):
        self.spark_context = sc
        self.master_network = model
        self.state_dict = model.state_dict()
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

    def start_server(self):
        ''' Start parameter server'''
        self.server = Process(target=self.service)
        self.server.start()

    def stop_server(self):
        ''' Terminate parameter server'''
        self.server.terminate()
        self.server.join()

    def service(self):
        ''' Define service and run flask app'''
        app = Flask(__name__)
        self.app = app

        @app.route('/')
        def home():
            return 'Dpplee3'

        @app.route('/parameters', methods=['GET'])
        def get_parameters():
            if self.mode == 'asynchronous':
                self.lock.acquire_read()
            self.pickled_state_dict = pickle.dumps(self.state_dict, -1)
            pickled_state_dict = self.pickled_state_dict
            if self.mode == 'asynchronous':
                self.lock.release()
            return pickled_state_dict

        @app.route('/worker_updates', methods=['POST'])
        def update_parameters():
            delta = pickle.loads(request.data) #get the update infomation from workers
            if self.mode == 'asynchronous':
                self.lock.acquire_write()

            self.master_network.train() #set the training or not training, is useful for batchnorm and dropout
            self.optimizer = optimizer.SGD(self.master_network.parameters(), lr=0.001, momentum=0.5)
            self.optimizer.zero_grad()
            self.optimizer.replace_grad(delta)
            self.optimizer.step()
            self.state_dict = self.master_network.state_dict()

            if self.mode == 'asynchronous':
                self.lock.release()

            return 'Update done'

        self.app.run(host='0.0.0.0', debug=True,
                     threaded=True, use_reloader=False)


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
            # self.paramserver = ParamServer(self.master_network, self.mode, self.server_optimizer, self.lock)
            self.start_server()

            train_config = self.get_worker_ train_config(epoch, batch_size)
            optimizer_config = self.get_worker_optimizer_config()

            worker = AsynchronousWorker(
                serialized_network, self.frequency, master_url,
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
