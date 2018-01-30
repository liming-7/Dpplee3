'''
define the worker function
'''
from __future__ import absolute_import


class AsynchronousWorker(object):
    '''
    distribute to spark worker by mapPartitions, works on spark worker
    '''
    def __init__(serialized_network, frequency, master_url, master_optimizer, train_config, optimizer_config):
        self.serialized_network = serialized_network
        self.frequency = frequency
        self.master_url = master_url
        self.master_optimizer = master_optimizer
        self.train_config = train_config
        self.optimizer_config = optimizer_config

    def train(self, data_iterator):
        '''
        Train a keras model on a worker and send asynchronous updates
        to parameter server
        '''



class SynchronousWorker(object):
    '''
    distribute to spark worker by mapPartitions, works on spark worker
    '''
    def __init__():
        pass

    def train():
        pass
