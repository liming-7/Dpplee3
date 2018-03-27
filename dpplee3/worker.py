'''
define the worker function
'''
from __future__ import absolute_import
import six.moves.cPickle as pickle
from six.moves import range

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable

from itertools import tee
import numpy as np
import math

import Dpplee3.function as func
from Dpplee3.pg_weights import Get_post_state_dict




class AsynchronousWorker(object):
    '''
    distribute to spark worker by mapPartitions, works on spark worker
    '''
    def __init__(self, serialized_network, frequency, master_url, worker_optimizer, train_config, optimizer_config, loss_function =None, frequency_num=1):
        self.serialized_network = serialized_network
        self.frequency = frequency
        self.frequency_num = frequency_num  #control the grain of the training procedure.
        self.master_url = master_url
        self.worker_optimizer = worker_optimizer
        self.train_config = train_config
        self.optimizer_config = optimizer_config
        self.get_post = Get_post_state_dict(master_url)
        self.loss_function = loss_function


    def train(self, data_iterator):
        '''
        Train a pytorch model on a worker and send asynchronous updates
        to parameter server
        '''
        data_all, target_all = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in data_all])
        y_train = np.asarray([y for x, y in target_all])

        if x_train.size == 0:
            return

        model = pickle.loads(serialized_network)
        epoch_num = self.train_config['epoch']
        batch_size = self.train_config['batch_size']
        sample_num = x_train.shape[0]
        batch_num = int(np.ceil(sample_num/float(batch_size)))

        '''grained of updates, frequency_num controls more concise grain of asyn training, leave for future work.'''
        if self.frequency == 'epoch':
            for epoch in range(epoch_num):
                state_dict_before_training = self.get_post.get_server_state_dict()
                model.load_state_dict(state_dict_before_training)
                optimizer = func.get_optimizer(self.worker_optimizer, self.optimizer_config, model.parameters())
                model.train()
                for idx in range(batch_num):
                    data = x_train[idx*batch_size:min((idx+1)*batch_size, sample_num)]
                    target = y_train[idx*batch_size:min((idx+1)*batch_size, sample_num)]
                    data = Variable(data)
                    target = Variable(target)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = func.get_loss(self.loss_function, output, target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                state_dict_after_training = model.state_dict()
                updates = func.compute_updates(state_dict_before_training, state_dict_after_training)
                self.get_post.post_updates_to_server(updates)
        elif self.frequency == 'batch':
            for epoch in range(epoch_num):
                for idx in range(batch_num):
                    state_dict_before_training = self.get_post.get_server_state_dict()
                    model.load_state_dict(state_dict_before_training)
                    optimizer = func.get_optimizer(self.worker_optimizer, self.optimizer_config, model.parameters())
                    model.train()
                    data = x_train[idx*batch_size: min((idx+1)*batch_size, sample_num)]
                    target = y_train[idx*batch_size: min((idx+1)*batch_size, sample_num)]
                    data = Variable(data)
                    target = Variable(target)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = func.get_loss(self.loss_function, output, target)
                    loss.backward()
                    optimizer.step()
                    state_dict_after_training = model.state_dict()
                    updates = func.compute_updates(state_dict_before_training, state_dict_after_training)
                    self.get_post.post_updates_to_server(updates)
        else:
            print ('please choose the frequency of training')

        yield []


class SynchronousWorker(object):
    '''
    distribute to spark worker by mapPartitions, works on spark worker
    '''
    def __init__(self, serialized_network, state_dict_before_training, worker_optimizer, train_config, optimizer_config, loss_function):
        self.serialized_network = serialized_network
        self.state_dict_before_training = state_dict_before_training
        self.optimizer = worker_optimizer
        self.train_config = train_config
        self.optimizer_config = optimizer_config
        self.loss_function = loss_function

    def train(self, data_iterator):
        '''
        train a pytorch model and post the updates to the master
        grain epoch only, because fine-grained hard to impeletation
        '''
        data_all, target_all = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in data_all])
        y_train = np.asarray([y for x, y in target_all])

        if x_train.shape[0] == 0:
            return

        model = pickle.loads(self.serialized_network)
        # model = self.serialized_network
        epoch_num = self.train_config['epoch']
        batch_size = self.train_config['batch_size']
        sample_num = x_train.shape[0]
        batch_num = int(math.ceil(sample_num/float(batch_size)))
        yield (type(self.state_dict_before_training))

        model.load_state_dict(self.state_dict)
        optimizer = get_optimizer(self.worker_optimizer, self.optimizer_config, model.parameters())
        model.train()
        for idx in range(batch_num):
            optimizer.zero_grad()

            data = x_train[idx*batch_size:min((idx+1)*batch_size, sample_num)]
            target = y_train[idx*batch_size:min((idx+1)*batch_size, sample_num)]
            data = Variable(data)
            target = Variable(target)

            output = model(data)
            loss = get_loss(self.loss_function, output, target)
            loss.backward()
            optimizer.step()

        state_dict_after_training = model.state_dict()
        updates = compute_updates(state_dict_before_training, state_dict_after_training)

        yield updates
