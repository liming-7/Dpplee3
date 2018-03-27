'''
copyright@uestcliming
'''
from __future__ import absolute_import

import numpy as np
from collections import OrderedDict
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

from .rwlock import RWLock
#import function
# from Dpplee3.param_server import ParamServer
from .pg_weights import Get_post_state_dict
# from Dpplee3.worker import AsynchronousWorker, SynchronousWorker

from .function import compute_updates, get_loss, get_optimizer
from .optimizer import SGD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math


class Dpplee3_model(object):
    def __init__(self, sc, model, frequency, server_optimizer, worker_optimizer, loss_function, mode, worker_num, frequency_num=1, save_path = None):
        self.spark_context = sc
        self.master_network = model
        self.state_dict = model.state_dict()
        self.network = model
        self.frequency = frequency
        self.mode = mode
        self.worker_optimizer = worker_optimizer
        self.loss_function = loss_function
        self.mode = mode
        self.worker_num = worker_num
        self.frequency_num = frequency_num  #grain
        self.lock = RWLock()
        # torch.save(model, 'model.pkl')
        # print('----model save------')
        if save_path:
            self.save_path = save_path

    def set_server_optimizer(self, server_optimizer, **arg):
        ''''''
        self.server_optimizer = server_optimizer
        server_optimizer_config = arg
        self.optimizer = get_optimizer(self.server_optimizer, server_optimizer_config, self.master_network.parameters())
        return server_optimizer_config

    def set_worker_optimizer(self, worker_optimizer, **arg):
        ''''''
        self.worker_optimizer = worker_optimizer
        self.worker_optimizer_config = arg
        return self.worker_optimizer_config

    def enable_tensorboard(self):
        pass

    def start_server(self):
        ''' Start parameter server'''
        self.server = Process(target=self.service)
        self.server.start()
        print('-----------start server-------------')

    def stop_server(self):
        ''' Terminate parameter server'''
        self.server.terminate()
        self.server.join()
        print('-----------stop server--------------')

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

        # @app.route('/model', methods=['GET'])
        # def get_model_structure()

        @app.route('/worker_updates', methods=['POST'])
        def update_parameters():
            updates = pickle.loads(request.data) #get the update infomation from workers
            if self.mode == 'asynchronous':
                self.lock.acquire_write()

            self.master_network.train() #set the training or not training, is responsiable for batchnorm and dropout
            # self.optimizer = SGD(self.master_network.parameters(), lr=0.001, momentum=0.5)
            self.optimizer.zero_grad()
            self.optimizer.replace_grad(updates)
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


    def train(self, rdd, epoch, batch_size):
        '''
        Distributed train using spark
        plan to add tensorboard function
        '''
        rdd = rdd.repartition(self.worker_num)
        master_url = self.determine_master()
        # print(master_url)
        # torch.save(self.network, 'hdfs://Master:9000/model.pkl')
        print('--------START TRAINING--------')
        # print('store pkl')
        #model = torch.load('hdfs://192.168.0.104:9000/model.pkl')
        # print(model)

        get_post = Get_post_state_dict(master_url)
        # print("ACM!")
        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._train(rdd, epoch, batch_size, master_url, get_post)

    def _train(self, rdd, epoch, batch_size, master_url, get_post):
        '''
        Wrap train method
        '''
        # serialized_network = self.network2serial(self.network)
        # print('----ggffgg',len(serialized_network))

        if self.mode in ['asynchronous', 'hogwild']:
            '''start a Flask web service to handle asynchronous parameter server'''
            # self.paramserver = ParamServer(self.master_network, self.mode, self.server_optimizer, self.lock)
            self.start_server()

            train_config = self.get_worker_train_config(epoch, batch_size)
            optimizer_config = self.worker_optimizer_config

            worker = AsynchronousWorker(
                self.network, self.frequency, master_url,
                self.worker_optimizer, train_config, optimizer_config, self.loss_function, self.frequency_num)
            print(worker)
            rdd.mapPartitions(worker.train).collect()
            print('------------------train done------------------')
            # new_state_dict = get_post.get_server_state_dict()

        elif self.mode == 'synchronous':
            '''don't need asynchronous parameter server'''
            state_dict = self.master_network.state_dict()
            state_dict = self.spark_context.broadcast(state_dict)
            train_config = self.get_worker_train_config(epoch, batch_size)
            optimizer_config = self.worker_optimizer_config

            worker = SynchronousWorker(self.network, state_dict, self.worker_optimizer, train_config, optimizer_config, self.loss_function)
            updates = rdd.mapPartitions(worker.train).collect()
            # self.master_network.train()
            for update in updates:
                self.master_network.train()
                self.optimizer.zero_grad()
                self.optimizer.replace_grad(update)
                self.optimizer.step()

        torch.save(self.master_network, self.save_path)

        if self.mode in ['asynchronous', 'hogwild']:
            self.stop_server()


class AsynchronousWorker(object):
    '''
    distribute to spark worker by mapPartitions, works on spark worker
    '''
    def __init__(self, network, frequency, master_url, worker_optimizer, train_config, optimizer_config, loss_function, frequency_num):
        self.network = network
        self.frequency = frequency
        self.frequency_num = frequency_num  #control the grain of the training procedure.
        self.master_url = master_url
        self.worker_optimizer = worker_optimizer
        self.train_config = train_config
        self.optimizer_config = optimizer_config
        self.get_post = Get_post_state_dict(master_url)
        self.loss_function = loss_function
        print('----worker initialing----')

    def train(self, data_iterator):
        '''
        Train a pytorch model on a worker and send asynchronous updates
        to parameter server
        '''
        print(self.master_url)
        print(self.optimizer_config)
        data_all, target_all = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in data_all])
        y_train = np.asarray([y for x, y in target_all])
        # print(self.frequency)
        # print('-------worker open----')
        # f = open('model.pkl', 'wb')
        # print(len(self.serialized_network))
        # f.write(self.serialized_network)
        # f.close()
        # print('-----close f')
        # print(self.serialized_network.state_dict())
        # print('`````````'

        if x_train.size == 0:
            return
        # print('picke load model')
        # model = pickle.loads(self.serialized_network)
        # print('picke load model hhh')
        # model = torch.load('model.pkl')
        # model = nn.Sequential(OrderedDict([
        #   ('conv1', nn.Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))),
        #   ('conv2', nn.Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))),
        #   ('conv2_drop', nn.Dropout2d(p=0.5)),
        #   ('fc1', nn.Linear(in_features=320, out_features=50, bias=True)),
        #   ('fc2', nn.Linear(in_features=50, out_features=10, bias=True))

        # ]))
        model = self.network
        epoch_num = self.train_config['epoch']
        batch_size = self.train_config['batch_size']
        sample_num = x_train.shape[0]
        batch_num = int(np.ceil(sample_num/batch_size))-5

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model.cuda()
        '''grained of updates, frequency_num controls more concise grain of asyn training, leave for future work.'''
        cnt = 0
        if self.frequency == 'epoch':
            for epoch in range(epoch_num):
                state_dict_before_training = self.get_post.get_server_state_dict()
                print('get_server_state_dict')
                model.load_state_dict(state_dict_before_training)
                optimizer = get_optimizer(self.worker_optimizer, self.optimizer_config, model.parameters())
                model.train()
                for idx in range(batch_num):
                    data = x_train[idx*batch_size : min((idx+1)*batch_size, sample_num)]
                    target = y_train[idx*batch_size : min((idx+1)*batch_size, sample_num)]
                    # print(target)
                    # print(type(target))
                    if use_gpu:
                        data = torch.from_numpy(data).cuda()
                        target = torch.from_numpy(target).cuda()

                    data = Variable(torch.from_numpy(data))
                    target = Variable(torch.from_numpy(target))
                    # print(data.size())
                    # print(target.size())
                    optimizer.zero_grad()
                    # print(optimizer)
                    output = model(data)
                    # print(output)
                    # print(target)
                    loss = get_loss(self.loss_function, output, target, use_gpu)
                    # loss = F.nll_loss(output, target)
                    # print(idx, '     ',loss)
                    loss.backward()
                    optimizer.step()
                    # optimizer.zero_grad()

                cnt = cnt+1
                if cnt == self.frequency_num:
                    eval_output = model(data)
                    eval_loss = get_loss(self.loss_function, eval_output, target)
                    print(epoch, '------------', eval_loss)
                    state_dict_after_training = model.state_dict()
                    # print(state_dict_after_training, 'AAAAAAAAAAAAAAAAAAAAAAAAA')
                    updates = compute_updates(state_dict_before_training, state_dict_after_training)
                    # print(updates, 'update delta to parameter server~~')
                    self.get_post.post_updates_to_server(updates)
                    cnt = 0
        elif self.frequency == 'batch':
            for epoch in range(epoch_num):
                for idx in range(batch_num):
                    state_dict_before_training = self.get_post.get_server_state_dict()
                    model.load_state_dict(state_dict_before_training)
                    optimizer = get_optimizer(self.worker_optimizer, self.optimizer_config, model.parameters())
                    model.train()
                    data = x_train[idx*batch_size: min((idx+1)*batch_size, sample_num)]
                    target = y_train[idx*batch_size: min((idx+1)*batch_size, sample_num)]
                    if use_gpu:
                        data = torch.from_numpy(data).cuda()
                        target = torch.from_numpy(target).cuda()
                    data = Variable(torch.Tensor(data))
                    target = Variable(torch.Tensor(target))
                    optimizer.zero_grad()
                    output = model(data)
                    loss = get_loss(self.loss_function, output, target)
                    loss.backward()
                    optimizer.step()
                    cnt = cnt+1
                    if cnt == self.frequency_num:
                        eval_output = model(data)
                        eval_loss = get_loss(self.loss_function, eval_output, target)
                        print(epoch, '------------', eval_loss)
                        state_dict_after_training = model.state_dict()
                        updates = compute_updates(state_dict_before_training, state_dict_after_training)
                        self.get_post.post_updates_to_server(updates)
                        cnt = 0
        else:
            print ('please choose the frequency of training')

        yield []


class SynchronousWorker(object):
    '''
    distribute to spark worker by mapPartitions, works on spark worker
    '''
    def __init__(self, network, state_dict_before_training, worker_optimizer, train_config, optimizer_config, loss_function):
        self.network = network
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
        # model = pickle.loads(self.serialized_network)
        # model = self.serialized_network
        epoch_num = self.train_config['epoch']
        batch_size = self.train_config['batch_size']
        sample_num = x_train.shape[0]
        batch_num = int(math.ceil(sample_num/float(batch_size)))-5

        model = self.network
        model.load_state_dict(self.state_dict)
        optimizer = get_optimizer(self.worker_optimizer, self.optimizer_config, model.parameters())
        model.train()
        for idx in range(batch_num):
            optimizer.zero_grad()

            data = x_train[idx*batch_size:min((idx+1)*batch_size, sample_num)]
            target = y_train[idx*batch_size:min((idx+1)*batch_size, sample_num)]
            data = Variable(torch.from_numpy(data))
            target = Variable(torch.from_numpy(target))

            output = model(data)
            loss = get_loss(self.loss_function, output, target)
            loss.backward()
            optimizer.step()

        state_dict_after_training = model.state_dict()
        updates = compute_updates(state_dict_before_training, state_dict_after_training)

        yield updates
