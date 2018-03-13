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

import function as func


def get_loss(loss_function, output, label):
    '''
    get objective loss of model and backprograte to compute gradients
    some loss function not impelement
    '''
    if not isinstance(loss_function, str):
        raise TypeError('loss_function should be str object')

    if loss_function == 'binary_cross_entropy':
        loss = F.binary_cross_entropy(output, label)
    elif loss_function == 'poisson_nll_loss':
        loss = F.poisson_nll_loss(output, target)
    elif loss_function == 'cross_entropy':
        loss = F.cross_entropy(output, target)
    elif loss_function == 'hinge_embedding_loss':
        loss = F.hinge_embedding_loss(output, label)
    elif loss_function == 'margin_ranking_loss':
        loss = F.margin_ranking_loss(output, label)
    elif loss_function == 'multilabel_soft_margin_loss':
        loss = F.multilabel_soft_margin_loss(output, label)
    elif loss_function == 'multi_margin_loss':
        loss = F.multi_margin_loss(output, label)
    elif loss_function == 'nll_loss':
        loss = F.nll_loss(output, label)
    elif loss_function == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(output, label)

    return loss

def get_optimizer(optimizer, optimizer_config, params):
    '''
    get the optimizer of worker model
    '''
    if optimizer == 'SGD':
        p = ['lr', 'momentum', 'dampening', 'weight_decay', 'nesterov']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'nesterov':
                optimizer_config[i] = False
            else:
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], momentum=optimizer_config['momentum'],
                dampening=optimizer_config['dampening'], weight_decay=optimizer_config['weight_decay'],
                nesterov=optimizer_config['nesterov'])
        return opti

    elif optimizer=='Rprop':
        p = ['lr', 'etas', 'step_sizes']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1e-2
            elif i == 'etas':
                optimizer_config[i] = (0.5, 1.2)
            elif i =='step_sizes':
                optimizer_config[i] = (1e-6, 50)

        opti = optim.Rprop(params, lr=optimizer_config['lr'], etas=optimizer_config['etas'], step_sizes=optimizer_config['step_sizes'])
        return opti

    elif optimizer=='RMSprop':
        p = ['lr', 'alpha', 'eps', 'weight_decay', 'momentum', 'centered']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1e-2
            elif i == 'alpha':
                optimizer_config[i] = 0.99
            elif i == 'eps':
                optimizer_config[i] = 1e-8
            elif i == 'weight_decay':
                optimizer_config[i] = 0
            elif i == 'momentum':
                optimizer_config[i] = 0
            elif i == 'centered':
                optimizer_config[i] = False

        opti = optim.SGD(params, lr=optimizer_config['lr'], alpha=optimizer_config['alpha'],
                eps=optimizer_config['eps'], weight_decay=optimizer_config['weight_decay'],
                momentum=optimizer_config['momentum'], centered = optimizer_config['centered'])
        return opti

    elif optimizer=='LBFGS':
        p = ['lr', 'max_iter', 'max_eval', 'tolerance_grad', 'tolerance_change', 'history_size', 'line_search_fn']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1
            elif i == 'max_iter':
                optimizer_config[i] = 20
            elif i == 'max_eval':
                optimizer_config[i] = None
            elif i == 'tolerance_grad':
                optimizer_config[i] = 1e-5
            elif i == 'tolerance_change':
                optimizer_config[i] = 1e-9
            elif i == 'history_size':
                optimizer_config[i] = 100
            elif i == 'line_search_fn':
                optimizer_config[i] = None

        opti = optim.SGD(params, lr=optimizer_config['lr'], max_iter=optimizer_config['max_iter'],
                tolerance_grad=optimizer_config['tolerance_grad'], tolerance_change=optimizer_config['tolerance_change'],
                history_size=optimizer_config['history_size'], line_search_fn = optimizer_config['line_search_fn'])
        return opti

    elif optimizer=='ASGD':
        p = ['lr', 'lambd', 'alpha', 't0', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1e-2
            elif i == 'lambd':
                optimizer_config[i] = 1e-4
            elif i == 'alpha':
                optimizer_config[i] = 0.75
            elif i == 't0':
                optimizer_config[i] = 1e-6
            elif i == 'weight_decay':
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], lambd=optimizer_config['lambd'],
                alpha=optimizer_config['alpha'], t0=optimizer_config['t0'],
                weight_decay=optimizer_config['weight_decay'])
        return opti

    elif optimizer=='Adamax':
        p = ['lr', 'betas', 'eps', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 0.002
            elif i == 'betas':
                optimizer_config[i] = (0.9, 0.999)
            elif i == 'eps':
                optimizer_config[i] = 1e-08
            elif i == 'weight_decay':
                optimizer_config[i] = 0


        opti = optim.SGD(params, lr=optimizer_config['lr'], betas=optimizer_config['betas'],
                eps=optimizer_config['eps'], weight_decay=optimizer_config['weight_decay'])
        return opti

    elif optimizer=='SparseAdam':
        p = ['lr', 'betas', 'eps']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 0.001
            elif i == 'betas':
                optimizer_config[i] = (0.9, 0.999)
            elif i == 'eps':
                optimizer_config[i] = 1e-08

        opti = optim.SGD(params, lr=optimizer_config['lr'], betas=optimizer_config['betas'],
                eps=optimizer_config['eps'])
        return opti

    elif optimizer=='Adam':
        p = ['lr', 'betas', 'eps', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 0.001
            elif i == 'betas':
                optimizer_config[i] = (0.9, 0.999)
            elif i == 'eps':
                optimizer_config[i] = 1e-08
            elif i == 'weight_decay':
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], betas=optimizer_config['betas'],
                eps=optimizer_config['eps'], weight_decay=optimizer_config['weight_decay'])
        return opti

    elif optimizer=='Adagrad':
        p = ['lr', 'lr_decay', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 0.01
            elif i == 'lr_decay':
                optimizer_config[i] = 0
            elif i == 'weight_decay':
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], lr_decay=optimizer_config['lr_decay'],
                weight_decay=optimizer_config['weight_decay'])
        return opti

    elif optimizer=='Adadelta':
        p = ['lr', 'rho', 'eps', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1.0
            elif i == 'rho':
                optimizer_config[i] = 0.9
            elif i == 'eps':
                optimizer_config[i] = 1e-06
            elif i == 'weight_decay':
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], rho=optimizer_config['rho'],
                eps=optimizer_config['eps'], weight_decay=optimizer_config['weight_decay'])
        return opti

    else:
        raise ValueError('the optimizer is exactly the same as the original pytorch, please check again!')

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
        y_train = np.asarray([y for x, y in target_all)

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
                optimizer = get_optimizer(self.worker_optimizer, self.optimizer_config, model.parameters())
                model.train()
                for idx in range(batch_num):
                    data = x_train[idx*batch_size:min((idx+1)*batch_size, sample_num)]
                    target = y_train[idx*batch_size:min((idx+1)*batch_size, sample_num)]
                    data = Variable(data)
                    target = Variable(target)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = get_loss(self.loss_function, output, target)
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
                    optimizer = get_optimizer(self.worker_optimizer, self.optimizer_config, model.parameters())
                    model.train()
                    data = x_train[idx*batch_size: min((idx+1)*batch_size, sample_num)]
                    target = y_train[idx*batch_size: min((idx+1)*batch_size, sample_num)]
                    data = Variable(data)
                    target = Variable(target)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = get_loss(self.loss_function, output, target)
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
        epoch_num = self.train_config['epoch_num']
        batch_size = self.train_config['batch_size']
        sample_num = x_train.shape[0]
        batch_num = int(np.ceil(sample_num, float(batch_size)))

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
